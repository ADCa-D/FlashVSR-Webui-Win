# -*- coding: utf-8 -*-
"""真实形状下把「能用到的 attention 后端」全跑一遍。

对照组（sd.cpp 在没有 flash-attn 时的做法＝3 个 tensor-core GEMM + softmax）
也一起测，因为它常常在消费卡上打败融合 kernel。

形状来自真实前向：q/k/v [1,12,1536,128]，block mask [1,12,12,24]（密度 98.96%）。
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import engine as E  # noqa: E402

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    HAVE_SDPA_CTX = True
except Exception:
    HAVE_SDPA_CTX = False


def bench(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="none")
    pipe = eng.pipe
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    wan = sys.modules["flashvsr_node.src.models.wan_video_dit"]
    pl = sys.modules["flashvsr_node.src.pipelines.flashvsr_tiny_long"]

    grabbed = {}
    orig = wan.sparse_sageattn

    def spy(q, k, v, **kw):
        if not grabbed:
            grabbed.update(q=q.detach().clone(), k=k.detach().clone(),
                           v=v.detach().clone(), mask=kw.get("mask_id").detach().clone(),
                           kw={kk: vv for kk, vv in kw.items() if kk != "mask_id"})
        return orig(q, k, v, **kw)

    wan.sparse_sageattn = spy
    pl.model_fn_wan_video.__globals__["sparse_sageattn"] = spy
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 16, 6, 32, 32, generator=g).to(dtype=torch.bfloat16, device="cuda")
    ts = torch.tensor([500.0], dtype=torch.bfloat16, device="cuda")
    dit.reinit_cross_kv(pipe.prompt_emb_posi["context"])
    with torch.no_grad():
        t = dit.time_embedding(wan.sinusoidal_embedding_1d(dit.freq_dim, ts))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        pl.model_fn_wan_video(dit, x, ts, None, LQ_latents=None, is_stream=True,
                              pre_cache_k=[None] * len(dit.blocks),
                              pre_cache_v=[None] * len(dit.blocks),
                              topk_ratio=22.5, kv_ratio=2.0, cur_process_idx=0,
                              t_mod=t_mod, t=t, local_range=9)
    wan.sparse_sageattn = orig
    pl.model_fn_wan_video.__globals__["sparse_sageattn"] = orig

    q, k, v, mask = grabbed["q"], grabbed["k"], grabbed["v"], grabbed["mask"]
    print("q %s  mask %s  密度 %.4f" % (tuple(q.shape), tuple(mask.shape), float(mask.float().mean())))
    head_dim = q.shape[-1]
    scale = head_dim ** -0.5

    # 把 block mask 展开成元素级加性 mask（0 / -inf），供 dense 后端使用
    def build_add_mask():
        qb = q.shape[-2] // mask.shape[-2]      # q 侧 block = 128 token
        kb = k.shape[-2] // mask.shape[-1]      # kv 侧 block = 64 token（sage 把 K 拆两半）
        m = mask.to(torch.bool)
        m = m.repeat_interleave(qb, dim=-2).repeat_interleave(kb, dim=-1)
        add = torch.zeros(m.shape, dtype=q.dtype, device=q.device)
        add.masked_fill_(~m, float("-inf"))
        return add

    add_mask = build_add_mask()
    print("加性 mask %s  显存 %.1f MB" % (tuple(add_mask.shape), add_mask.numel() * 2 / 2 ** 20))

    results = {}

    with torch.no_grad():
        results["sparse_sageattn(现状)"] = bench(lambda: orig(q, k, v, **grabbed["kw"]))

        results["SDPA 自动选择"] = bench(lambda: F.scaled_dot_product_attention(q, k, v))

        if HAVE_SDPA_CTX:
            for name, be in (("flash", SDPBackend.FLASH_ATTENTION),
                             ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                             ("cudnn", SDPBackend.CUDNN_ATTENTION),
                             ("math", SDPBackend.MATH)):
                try:
                    def run(be=be):
                        with sdpa_kernel(be):
                            return F.scaled_dot_product_attention(q, k, v)
                    results["SDPA 强制 %s" % name] = bench(run)
                except Exception as e:
                    results["SDPA 强制 %s" % name] = float("nan")
                    print("  %s 不可用: %s" % (name, str(e)[:60]))

        # sd.cpp 的做法：3 个 GEMM + softmax（tensor core 满负荷）
        def three_gemm():
            a = torch.matmul(q, k.transpose(-1, -2)) * scale
            a = torch.softmax(a, dim=-1)
            return torch.matmul(a, v)

        results["3×GEMM+softmax (sd.cpp)"] = bench(three_gemm)

        def three_gemm_masked():
            a = torch.matmul(q, k.transpose(-1, -2)) * scale
            a = torch.softmax(a + add_mask, dim=-1)
            return torch.matmul(a, v)

        results["3×GEMM+mask (sd.cpp)"] = bench(three_gemm_masked)

    base = results["sparse_sageattn(现状)"]
    print("\n%-30s %8s %8s" % ("后端", "ms", "相对现状"))
    for name, ms in sorted(results.items(), key=lambda kv: (kv[1] != kv[1], kv[1])):
        if ms != ms:
            print("%-30s   n/a" % name)
        else:
            print("%-30s %8.3f %7.2fx" % (name, ms, base / ms))

    # 数值：以 3×GEMM 为参考（数学上最准），看现状偏多少
    with torch.no_grad():
        ref = three_gemm().float()
        cur = orig(q, k, v, **grabbed["kw"]).float()
        print("\n现状 vs 3×GEMM 参考: 相对误差 %.2e  maxdiff %.4f"
              % ((cur - ref).norm() / ref.norm().clamp_min(1e-6), (cur - ref).abs().max()))


if __name__ == "__main__":
    main()
