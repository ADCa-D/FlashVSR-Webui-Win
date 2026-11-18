# -*- coding: utf-8 -*-
"""SageAttention vs 现状 sparse_sageattn：真实形状直接对拍。"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

import engine as E  # noqa: E402


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
    from sageattention import sageattn
    print("sageattn:", sageattn)

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
            grabbed.update(q=q.detach().clone(), k=k.detach().clone(), v=v.detach().clone(),
                           mask=kw.get("mask_id").detach().clone(),
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
                              pre_cache_k=[None] * len(dit.blocks), pre_cache_v=[None] * len(dit.blocks),
                              topk_ratio=22.5, kv_ratio=2.0, cur_process_idx=0,
                              t_mod=t_mod, t=t, local_range=9)
    wan.sparse_sageattn = orig
    pl.model_fn_wan_video.__globals__["sparse_sageattn"] = orig

    q, k, v, mask = grabbed["q"], grabbed["k"], grabbed["v"], grabbed["mask"]
    print("q %s mask %s 密度 %.4f" % (tuple(q.shape), tuple(mask.shape), float(mask.float().mean())))

    with torch.no_grad():
        try:
            out_sage = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
            print("sageattn 首次调用 OK:", tuple(out_sage.shape), out_sage.dtype)
        except Exception as e:
            print("sageattn 调用失败:", type(e).__name__, str(e)[:300])
            return

        ms_sparse = bench(lambda: orig(q, k, v, **grabbed["kw"]))
        ms_sage = bench(lambda: sageattn(q, k, v, tensor_layout="HND", is_causal=False))

        ref = (torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)).softmax(-1)
        ref = torch.matmul(ref, v).float()
        a = orig(q, k, v, **grabbed["kw"]).float()
        b = sageattn(q, k, v, tensor_layout="HND", is_causal=False).float()

        print("\nsparse_sageattn : %.3f ms" % ms_sparse)
        print("sageattn        : %.3f ms   (%.2fx)" % (ms_sage, ms_sparse / ms_sage))
        print("相对精确 softmax 的误差: 现状 %.2e   sage %.2e"
              % ((a - ref).norm() / ref.norm(), (b - ref).norm() / ref.norm()))
        print("sage vs 现状: 相对 %.2e  maxdiff %.4f"
              % ((a - b).norm() / a.norm().clamp_min(1e-6), (a - b).abs().max()))


if __name__ == "__main__":
    main()
