# -*- coding: utf-8 -*-
"""在真实形状上对比：自研 sparse_sageattn（近全掩码） vs SDPA(flash/mem-efficient)。

做法：跑一步 DiT，把第一次 sparse_sageattn 的 (q,k,v,mask) 抓下来，
然后在同一组张量上分别计时 + 比数值。
"""
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
            grabbed["q"] = q.detach().clone()
            grabbed["k"] = k.detach().clone()
            grabbed["v"] = v.detach().clone()
            grabbed["mask"] = kw.get("mask_id").detach().clone()
            grabbed["kw"] = {kk: vv for kk, vv in kw.items() if kk != "mask_id"}
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
    nb = len(dit.blocks)
    with torch.no_grad():
        pl.model_fn_wan_video(dit, x, ts, None, LQ_latents=None, is_stream=True,
                              pre_cache_k=[None] * nb, pre_cache_v=[None] * nb,
                              topk_ratio=22.5, kv_ratio=2.0, cur_process_idx=0,
                              t_mod=t_mod, t=t, local_range=9)
    wan.sparse_sageattn = orig
    pl.model_fn_wan_video.__globals__["sparse_sageattn"] = orig

    q, k, v, mask = grabbed["q"], grabbed["k"], grabbed["v"], grabbed["mask"]
    print("q", tuple(q.shape), "k", tuple(k.shape), "mask", tuple(mask.shape),
          "mask 密度 %.4f" % float(mask.float().mean()))

    with torch.no_grad():
        ms_sparse = bench(lambda: orig(q, k, v, **grabbed["kw"]))

        def sdpa():
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        ms_sdpa = bench(sdpa)

        # 数值对比（同一份 q/k/v）
        a = orig(q, k, v, **grabbed["kw"]).float()
        b = sdpa().float()
        denom = a.norm().clamp_min(1e-6)
        print("sparse_sageattn  %.3f ms" % ms_sparse)
        print("SDPA             %.3f ms   (%.2fx)" % (ms_sdpa, ms_sparse / ms_sdpa))
        print("两者 相对误差 %.2e   maxdiff %.4f   均值 |a|=%.3f |b|=%.3f"
              % ((a - b).norm() / denom, (a - b).abs().max(), a.abs().mean(), b.abs().mean()))

    # 空掩码（NaN 风险）也测一下，看是不是真的会炸
    with torch.no_grad():
        empty = torch.zeros_like(mask)
        try:
            out = orig(q, k, v, **{**grabbed["kw"], "mask_id": empty})
            print("空 mask 结果: nan=%d  inf=%d  |mean|=%.4f"
                  % (int(torch.isnan(out).sum()), int(torch.isinf(out).sum()),
                     float(out.float().abs().mean())))
        except Exception as e:
            print("空 mask 直接报错:", type(e).__name__, str(e)[:100])


if __name__ == "__main__":
    main()
