# -*- coding: utf-8 -*-
"""NVFP4 GEMM 的固定开销能否被 CUDA Graph 吃掉。"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import comfy_kitchen as ck  # noqa: E402

M, K, N = 2304, 8960, 1536


def wall(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.02
    ts = (w.abs().amax() / (6.0 * 448.0)).clamp(min=1e-8).float().reshape(1)
    wq, wb = ck.quantize_nvfp4(w, ts, pad_16x=True)
    xq, xb = ck.quantize_nvfp4(x, ts, pad_16x=True)

    f = lambda: ck.scaled_mm_nvfp4(xq, wq, ts, ts, xb, wb, out_dtype=torch.bfloat16)
    print(f"eager wall {wall(f):6.3f} ms", flush=True)

    g = torch.cuda.CUDAGraph()
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                f()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            y = f()
        torch.cuda.synchronize()
        t = wall(lambda: g.replay(), n=50)
        print(f"graph replay {t:6.3f} ms  → 固定开销{'被吃掉' if t < 1.0 else '仍在'}", flush=True)
        print("输出形状", y.shape, y.dtype, flush=True)
    except Exception as e:  # noqa
        print("CUDA Graph 失败:", type(e).__name__, str(e)[:200], flush=True)


if __name__ == "__main__":
    main()
