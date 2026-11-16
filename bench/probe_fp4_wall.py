# -*- coding: utf-8 -*-
"""NVFP4 GEMM：设备时间 vs wall time 的差异来源。"""
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


def device(fn, n=30):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    tot = sum(e.device_time_total for e in prof.key_averages())
    return tot / n / 1000.0


def main():
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.02
    ts = (w.abs().amax() / (6.0 * 448.0)).clamp(min=1e-8).float().reshape(1)
    wq, wb = ck.quantize_nvfp4(w, ts, pad_16x=True)
    xq, xb = ck.quantize_nvfp4(x, ts, pad_16x=True)
    print("xq", xq.shape, xq.dtype, "| xb", xb.shape, "| wq", wq.shape, wb.shape, flush=True)

    f = lambda: ck.scaled_mm_nvfp4(xq, wq, ts, ts, xb, wb, out_dtype=torch.bfloat16)
    y = f()
    print("y", y.shape, y.dtype, flush=True)
    print(f"wall   {wall(f):6.3f} ms", flush=True)
    print(f"device {device(f):6.3f} ms", flush=True)

    # 对比：直接把 M 档拉大，看是否只是启动/同步开销
    for MM in (512, 2304, 9216):
        xx = torch.randn(MM, K, device="cuda", dtype=torch.bfloat16) * 0.02
        xq2, xb2 = ck.quantize_nvfp4(xx, ts, pad_16x=True)
        g = lambda: ck.scaled_mm_nvfp4(xq2, wq, ts, ts, xb2, wb, out_dtype=torch.bfloat16)
        print(f"M={MM:5d}  wall {wall(g):6.3f} ms  device {device(g):6.3f} ms", flush=True)


if __name__ == "__main__":
    main()
