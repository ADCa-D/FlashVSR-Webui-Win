# -*- coding: utf-8 -*-
"""看 NVFP4 / MXFP8 的 GEMM 到底调了什么 CUDA 内核（判断是不是真 FP4 张量核）。"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import comfy_kitchen as ck  # noqa: E402

M, K, N = 2304, 8960, 1536
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.02


def top_kernels(fn, tag, n=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
    torch.cuda.synchronize()
    print(f"\n=== {tag} ===")
    rows = []
    for e in prof.key_averages():
        if e.device_time_total > 0:
            rows.append((e.device_time_total / n / 1000.0, e.key))
    rows.sort(reverse=True)
    for ms, k in rows[:6]:
        print(f"  {ms:7.3f} ms  {k[:88]}")


def main():
    wq8, wb8 = ck.quantize_mxfp8(w)
    xq8, xb8 = ck.quantize_mxfp8(x)
    top_kernels(lambda: ck.scaled_mm_mxfp8(xq8, wq8, xb8, wb8, out_dtype=torch.bfloat16), "MXFP8")

    ts = (w.abs().amax() / (6.0 * 448.0)).clamp(min=1e-8).float().reshape(1)
    wq4, wb4 = ck.quantize_nvfp4(w, ts, pad_16x=True)
    tx = (x.abs().amax() / (6.0 * 448.0)).clamp(min=1e-8).float().reshape(1)
    xq4, xb4 = ck.quantize_nvfp4(x, tx, pad_16x=True)
    top_kernels(lambda: ck.scaled_mm_nvfp4(xq4, wq4, tx, ts, xb4, wb4, out_dtype=torch.bfloat16),
                "NVFP4")

    top_kernels(lambda: torch.nn.functional.linear(x, w), "bf16(F.linear)")


if __name__ == "__main__":
    main()
