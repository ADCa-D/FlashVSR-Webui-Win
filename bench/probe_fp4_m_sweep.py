# -*- coding: utf-8 -*-
"""M 扫描：fp4 / fp8 GEMM 何时才比对方快？（我们的分块推理 M 偏小）

只测 mm 本身（权重预算好），避免量化/归约干扰。
"""
import time

import torch
import comfy_kitchen as ck

K, N = 8960, 1536


def gpu_ms(fn, n=30, warm=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def ts(t):
    return (t.abs().amax() / (6.0 * 448.0)).clamp(min=1e-8).float().reshape(1)


w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
wq4, wb4 = ck.quantize_nvfp4(w, ts(w), pad_16x=True)
wq8, wb8 = ck.quantize_mxfp8(w)
tw = ts(w)

print(f"K={K} N={N}  权重已预量化；只比 GEMM 本身（GPU 时间，ms）")
print(f"{'M':>6s} {'bf16':>7s} {'MXFP8':>7s} {'NVFP4':>7s}   {'fp8/bf16':>8s} {'fp4/fp8':>8s}")
for M in (256, 576, 1152, 2304, 4608, 9216, 18432):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
    tbf = gpu_ms(lambda: x @ w.t())
    xq8, xb8 = ck.quantize_mxfp8(x)
    t8 = gpu_ms(lambda: ck.scaled_mm_mxfp8(xq8, wq8, xb8, wb8, out_dtype=torch.bfloat16))
    xq4, xb4 = ck.quantize_nvfp4(x, ts(x), pad_16x=True)
    t4 = gpu_ms(lambda: ck.scaled_mm_nvfp4(xq4, wq4, ts(x), tw, xb4, wb4,
                                           out_dtype=torch.bfloat16))
    print(f"{M:6d} {tbf:7.3f} {t8:7.3f} {t4:7.3f}   {tbf/t8:8.2f} {t8/t4:8.2f}")
    del x, xq8, xb8, xq4, xb4
    torch.cuda.empty_cache()
