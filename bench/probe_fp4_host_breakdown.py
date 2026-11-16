# -*- coding: utf-8 -*-
"""逐段拆解 fp4/fp8 每次调用的耗时（GPU vs host），两种 M（大 tile / 小 tile）。

修正上一版：④ 的 lambda 解包写错了，这版分开写清楚。
"""
import time

import torch
import comfy_kitchen as ck


def wall(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def gpu_ms(fn, n=30, warm=5):
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


def report(tag, fn):
    w, g = wall(fn), gpu_ms(fn)
    print(f"  {tag:28s} wall {w:6.2f}  GPU {g:6.2f}  host {w - g:6.2f}")


for M in (2304, 576):
    K, N = 8960, 1536
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    print(f"\n=== M={M} K={K} N={N} ===")
    print(f"{'步骤':28s} {'wall':>7s} {'GPU':>7s} {'host':>7s}")

    # fp4
    def ts(t):
        return (t.abs().amax() / (6.0 * 448.0)).clamp(min=1e-8).float().reshape(1)

    ta, tw = ts(x), ts(w)
    xq, xb = ck.quantize_nvfp4(x, ta, pad_16x=True)
    wq, wb = ck.quantize_nvfp4(w, tw, pad_16x=True)

    report("NVFP4 ts(amax)", lambda: ts(x))
    report("NVFP4 quantize(x)", lambda: ck.quantize_nvfp4(x, ta, pad_16x=True))
    report("NVFP4 mm(预算好)", lambda: ck.scaled_mm_nvfp4(xq, wq, ta, tw, xb, wb,
                                                         out_dtype=torch.bfloat16))

    def fp4_full():
        t = ts(x)
        q, b = ck.quantize_nvfp4(x, t, pad_16x=True)
        return ck.scaled_mm_nvfp4(q, wq, t, tw, b, wb, out_dtype=torch.bfloat16)

    report("NVFP4 全流程（引擎实际）", fp4_full)

    # fp8
    xq8, xb8 = ck.quantize_mxfp8(x)
    wq8, wb8 = ck.quantize_mxfp8(w)
    report("MXFP8 quantize(x)", lambda: ck.quantize_mxfp8(x))
    report("MXFP8 mm(预算好)", lambda: ck.scaled_mm_mxfp8(xq8, wq8, xb8, wb8,
                                                          out_dtype=torch.bfloat16))

    def fp8_full():
        q, b = ck.quantize_mxfp8(x)
        return ck.scaled_mm_mxfp8(q, wq8, b, wb8, out_dtype=torch.bfloat16)

    report("MXFP8 全流程（引擎实际）", fp8_full)
    report("bf16 matmul（参照）", lambda: x @ w.t())
