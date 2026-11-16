# -*- coding: utf-8 -*-
"""干净对照：fp4 vs fp8 的"每层实际开销"（scale 只在循环外算一次）。

三种口径：
  A. 裸 GEMM（权重、激活都已量化好）
  B. 生产口径（激活每次量化 + GEMM；scale 用标定常数，不算 amax）
  C. 引擎现状（每层 x.abs().amax() 重算 scale）—— 用来量化那 0.5 ms 的浪费
"""
import time

import torch
import comfy_kitchen as ck

K, N = 8960, 1536


def wall(fn, n=30, warm=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def ts(t):
    return (t.abs().amax() / (6.0 * 448.0)).clamp(min=1e-8).float().reshape(1)


w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
tw = ts(w)
wq4, wb4 = ck.quantize_nvfp4(w, tw, pad_16x=True)
wq8, wb8 = ck.quantize_mxfp8(w)

print(f"K={K} N={N}   单位 ms（越低越好）")
hdr = (f"{'M':>6s} {'fp4裸':>7s} {'fp8裸':>7s} {'裸比':>5s} | "
       f"{'fp4生产':>8s} {'fp8生产':>8s} {'生产比':>6s} | {'fp4现状':>8s}")
print(hdr)

for M in (256, 576, 1152, 2304, 4608):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
    tx = ts(x)                                  # 标定：只算一次
    xq4, xb4 = ck.quantize_nvfp4(x, tx, pad_16x=True)
    xq8, xb8 = ck.quantize_mxfp8(x)

    # A 裸 GEMM
    t4a = wall(lambda: ck.scaled_mm_nvfp4(xq4, wq4, tx, tw, xb4, wb4, out_dtype=torch.bfloat16))
    t8a = wall(lambda: ck.scaled_mm_mxfp8(xq8, wq8, xb8, wb8, out_dtype=torch.bfloat16))

    # B 生产口径：量化 + GEMM（scale 复用）
    def fp4_prod():
        q, b = ck.quantize_nvfp4(x, tx, pad_16x=True)
        return ck.scaled_mm_nvfp4(q, wq4, tx, tw, b, wb4, out_dtype=torch.bfloat16)

    def fp8_prod():
        q, b = ck.quantize_mxfp8(x)
        return ck.scaled_mm_mxfp8(q, wq8, b, wb8, out_dtype=torch.bfloat16)

    t4b, t8b = wall(fp4_prod), wall(fp8_prod)

    # C 引擎现状：每层重算 amax
    def fp4_cur():
        t = ts(x)
        q, b = ck.quantize_nvfp4(x, t, pad_16x=True)
        return ck.scaled_mm_nvfp4(q, wq4, t, tw, b, wb4, out_dtype=torch.bfloat16)

    t4c = wall(fp4_cur)

    print(f"{M:6d} {t4a:7.3f} {t8a:7.3f} {t8a/t4a:5.2f} | "
          f"{t4b:8.3f} {t8b:8.3f} {t8b/t4b:6.2f} | {t4c:8.3f}")
    del x, xq4, xb4, xq8, xb8
    torch.cuda.empty_cache()
