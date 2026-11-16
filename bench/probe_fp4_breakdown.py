# -*- coding: utf-8 -*-
"""拆解 NVFP4 单层开销：amax / 激活量化 / GEMM 各占多少。"""
import sys
import time

import torch

sys.path.insert(0, r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web")
import comfy_kitchen as ck  # noqa: E402

M, K, N = 2304, 8960, 1536


def bench(fn, n=20, warm=5):
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

    t_amax = bench(lambda: (x.abs().amax() / (6.0 * 448.0)).clamp(min=1e-8).float().reshape(1))
    t_qx = bench(lambda: ck.quantize_nvfp4(x, ts, pad_16x=True))
    xq, xb = ck.quantize_nvfp4(x, ts, pad_16x=True)
    t_gemm = bench(lambda: ck.scaled_mm_nvfp4(xq, wq, ts, ts, xb, wb, out_dtype=torch.bfloat16))

    wq8, wb8 = ck.quantize_mxfp8(w)
    t_qx8 = bench(lambda: ck.quantize_mxfp8(x))
    xq8, xb8 = ck.quantize_mxfp8(x)
    t_gemm8 = bench(lambda: ck.scaled_mm_mxfp8(xq8, wq8, xb8, wb8, out_dtype=torch.bfloat16))

    print(f"NVFP4  amax {t_amax:6.3f} ms | 激活量化 {t_qx:6.3f} ms | GEMM {t_gemm:6.3f} ms "
          f"| 合计 {t_amax + t_qx + t_gemm:6.3f} ms", flush=True)
    print(f"MXFP8  激活量化 {t_qx8:6.3f} ms | GEMM {t_gemm8:6.3f} ms "
          f"| 合计 {t_qx8 + t_gemm8:6.3f} ms", flush=True)


if __name__ == "__main__":
    main()
