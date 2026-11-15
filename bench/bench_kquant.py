# -*- coding: utf-8 -*-
"""GGUF Q4_K / Q6_K 的 torch dequant 实现 + 单层 GEMM 速度对比。

说明：gguf-py 只有 dequantize，没有 K-quant 的 quantize（NotImplementedError）。
本脚本用**合成块**验证 dequant 布局是否与 gguf 参考一致（数学等价性），
再按真实形状测速度——速度只取决于布局与形状，与具体数值无关。

用法：python bench/bench_kquant.py
"""
import os
import sys
import time

import numpy as np
import torch
from gguf.constants import GGMLQuantizationType as QT
from gguf.quants import dequantize as gguf_dequant

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import comfy_kitchen as ck  # noqa: E402

Q4K_B = 144     # Q4_K 每 256 权重 144 字节


def dequant_q4k(buf, n):
    nb = buf.shape[0]
    d = buf[:, 0:2].contiguous().view(torch.float16).view(-1).float()
    dmin = buf[:, 2:4].contiguous().view(torch.float16).view(-1).float()
    sc = buf[:, 4:16]
    qs = buf[:, 16:144].reshape(nb, 8, 16)
    scales = torch.empty(nb, 8, device=buf.device, dtype=torch.float32)
    mins = torch.empty(nb, 8, device=buf.device, dtype=torch.float32)
    for j in range(8):
        if j < 4:
            scales[:, j] = (sc[:, j] & 63).float()
            mins[:, j] = (sc[:, j + 4] & 63).float()
        else:
            scales[:, j] = ((sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4)).float()
            mins[:, j] = ((sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)).float()
    # 真正的布局：128 字节 = 4 段 × 32 字节；每段先 32 个低 nibble（scale 2c），
    # 再 32 个高 nibble（scale 2c+1）
    qs4 = buf[:, 16:144].reshape(nb, 4, 32)
    lo = (qs4 & 0xF).to(torch.float32)
    hi = ((qs4 >> 4) & 0xF).to(torch.float32)
    q = torch.stack([lo, hi], dim=2).reshape(nb, 4, 64)          # [低32, 高32]
    sc2 = scales.view(nb, 4, 2).repeat_interleave(32, dim=2)     # (nb,4,64)
    mi2 = mins.view(nb, 4, 2).repeat_interleave(32, dim=2)
    y = d.view(nb, 1, 1) * sc2 * q - dmin.view(nb, 1, 1) * mi2
    return y.reshape(-1)[:n]


Q4K_DEQ = dequant_q4k


def main():
    OUT_F, IN_F = 1536, 8960
    n = OUT_F * IN_F
    nb = n // 256

    torch.manual_seed(0)
    # 合成 Q4_K 块：d/dmin 用合理 fp16，其余随机
    raw = np.random.randint(0, 256, size=(nb, Q4K_B), dtype=np.uint8)
    d = np.float16(0.002)
    dmin = np.float16(0.001)
    for i in range(nb):
        raw[i, 0:2] = np.frombuffer(d.tobytes(), dtype=np.uint8)
        raw[i, 2:4] = np.frombuffer(dmin.tobytes(), dtype=np.uint8)
    ref = gguf_dequant(raw.reshape(-1), QT.Q4_K).astype(np.float32)
    buf = torch.from_numpy(raw.copy()).cuda().view(-1, Q4K_B)
    with torch.no_grad():
        got = Q4K_DEQ(buf, n).cpu().numpy()
    md = float(np.abs(got - ref).max())
    print(f"Q4_K 布局校验：与 gguf 参考 maxdiff = {md:.3g}  "
          f"（相对 {md / (np.abs(ref).max() + 1e-9) * 100:.2f}%）", flush=True)
    print(f"packed 大小 = {nb * Q4K_B / 2**20:.2f} MiB  "
          f"(bf16 {n * 2 / 2**20:.1f} MiB)", flush=True)

    w = (torch.randn(OUT_F, IN_F, device="cuda", dtype=torch.bfloat16) * 0.02)
    for M in (2304, 4608):
        x = torch.randn(M, IN_F, device="cuda", dtype=torch.bfloat16) * 0.02

        def bench(fn, n=10, warm=3):
            for _ in range(warm):
                fn()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t) / n * 1000

        with torch.no_grad():
            t_bf = bench(lambda: torch.nn.functional.linear(x, w))
            wq, wb = ck.quantize_mxfp8(w)
            xq, xb = ck.quantize_mxfp8(x.reshape(-1, IN_F).contiguous())
            t_fp8 = bench(lambda: ck.scaled_mm_mxfp8(xq, wq, xb, wb, out_dtype=torch.bfloat16))
            t_dq = bench(lambda: Q4K_DEQ(buf, n), n=20)
            t_tot = bench(lambda: torch.nn.functional.linear(
                x, Q4K_DEQ(buf, n).reshape(OUT_F, IN_F).to(torch.bfloat16)))
            print(f"\nM={M}", flush=True)
            print(f"  bf16         {t_bf:6.2f} ms", flush=True)
            print(f"  MXFP8        {t_fp8:6.2f} ms  (bf16/fp8 = {t_bf/t_fp8:.2f}×)", flush=True)
            print(f"  Q4_K dequant {t_dq:6.2f} ms（纯反量化）", flush=True)
            print(f"  Q4_K 合计    {t_tot:6.2f} ms  (vs bf16 {t_bf/t_tot:.2f}×, "
                  f"vs fp8 {t_fp8/t_tot:.2f}×)", flush=True)


if __name__ == "__main__":
    main()
