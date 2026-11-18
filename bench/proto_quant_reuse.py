# -*- coding: utf-8 -*-
"""A 的原型：quantize_mxfp8 复用缓冲。

要证明两件事：
  1. 复用缓冲的输出与原实现 **逐位相同**（fp8 字节 + scale 字节）
  2. 单次调用的 CPU 时间确实下降（原实现每次 empty+zeros+3 次 DLPack 包装）

前提（必须先验证）：这些 shape 的 scale 没有 padding 尾巴，会被 _C 全量覆写。
  scale_rows = roundup(rows,128)  scale_cols = roundup(cols//32,4)
"""
import importlib
import sys
import time

import torch

ck = importlib.import_module("comfy_kitchen")
ck_cuda = importlib.import_module("comfy_kitchen.backends.cuda")


def roundup(v, m):
    return (v + m - 1) // m * m


BUF = {}


def quant_reuse(x):
    assert x.is_contiguous()
    rows, cols = x.shape
    if rows % 32 or cols % 32:
        return ck.quantize_mxfp8(x)          # 不满足对齐 → 原路
    key = (rows, cols)
    hit = BUF.get(key)
    if hit is None:
        qx = torch.empty((rows, cols), device=x.device, dtype=torch.float8_e4m3fn,
                         memory_format=torch.contiguous_format)
        sx = torch.zeros((roundup(rows, 128), roundup(cols // 32, 4)),
                         device=x.device, dtype=torch.uint8)
        hit = BUF[key] = (qx, sx)
    qx, sx_u8 = hit
    stream = torch.cuda.current_stream(x.device).cuda_stream
    ck_cuda._C.quantize_mxfp8(ck_cuda._wrap_for_dlpack(x),
                              ck_cuda._wrap_for_dlpack(qx),
                              ck_cuda._wrap_for_dlpack(sx_u8),
                              False, stream)
    return qx, sx_u8.view(torch.float8_e8m0fnu)


def bench(fn, n=300, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n
    torch.cuda.synchronize()
    return dt * 1e6      # µs


def main():
    print("shape            padding?  逐位相同  原实现 µs   复用 µs   降幅")
    for shape in [(1536, 1536), (1536, 8960), (512, 1536), (1536, 4608)]:
        x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        a_q, a_s = ck.quantize_mxfp8(x)
        b_q, b_s = quant_reuse(x)
        same = (a_q.view(torch.uint8).equal(b_q.view(torch.uint8)) and
                a_s.view(torch.uint8).equal(b_s.view(torch.uint8)))
        pad = (roundup(shape[1] // 32, 4) != shape[1] // 32) or (roundup(shape[0], 128) != shape[0])
        t_old = bench(lambda: ck.quantize_mxfp8(x))
        t_new = bench(lambda: quant_reuse(x))
        print("%-16s %-9s %-9s %8.1f %8.1f  %+.0f%%"
              % ("%dx%d" % shape, "有" if pad else "无", "是" if same else "**否**",
                 t_old, t_new, 100 * (t_new / t_old - 1)))

    # 复用缓冲在"连续两次不同 shape"时是否仍正确（模拟真实交替）
    x1 = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    x2 = torch.randn(1536, 8960, device="cuda", dtype=torch.bfloat16)
    ok = True
    for _ in range(3):
        b1q, b1s = quant_reuse(x1)
        b2q, b2s = quant_reuse(x2)
        r1q, r1s = ck.quantize_mxfp8(x1)
        r2q, r2s = ck.quantize_mxfp8(x2)
        ok &= b1q.view(torch.uint8).equal(r1q.view(torch.uint8))
        ok &= b2q.view(torch.uint8).equal(r2q.view(torch.uint8))
    print("交替 shape 复用仍逐位相同:", ok)


if __name__ == "__main__":
    main()
