# -*- coding: utf-8 -*-
"""参考 llama.cpp / sdcpp 的现成做法，评估"im2col + fp8 GEMM"能否替代 cuDNN bf16 卷积。

背景：ggml 里 conv2d = im2col + mul_mat，量化权重靠 MMQ 内核（dequant 在 matmul 内完成）。
我们没有 ggml，但有 torch._scaled_mm（cublasLt fp8，实测 ~60 TFLOPS），
所以只要 im2col 的搬运不贵，就能走同一路线而不用自己写 Triton MMA。

测：3×3 s1 p1，NHWC，fp8(e4m3) 激活 + fp8 权重，per-tensor scale，fp32 累加。
"""
import sys
import time

import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True


def bench(fn, warm=3, iters=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def im2col_nhwc(x, k=3):
    """(N,H,W,C) → (N*H*W, C*k*k)，按 tap 顺序与权重对齐。"""
    n, h, w, c = x.shape
    p = k // 2
    xp = F.pad(x, (0, 0, p, p, p, p))                  # pad H/W
    cols = []
    for dy in range(k):
        for dx in range(k):
            cols.append(xp[:, dy:dy + h, dx:dx + w, :])
    return torch.stack(cols, dim=1).reshape(n * h * w, -1)


SHAPES = [
    ("8x512x128x128", (8, 512, 128, 128), 512),
    ("8x256x256x256", (8, 256, 256, 256), 256),
    ("8x128x256x256", (8, 128, 256, 256), 128),
]

print(f"{'形状':16s} {'cudnn bf16':>11s} {'im2col+fp8':>11s} {'加速':>6s} {'备注'}")
for tag, (n, c, h, w), cout in SHAPES:
    x_bf = torch.randn(n, h, w, c, device="cuda", dtype=torch.bfloat16)
    wt = torch.randn(cout, c, 3, 3, device="cuda", dtype=torch.bfloat16)

    # cuDNN bf16（NCHW 是 PyTorch 的主路径）
    x_nchw = x_bf.permute(0, 3, 1, 2).contiguous()
    t_bf = bench(lambda: F.conv2d(x_nchw, wt, padding=1))

    # im2col + fp8 GEMM
    xq = (x_bf.float() / x_bf.abs().max().float()).to(torch.float8_e4m3fn)
    wq = (wt.reshape(cout, -1).t().contiguous() / wt.abs().max().float()).to(torch.float8_e4m3fn)
    sa = torch.tensor([x_bf.abs().max().item()], device="cuda")
    sb = torch.tensor([wt.abs().max().item()], device="cuda")

    def fp8_path():
        cols = im2col_nhwc(xq)                                  # (M, C*9) fp8
        out = torch._scaled_mm(cols, wq, sa, sb, out_dtype=torch.bfloat16)
        return out

    try:
        t_fp8 = bench(fp8_path)
        note = ""
        ok = True
    except Exception as e:  # noqa
        t_fp8, ok = float("nan"), False
        note = f"{type(e).__name__}: {str(e)[:60]}"

    if ok:
        print(f"{tag:16s} {t_bf*1e3:9.2f}ms {t_fp8*1e3:9.2f}ms {t_bf/t_fp8:5.2f}x  峰值理论上限 1.68x")
    else:
        print(f"{tag:16s} {t_bf*1e3:9.2f}ms {'FAIL':>11s} {'-':>6s}  {note}")
    del x_bf, x_nchw, wt, xq, wq
    torch.cuda.empty_cache()
