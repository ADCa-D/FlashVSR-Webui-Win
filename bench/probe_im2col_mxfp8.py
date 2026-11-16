# -*- coding: utf-8 -*-
"""走 sdcpp 同款设计（im2col + 量化 GEMM），但复用我们已有的 comfy_kitchen MXFP8 内核。

这样不需要写任何 Triton MMA：im2col → ck.quantize_mxfp8 → ck.scaled_mm_mxfp8。
对比 cuDNN bf16 卷积；同时给出数值误差（与 fp32/bf16 卷积的 max/mean 差）。
"""
import time

import torch
import torch.nn.functional as F
import comfy_kitchen as ck

torch.backends.cudnn.benchmark = True


def bench(fn, warm=3, iters=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def im2col_nhwc(x, k=3):
    n, h, w, c = x.shape
    p = k // 2
    xp = F.pad(x, (0, 0, p, p, p, p))
    cols = [xp[:, dy:dy + h, dx:dx + w, :] for dy in range(k) for dx in range(k)]
    return torch.stack(cols, dim=1).reshape(n * h * w, -1)


SHAPES = [
    ("8x512x128x128", (8, 512, 128, 128), 512),
    ("8x256x256x256", (8, 256, 256, 256), 256),
    ("8x128x256x256", (8, 128, 256, 256), 128),
]

print(f"{'形状':16s} {'cudnn bf16':>11s} {'im2col+MXFP8':>13s} {'加速':>6s} {'数值差(mean/max)':>18s}")
for tag, (n, c, h, w), cout in SHAPES:
    x = torch.randn(n, h, w, c, device="cuda", dtype=torch.bfloat16) * 0.5
    wt = torch.randn(cout, c, 3, 3, device="cuda", dtype=torch.bfloat16) * 0.05
    x_nchw = x.permute(0, 3, 1, 2).contiguous()

    ref = F.conv2d(x_nchw, wt, padding=1).permute(0, 2, 3, 1).reshape(-1, cout)
    t_bf = bench(lambda: F.conv2d(x_nchw, wt, padding=1))

    # scaled_mm_mxfp8 的 mat_b 是 (N, K)（和 DiT 里的 nn.Linear 权重同构），不要转置
    w_mat = wt.reshape(cout, -1).contiguous()
    wq, wb = ck.quantize_mxfp8(w_mat)                      # 每 32 元素 e8m0

    def fp8_path():
        cols = im2col_nhwc(x)                              # (M, K) bf16
        xq, xb = ck.quantize_mxfp8(cols)
        return ck.scaled_mm_mxfp8(xq, wq, xb, wb, out_dtype=torch.bfloat16)

    try:
        out = fp8_path()
        t_fp8 = bench(fp8_path)
        d = (out.float() - ref.float()).abs()
        print(f"{tag:16s} {t_bf*1e3:9.2f}ms {t_fp8*1e3:11.2f}ms {t_bf/t_fp8:5.2f}x "
              f"{d.mean():8.5f}/{d.max():8.4f}")
    except Exception as e:  # noqa
        print(f"{tag:16s} {t_bf*1e3:9.2f}ms FAIL {type(e).__name__}: {str(e)[:70]}")
    del x, x_nchw, wt, w_mat, wq, wb
    torch.cuda.empty_cache()
