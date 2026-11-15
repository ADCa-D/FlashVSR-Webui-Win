# -*- coding: utf-8 -*-
"""对比 VAE 解码器里几层卷积的三种实现：
  A. cuDNN（benchmark=True，让它自己挑，可能选 DIRECT/WINOGRAD/IMPLICIT_GEMM）
  B. torch.compile（inductor，Triton 模板，本质是 direct 风格）
  C. im2col + GEMM（手工实现，等价于 implicit GEMM 的数学形式）
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

# TCDecoder 里真实出现的大 conv 形状：(N, C_in, H, W, C_out)
SHAPES = [
    ("latent 512ch @64²", 1, 512, 64, 64, 512),
    ("mid 256ch @128²", 1, 256, 128, 128, 256),
    ("mid2 128ch @256²", 1, 128, 256, 256, 128),
    ("final 128ch @512²", 1, 128, 512, 512, 128),
]


def bench(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000  # ms


def compiled_conv(x, w, b):
    return torch.compile(lambda a, k, c: F.conv2d(a, k, c, padding=1), dynamic=False)(x, w, b)


def im2col_conv(x, w, b):
    n, c, h, wd = x.shape
    cols = F.unfold(x, 3, padding=1)                    # (N, C*9, H*W)
    wm = w.reshape(w.shape[0], -1)                      # (Cout, C*9)
    out = wm @ cols                                     # (N, Cout, H*W)
    out = out.reshape(n, w.shape[0], h, wd)
    return out + b.view(1, -1, 1, 1)


def main():
    dev, dt = "cuda", torch.bfloat16
    print(f"{'layer':22s} {'cudnn(ms)':>10s} {'compile(ms)':>12s} {'im2col(ms)':>11s}  {'best':>8s}", flush=True)
    for name, n, cin, h, wd, cout in SHAPES:
        x = torch.randn(n, cin, h, wd, device=dev, dtype=dt)
        w = torch.randn(cout, cin, 3, 3, device=dev, dtype=dt) * 0.02
        b = torch.zeros(cout, device=dev, dtype=dt)
        try:
            t_cudnn = bench(lambda: F.conv2d(x, w, b, padding=1))
        except Exception as e:
            t_cudnn = float("nan")
            print("cudnn fail", e)
        try:
            cf = torch.compile(lambda a, k, c: F.conv2d(a, k, c, padding=1), dynamic=False)
            t_comp = bench(lambda: cf(x, w, b))
        except Exception as e:
            t_comp = float("nan")
            print("compile fail:", str(e)[:80])
        try:
            t_col = bench(lambda: im2col_conv(x, w, b))
        except Exception:
            t_col = float("nan")
        vals = {"cudnn": t_cudnn, "compile": t_comp, "im2col": t_col}
        best = min((v, k) for k, v in vals.items() if v == v)
        print(f"{name:22s} {t_cudnn:10.2f} {t_comp:12.2f} {t_col:11.2f}  {best[1]:>8s}", flush=True)

    # 数值一致性（im2col vs cudnn）
    x = torch.randn(1, 128, 64, 64, device=dev, dtype=dt)
    w = torch.randn(128, 128, 3, 3, device=dev, dtype=dt) * 0.02
    b = torch.zeros(128, device=dev, dtype=dt)
    a = F.conv2d(x, w, b, padding=1).float()
    c = im2col_conv(x, w, b).float()
    print(f"im2col vs cudnn 最大差 {float((a - c).abs().max()):.5f}", flush=True)


if __name__ == "__main__":
    main()
