# -*- coding: utf-8 -*-
"""卷积 roofline：代表性形状的实测 TFLOPS vs 本机 bf16 GEMM 实测峰值。

判据：
  卷积达到 GEMM 峰值的 >60% → 算力受限，fp8（峰值×2）值得做；
  <40% → 带宽/延迟受限，fp8 只减权重字节，收益有限，不值得写 kernel。
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


# --- 本机 bf16 GEMM 实测峰值（M=N=K=4096）
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
t = bench(lambda: a @ b)
peak = 2 * 4096 ** 3 / t / 1e12
print(f"bf16 GEMM 实测峰值 {peak:.1f} TFLOPS（4096³，{t*1e3:.1f} ms）\n")

# --- 代表性卷积（VAE/TCDecoder 常见：3×3、通道 256/512、8 帧 batch）
SHAPES = [
    ("8x512x128x128 k3", (8, 512, 128, 128), 512, 3, 1),
    ("8x256x256x256 k3", (8, 256, 256, 256), 256, 3, 1),
    ("8x128x256x256 k3", (8, 128, 256, 256), 128, 3, 1),
    ("8x64x512x512 k3",  (8, 64, 512, 512), 64, 3, 1),
]
print(f"{'形状':22s} {'ms':>7s} {'TFLOPS':>7s} {'占峰值':>7s}")
for tag, shape, cout, k, stride in SHAPES:
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(cout, shape[1], k, k, device="cuda", dtype=torch.bfloat16)
    fn = lambda: F.conv2d(x, w, padding=k // 2, stride=stride)   # noqa: E731
    t = bench(fn)
    flops = 2 * shape[0] * cout * shape[2] * shape[3] * shape[1] * k * k
    tf = flops / t / 1e12
    print(f"{tag:22s} {t*1e3:7.2f} {tf:7.1f} {tf/peak*100:6.0f}%")
    del x, w
    torch.cuda.empty_cache()
