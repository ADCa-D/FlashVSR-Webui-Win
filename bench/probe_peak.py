# -*- coding: utf-8 -*-
"""量一下这台卡实测的 bf16 矩阵乘峰值（TFLOPS），用来给解码器算利用率。"""
import time

import torch

torch.backends.cuda.matmul.allow_tf32 = True
for n in (4096, 8192):
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    it = 10
    for _ in range(it):
        a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / it
    print(f"matmul {n}x{n} bf16: {dt*1000:.1f} ms  -> {2*n**3/dt/1e12:.1f} TFLOPS", flush=True)
