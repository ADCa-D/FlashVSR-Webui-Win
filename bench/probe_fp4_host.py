# -*- coding: utf-8 -*-
"""NVFP4 的 host 开销来自哪：CPU 侧 profiler 看 top 操作。"""
import sys

import torch

sys.path.insert(0, r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web")
import comfy_kitchen as ck  # noqa: E402

M, K, N = 2304, 8960, 1536
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.02
ts = (w.abs().amax() / (6.0 * 448.0)).clamp(min=1e-8).float().reshape(1)
wq, wb = ck.quantize_nvfp4(w, ts, pad_16x=True)
xq, xb = ck.quantize_nvfp4(x, ts, pad_16x=True)

f = lambda: ck.scaled_mm_nvfp4(xq, wq, ts, ts, xb, wb, out_dtype=torch.bfloat16)
for _ in range(5):
    f()
torch.cuda.synchronize()

from torch.profiler import ProfilerActivity, profile
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(20):
        f()
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=14))
