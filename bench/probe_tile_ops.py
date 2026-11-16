# -*- coding: utf-8 -*-
"""单 tile 的 op 级 profile：CONV / GEMM / ATTN / MEM 各占多少（当前设置：8 帧、2×、tile128）。

知识库里那张表是早期素材上测的；这次在当前默认设置下复核，决定"该往哪投精力"。

用法：python bench/probe_tile_ops.py [帧数] [quant]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import engine as E  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

SRC = os.path.join(r"D:\ComfyUI_windows_portable\ComfyUI\input", "flashvsr_long_544.mp4")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
QUANT = sys.argv[2] if len(sys.argv) > 2 else "mxfp8"

PAT = {
    "CONV": ("conv", "implicit_gemm", "cudnn"),
    "GEMM": ("gemm", "addmm", "matmul", "cutlass", "scaled_mm", "mm_"),
    "ATTN": ("flash", "attn", "sage", "softmax"),
    "MEM": ("copy", "memcpy", "cat", "permute", "contiguous", "transpose", "fill", "zero"),
}


def cat(name):
    low = name.lower()
    for k, keys in PAT.items():
        if any(x in low for x in keys):
            return k
    return "ELEM"


eng = E.Engine()
print("加载模型…", flush=True)
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant=QUANT)
frames = E.decode_frames_range(SRC, 0, N, pre_scale=0.5, fps=25.0)
print(f"输入 {frames.shape}", flush=True)

eng._run_window(frames[:4], 2, 128, 3, tile_overlap=32)   # 预热

torch.cuda.synchronize()
t0 = time.perf_counter()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    eng._run_window(frames, 2, 128, 3, tile_overlap=32)
    torch.cuda.synchronize()
wall = time.perf_counter() - t0

evts = [e for e in prof.key_averages() if getattr(e, "self_device_time_total", 0) > 0]
total = sum(e.self_device_time_total for e in evts)
agg = {}
for e in evts:
    agg.setdefault(cat(e.key), 0.0)
    agg[cat(e.key)] += e.self_device_time_total

print(f"\n墙钟 {wall:.2f}s   kernel 合计 {total/1e6:.2f}s（含重叠，仅作比例参考）\n")
for k, v in sorted(agg.items(), key=lambda x: -x[1]):
    print(f"  {k:5s} {v/1e6:6.3f}s  {v/total*100:5.1f}%")

print("\nTop 12 单个算子（按 self CUDA 时间）：")
for e in sorted(evts, key=lambda x: -x.self_device_time_total)[:12]:
    print(f"  {e.self_device_time_total/1e6:6.3f}s  {e.count:5d}x  {e.key[:70]}")
