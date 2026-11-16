# -*- coding: utf-8 -*-
"""先做 roofline：最大的几个卷积到底是"算力受限"还是"带宽/延迟受限"？

只有算力受限，fp8 卷积（峰值翻倍）才有意义；带宽受限则 fp8 只减权重字节，收益有限。
做法：带 record_shapes 抓 conv 调用，取最耗时的几个 shape，按 FLOPS/Bytes 估算。

用法：python bench/probe_conv_roofline.py [帧数]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import engine as E  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

SRC = os.path.join(r"D:\ComfyUI_windows_portable\ComfyUI\input", "flashvsr_long_544.mp4")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8

eng = E.Engine()
print("加载模型…", flush=True)
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
frames = E.decode_frames_range(SRC, 0, N, pre_scale=0.5, fps=25.0)
eng._run_window(frames[:4], 2, 128, 3, tile_overlap=32)

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    eng._run_window(frames, 2, 128, 3, tile_overlap=32)
    torch.cuda.synchronize()

rows = []
for e in prof.key_averages():
    t = getattr(e, "self_device_time_total", 0)
    if t <= 0:
        continue
    low = e.key.lower()
    if not any(k in low for k in ("conv", "implicit_gemm")):
        continue
    rows.append((t / 1e6, e.count, e.key[:56], e.input_shapes))

rows.sort(reverse=True)
print(f"\n{'CUDA(s)':>8s} {'次数':>5s}  形状")
for t, c, k, shapes in rows[:8]:
    print(f"{t:8.3f} {c:5d}  {k}")
    for s in (shapes or [])[:2]:
        print(f"           {s}")
