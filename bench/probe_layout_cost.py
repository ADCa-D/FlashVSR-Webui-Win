# -*- coding: utf-8 -*-
"""手搓算子前的定位：TCDecoder 里布局转换占多少、conv 还能挖多少。

输出：解码路径的 kernel 分类（CONV / 布局转换 / ELEM / GEMM）与占比。
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

eng = E.Engine()
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
dec = eng.pipe.TCDecoder
frames = E.decode_frames_range(SRC, 0, 8, pre_scale=0.5, fps=25.0)
eng._run_window(frames[:4], 2, 128, 3, tile_overlap=32)

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    eng._run_window(frames, 2, 128, 3, tile_overlap=32)
    torch.cuda.synchronize()

evts = [e for e in prof.key_averages() if getattr(e, "self_device_time_total", 0) > 0]
total = sum(e.self_device_time_total for e in evts) / 1e6


def bucket(k):
    low = k.lower()
    if "nchwtonhwc" in low or "nhwctonchw" in low or "copy" in low:
        return "布局转换/拷贝"
    if "conv" in low or "implicit_gemm" in low:
        return "CONV"
    if "gemm" in low or "addmm" in low or "cutlass" in low:
        return "GEMM"
    if "reduce" in low or "elementwise" in low:
        return "ELEM"
    return "其它"


agg = {}
for e in evts:
    b = bucket(e.key)
    agg[b] = agg.get(b, 0) + e.self_device_time_total / 1e6

print(f"kernel 合计 {total:.2f}s")
for b, t in sorted(agg.items(), key=lambda x: -x[1]):
    print(f"  {b:12s} {t:6.3f}s  {t/total*100:5.1f}%")

print("\n布局转换明细（次数 × 时间）：")
for e in sorted([e for e in evts if bucket(e.key) == "布局转换/拷贝"],
                key=lambda x: -x.self_device_time_total)[:5]:
    print(f"  {e.self_device_time_total/1e6:6.3f}s {e.count:5d}x  {e.key[:70]}")
