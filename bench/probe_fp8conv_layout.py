# -*- coding: utf-8 -*-
"""诊断：fp8 卷积在 TCDecoder 里每层到底有没有触发布局拷贝？

统计：输入是否已是 channels_last（no-copy）、是否被 .contiguous 复制、以及每次拷贝的字节数。
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import engine as E  # noqa: E402
import fp8_conv  # noqa: E402

SRC = os.path.join(r"D:\ComfyUI_windows_portable\ComfyUI\input", "flashvsr_long_544.mp4")
stat = {"already_cl": 0, "copy_cl": 0, "bytes": 0, "calls": 0}

orig_forward = fp8_conv.Conv2dFP8.forward


def counted(self, x):
    stat["calls"] += 1
    if x.is_contiguous(memory_format=torch.channels_last):
        stat["already_cl"] += 1
    else:
        stat["copy_cl"] += 1
        stat["bytes"] += x.numel() * x.element_size()
    return orig_forward(self, x)


fp8_conv.Conv2dFP8.forward = counted

eng = E.Engine()
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
frames = E.decode_frames_range(SRC, 0, 8, pre_scale=0.5, fps=25.0)

n = fp8_conv.patch_module(eng.pipe.TCDecoder, verbose=True)
print("替换卷积:", n)

eng._run_window(frames[:4], 2, 128, 3, tile_overlap=32)   # 预热（不计）
stat.update(already_cl=0, copy_cl=0, bytes=0, calls=0)

torch.cuda.synchronize()
t0 = time.perf_counter()
eng._run_window(frames, 2, 128, 3, tile_overlap=32)
torch.cuda.synchronize()
dt = time.perf_counter() - t0

print(f"一次窗口 {dt:.2f}s，Conv2dFP8 调用 {stat['calls']} 次")
print(f"  已是 channels_last（零拷贝）: {stat['already_cl']}")
print(f"  需要 .contiguous 复制      : {stat['copy_cl']}  "
      f"累计 {stat['bytes']/2**20:.0f} MiB")
