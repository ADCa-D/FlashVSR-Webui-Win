# -*- coding: utf-8 -*-
"""NVFP4 单 tile 计时：AMAX_EVERY=0（每层重算 scale）vs =16（缓存/16 次刷新）。"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import engine as E  # noqa: E402

SRC = os.path.join(r"D:\ComfyUI_windows_portable\ComfyUI\input", "flashvsr_long_544.mp4")
N = 8

eng = E.Engine()
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="nvfp4")
frames = E.decode_frames_range(SRC, 0, N, pre_scale=0.5, fps=25.0)
eng._run_window(frames[:4], 2, 128, 3, tile_overlap=32)

torch.cuda.synchronize()
t0 = time.perf_counter()
eng._run_window(frames, 2, 128, 3, tile_overlap=32)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"AMAX_EVERY={os.environ.get('FLASHVSR_FP4_AMAX_EVERY', '16')}  {dt:5.2f}s", flush=True)
