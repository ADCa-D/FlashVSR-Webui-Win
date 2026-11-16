# -*- coding: utf-8 -*-
"""A/B：channels_last 开/关（clean_vram 已是空操作）。一次跑一轮，用环境变量切换。"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import engine as E  # noqa: E402

SRC = os.path.join(r"D:\ComfyUI_windows_portable\ComfyUI\input", "flashvsr_long_544.mp4")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8

eng = E.Engine()
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
frames = E.decode_frames_range(SRC, 0, N, pre_scale=0.5, fps=25.0)
eng._run_window(frames[:4], 2, 128, 3, tile_overlap=32)      # 预热
torch.cuda.synchronize()
t0 = time.perf_counter()
eng._run_window(frames, 2, 128, 3, tile_overlap=32)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print("VAE_CL=%s  %5.2fs" % (os.environ.get("FLASHVSR_VAE_CL", "1"), dt), flush=True)
