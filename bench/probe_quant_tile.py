# -*- coding: utf-8 -*-
"""量化 × tile 的单 tile 计时（M 效应）：python bench/probe_quant_tile.py <quant> <tile>"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import engine as E  # noqa: E402

quant = sys.argv[1] if len(sys.argv) > 1 else "mxfp8"
tile = int(sys.argv[2]) if len(sys.argv) > 2 else 128
SRC = os.path.join(r"D:\ComfyUI_windows_portable\ComfyUI\input", "flashvsr_long_544.mp4")

eng = E.Engine()
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant=quant)
frames = E.decode_frames_range(SRC, 0, 8, pre_scale=0.5, fps=25.0)
ov = max(24, tile // 4)
eng._run_window(frames[:4], 2, tile, 3, tile_overlap=ov)
torch.cuda.synchronize()
t0 = time.perf_counter()
eng._run_window(frames, 2, tile, 3, tile_overlap=ov)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"{quant:6s} tile={tile:3d} ov={ov:2d}  {dt:5.2f}s", flush=True)
