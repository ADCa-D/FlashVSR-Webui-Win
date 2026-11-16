# -*- coding: utf-8 -*-
"""先回答"C++ 重写值不值"：这轮推理到底是 GPU 忙，还是 CPU/调度有空隙？

判据（关键）：
  用 CUDA event 量"整段 GPU 时间轴"，再和墙钟比。
  GPU 时间轴 ≈ 墙钟 → GPU 全程在跑（C++ 省不出东西）；
  GPU 时间轴 << 墙钟 → 有空隙（才轮到换语言/图捕获）。
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import engine as E  # noqa: E402

SRC = os.path.join(r"D:\ComfyUI_windows_portable\ComfyUI\input", "flashvsr_long_544.mp4")

eng = E.Engine()
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
frames = E.decode_frames_range(SRC, 0, 8, pre_scale=0.5, fps=25.0)
eng._run_window(frames[:4], 2, 128, 3, tile_overlap=32)          # 预热

for trial in range(2):
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    t0 = time.perf_counter()
    s.record()
    eng._run_window(frames, 2, 128, 3, tile_overlap=32)
    e.record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    gpu = s.elapsed_time(e) / 1e3
    print(f"trial{trial}: 墙钟 {wall:5.2f}s   GPU 时间轴 {gpu:5.2f}s   "
          f"空隙 {wall - gpu:5.2f}s（{(wall-gpu)/wall*100:.1f}%）", flush=True)
