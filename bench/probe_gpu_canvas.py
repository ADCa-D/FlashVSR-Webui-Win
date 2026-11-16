# -*- coding: utf-8 -*-
"""GPU 画布 A/B：同一轮推理，对比"节点原版 CPU 画布"与"GPU 画布"，并算输出差异。"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import engine as E  # noqa: E402

SRC = os.path.join(r"D:\ComfyUI_windows_portable\ComfyUI\input", "flashvsr_long_544.mp4")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8

eng = E.Engine()
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
frames = E.decode_frames_range(SRC, 0, N, pre_scale=0.5, fps=25.0)
print(f"输入 {frames.shape}", flush=True)


def run(tag):
    eng._run_window(frames[:4], 2, 128, 3, tile_overlap=32)      # 预热
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = eng._run_window(frames, 2, 128, 3, tile_overlap=32)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"{tag:12s} {dt:5.2f}s  {out.shape} {out.dtype}  max={out.max()}", flush=True)
    return out, dt


os.environ["FLASHVSR_GPU_CANVAS"] = "0"
a, ta = run("CPU 画布")
os.environ["FLASHVSR_GPU_CANVAS"] = "1"
b, tb = run("GPU 画布")

diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
print(f"\n加速   {(ta-tb)/ta*100:+.1f}%（{ta:.2f} → {tb:.2f}s）")
print(f"输出差 平均 {diff.mean():.3f}/255  最大 {diff.max()}  完全相同像素 "
      f"{(diff == 0).mean()*100:.1f}%")
