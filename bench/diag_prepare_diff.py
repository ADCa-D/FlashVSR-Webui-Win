# -*- coding: utf-8 -*-
"""诊断：A(GPU prepare) 与 C(节点 prepare + CPU 画布) 的差异到底在哪。

打印：0..255 尺度上的 PSNR、大误差像素占比、按列/按行的误差剖面（看是否集中在 tile 接缝）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import engine as E  # noqa: E402

_LOG = open(os.path.join(ROOT, "work", "smoke_diag.log"), "w", encoding="utf-8", buffering=1)


class _Tee(object):
    def write(self, s):
        _LOG.write(s)
        return len(s)

    def flush(self):
        pass


sys.stdout = _Tee()

SRC = os.path.join(r"D:\ComfyUI_windows_portable\ComfyUI\input", "flashvsr_long_544.mp4")
N, TILE, OV = 9, 128, 32

eng = E.Engine()
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
frames = E.decode_frames_range(SRC, 0, N, pre_scale=0.5, fps=25.0)


def run(prep, canvas):
    os.environ["FLASHVSR_GPU_PREPARE"] = "1" if prep else "0"
    os.environ["FLASHVSR_GPU_CANVAS"] = "1" if canvas else "0"
    return np.asarray(eng._run_window(frames, 2, TILE, 3, tile_overlap=OV), dtype=np.float32)


run(True, True)                       # 预热
a = run(True, True)
b = run(True, True)                   # A 重跑自身，看流水线本身的可复现性
c = run(False, False)


def report(tag, x, y):
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    scale = 255.0 if hi > 1.5 else 1.0
    d = np.abs(x - y)
    mse = float((d ** 2).mean())
    psnr = 10 * np.log10(scale ** 2 / max(mse, 1e-12))
    print(f"{tag}: range=[{lo:.1f},{hi:.1f}] max|Δ|={d.max():.1f} mean={d.mean():.2f} "
          f"PSNR={psnr:.2f} dB  >4:{(d > 4).mean() * 100:.2f}%  >16:{(d > 16).mean() * 100:.3f}%",
          flush=True)
    prof_x = d.mean(axis=(0, 2, 3))
    prof_y = d.mean(axis=(1, 2, 3))
    bx = np.argsort(prof_x)[-5:][::-1]
    by = np.argsort(prof_y)[-5:][::-1]
    print(f"  最差列 {[int(i) for i in bx]} 值 {[round(float(prof_x[i]),2) for i in bx]}；"
          f"最差行 {[int(i) for i in by]} 值 {[round(float(prof_y[i]),2) for i in by]}", flush=True)
    print(f"  列剖面均质度 mean={prof_x.mean():.2f} max/mean={prof_x.max()/max(prof_x.mean(),1e-9):.2f}；"
          f"行剖面 max/mean={prof_y.max()/max(prof_y.mean(),1e-9):.2f}", flush=True)


report("A vs A（同路径重跑，测流水线可复现性）", a, b)
report("A vs C（新路径 vs 旧路径）", a, c)
print(f"形状 {a.shape} vs {c.shape}", flush=True)
