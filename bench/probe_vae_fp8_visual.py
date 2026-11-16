# -*- coding: utf-8 -*-
"""fp8 卷积的肉眼画质：存片 + SSIM + 放大裁图。

PSNR 34.4 dB 听起来很糟，但 PSNR 对"轻微色偏/细节丢失"不敏感，也可能反过来高估损伤。
这里给出 SSIM 和两张 3× 放大裁图，交给眼睛判断。
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import config  # noqa: E402
import engine as E  # noqa: E402
import fp8_conv  # noqa: E402

SRC = os.path.join(r"D:\ComfyUI_windows_portable\ComfyUI\input", "flashvsr_long_544.mp4")
OUT = os.path.join(ROOT, "work", "fp8vis")
N = 8

os.makedirs(OUT, exist_ok=True)
eng = E.Engine()
print("加载模型…", flush=True)
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
frames = E.decode_frames_range(SRC, 0, N, pre_scale=0.5, fps=25.0)
dec = eng.pipe.TCDecoder


def run(tag):
    out = eng._run_window(frames, 2, 128, 3, tile_overlap=32)
    dst = os.path.join(OUT, f"{tag}.mp4")
    E.encode_frames(out, 25.0, dst, None, quality=10, backend="x265")
    print(f"{tag:5s} → {dst}  {out.shape}", flush=True)
    return dst, out


bf, o_bf = run("bf16")
n = fp8_conv.patch_module(dec, verbose=True)
print("替换卷积:", n, flush=True)
fp, o_fp = run("fp8")

# PSNR / SSIM（走 compare_videos.py，它同时给两个指标）
r = subprocess.run([sys.executable, os.path.join(ROOT, "compare_videos.py"), bf, fp, "--width", "960"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print("\n" + "\n".join([l for l in (r.stdout or "").splitlines() if l.strip()][-4:]))

# 放大裁图：挑纹理最多的一处，两种各一张
g = o_bf.astype(np.float32).mean(axis=3)
best = max(range(0, g.shape[1] - 160, 40),
           key=lambda y: float(np.abs(np.diff(g[0, y:y + 160, 200:520], axis=1)).mean()))
for tag, dst in (("bf16", bf), ("fp8", fp)):
    subprocess.run([config.FFMPEG, "-v", "error", "-y", "-i", dst, "-vf",
                    f"crop=320:160:200:{best},scale=iw*3:ih*3:flags=neighbor",
                    "-frames:v", "1", os.path.join(OUT, f"crop_{tag}.png")], check=True)
print(f"\n裁图：{OUT}\\crop_bf16.png / crop_fp8.png  (y={best})")
