# -*- coding: utf-8 -*-
"""验证：deband / dithering 是独立选项，可以和自定义 scaler 组合（不用 preset）。

同时测：耗时 + 对刻意色带输入的去色带效果。
"""
import os
import subprocess
import time

ROOT = r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web"
OUT = os.path.join(ROOT, "work", "hq_banding")
FF = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
SRC = os.path.join(OUT, "gradient_1080.mp4")
N = 50

BASE = "libplacebo=w=540:h=304:downscaler=ewa_lanczos4sharpest"
CASES = [
    ("只有 scaler", BASE),
    ("+ deband", BASE + ":deband=1"),
    ("+ dithering=blue", BASE + ":dithering=blue"),
    ("+ deband + 抖动", BASE + ":deband=1:deband_iterations=2:dithering=blue:dither_temporal=1"),
    ("preset=high_quality", "libplacebo=w=540:h=304:extra_opts=preset=high_quality"),
]


def run(filt, dst):
    cmd = [FF, "-hide_banner", "-benchmark", "-v", "info", "-y",
           "-init_hw_device", "vulkan=pl", "-filter_hw_device", "pl",
           "-i", SRC, "-frames:v", str(N), "-vf", filt + ",hwdownload,format=yuv420p",
           "-an", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", dst]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    return r.returncode == 0, dt, (r.stderr or "")


def luma_stats(path):
    import numpy as np
    p = subprocess.run([FF, "-v", "error", "-i", path, "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
    a = np.frombuffer(p.stdout, np.uint8).reshape(304, 540).astype(int)
    row = a[150]
    d = np.abs(np.diff(row))
    # 色带 = 相邻像素完全相同（平台）的比例；抖动会把这个比例压低
    plateau = float((d == 0).mean())
    # 高频能量（抖动会抬高）
    hf = float(np.abs(np.diff(row, 2)).mean())
    return plateau, hf, int(len(np.unique(row)))


print(f"{'配置':44s} {'耗时':>6s} {'体积KB':>7s} {'平台比':>7s} {'HF能量':>7s}")
for tag, filt in CASES:
    dst = os.path.join(OUT, "cmp_" + tag.replace(" ", "_").replace("+", "p").replace("=", "") + ".mp4")
    ok, dt, err = run(filt, dst)
    if not ok:
        print(f"{tag:44s} FAIL {err.strip().splitlines()[-1][:70]}")
        continue
    plateau, hf, uniq = luma_stats(dst)
    print(f"{tag:44s} {dt:5.1f}s {os.path.getsize(dst)//1024:7d} "
          f"{plateau*100:6.1f}% {hf:7.3f}")
