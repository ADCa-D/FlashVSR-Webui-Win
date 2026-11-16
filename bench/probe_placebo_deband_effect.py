# -*- coding: utf-8 -*-
"""deband 到底有没有用：喂一张**刻意色带化**的输入，量"最长同值平台"的长度。"""
import os
import subprocess

ROOT = r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web"
OUT = os.path.join(ROOT, "work", "hq_banding")
FF = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
N = 4

CASES = [
    ("无处理（只有 scaler）", "libplacebo=w=540:h=304:downscaler=ewa_lanczos4sharpest"),
    ("+ deband 默认", "libplacebo=w=540:h=304:downscaler=ewa_lanczos4sharpest:deband=1"),
    ("+ deband 强（iter3 th8）", "libplacebo=w=540:h=304:downscaler=ewa_lanczos4sharpest"
                                 ":deband=1:deband_iterations=3:deband_threshold=8"),
]


def make_banded(path):
    """把渐变量化成 8 级 → 明显色带。"""
    vf = ("geq=lum='16 + 200*X/W':cb=128:cr=128,"
          "lutyuv=y='round((val-16)/24)*24+16',format=yuv420p")
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"color=c=gray:s=1920x1080:r=25:d={N/25:.2f}",
                    "-vf", vf, "-frames:v", str(N), "-c:v", "libx264",
                    "-crf", "0", "-pix_fmt", "yuv420p", path], check=True)


def run(filt, dst):
    cmd = [FF, "-v", "error", "-y", "-init_hw_device", "vulkan=pl",
           "-filter_hw_device", "pl", "-i", SRC, "-frames:v", str(N),
           "-vf", filt + ",hwdownload,format=yuv420p", "-an",
           "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", dst]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode == 0, (r.stderr or "").strip().splitlines()[-1:]


def flats(path):
    import numpy as np
    p = subprocess.run([FF, "-v", "error", "-i", path, "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
    a = np.frombuffer(p.stdout, np.uint8).reshape(304, 540)
    row = a[150]
    longest = run_ = 1
    for i in range(1, len(row)):
        run_ = run_ + 1 if row[i] == row[i - 1] else 1
        longest = max(longest, run_)
    return longest, int(len(np.unique(row))), int(np.abs(np.diff(row.astype(int))).max())


SRC = os.path.join(OUT, "banded_1080.mp4")
os.makedirs(OUT, exist_ok=True)
if not os.path.isfile(SRC):
    make_banded(SRC)

print(f"{'配置':28s} {'最长平台(px)':>12s} {'不同值':>7s} {'最大跳变':>8s}")
for tag, filt in CASES:
    dst = os.path.join(OUT, "db_" + tag.replace(" ", "_").replace("（", "").replace("）", "")
                       .replace("+", "p").replace("(", "").replace(")", "") + ".mp4")
    ok, err = run(filt, dst)
    if not ok:
        print(f"{tag:28s} FAIL {err[0][:70]}")
        continue
    lf, uniq, mx = flats(dst)
    print(f"{tag:28s} {lf:12d} {uniq:7d} {mx:8d}")
