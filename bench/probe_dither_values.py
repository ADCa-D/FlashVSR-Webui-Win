# -*- coding: utf-8 -*-
"""实测 dithering 各枚举：哪些真的生效（输出不同），哪些其实是空转/一样。"""
import hashlib
import os
import subprocess

ROOT = r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web"
OUT = os.path.join(ROOT, "work", "dither_probe")
FF = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
SRC = os.path.join(ROOT, "work", "hq_banding", "banded_1080.mp4")
N = 3

VALUES = ["", "none", "blue", "ordered", "ordered_fixed", "white"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:10]


def run(val, force):
    tag = (val or "default") + ("_force" if force else "")
    dst = os.path.join(OUT, f"{tag}.mp4")
    filt = "libplacebo=w=540:h=304:downscaler=spline36"
    if val:
        filt += f":dithering={val}"
    if force:
        filt += ":force_dither=1"
    cmd = [FF, "-v", "error", "-y", "-init_hw_device", "vulkan=pl",
           "-filter_hw_device", "pl", "-i", SRC, "-frames:v", str(N),
           "-vf", filt + ",hwdownload,format=yuv420p", "-an",
           "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", dst]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        err = [l for l in (r.stderr or "").splitlines() if "rror" in l or "Invalid" in l]
        return tag, None, (err or [""])[0][:80]
    return tag, (os.path.getsize(dst), md5(dst)), ""


os.makedirs(OUT, exist_ok=True)
print(f"{'dithering':16s} {'force':>6s} {'体积KB':>7s}  md5")
rows = []
for force in (False, True):
    for v in VALUES:
        tag, info, err = run(v, force)
        if info is None:
            print(f"{tag:16s} {str(force):>6s}  FAIL {err}")
            continue
        size, h = info
        print(f"{tag:16s} {str(force):>6s} {size//1024:7d}  {h}")
        rows.append((tag, h))

print("\n输出相同（说明该枚举没实际效果）：")
seen = {}
for tag, h in rows:
    seen.setdefault(h, []).append(tag)
for h, tags in seen.items():
    if len(tags) > 1:
        print("  " + " == ".join(tags))
