# -*- coding: utf-8 -*-
"""核对：extra_opts=preset=high_quality 是否把 downscaler 覆盖掉了（HQ 反而更快，可疑）。"""
import hashlib
import os
import subprocess

ROOT = r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web"
OUT = os.path.join(ROOT, "work", "placebo")
FF = r"D:\ffmpeg-9.0.1\ffmpeg.exe"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def psnr(a, b):
    r = subprocess.run([FF, "-v", "error", "-i", a, "-i", b,
                        "-lavfi", "[0:v][1:v]psnr", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    for l in (r.stderr or "").splitlines():
        if "PSNR" in l:
            return l.strip()[-60:]
    return (r.stderr or "").strip()[-80:]


files = [f for f in sorted(os.listdir(OUT)) if f.endswith(".mp4")]
print(f"{'文件':40s} {'大小KB':>8s}  md5")
for f in files:
    p = os.path.join(OUT, f)
    print(f"{f:40s} {os.path.getsize(p)//1024:8d}  {md5(p)[:12]}")

pairs = [
    ("ewa_lanczos4sharpest_hq_HQ.mp4", "spline36_hq_HQ.mp4"),
    ("ewa_lanczos4sharpest_hq_HQ.mp4", "ewa_lanczos4sharpest.mp4"),
    ("spline36_hq_HQ.mp4", "spline36.mp4"),
]
print()
for a, b in pairs:
    pa, pb = os.path.join(OUT, a), os.path.join(OUT, b)
    if os.path.isfile(pa) and os.path.isfile(pb):
        same = md5(pa) == md5(pb)
        print(f"{a} vs {b}\n  {'完全相同（说明被覆盖）' if same else psnr(pa, pb)}")
