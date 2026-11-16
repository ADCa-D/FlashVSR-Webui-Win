# -*- coding: utf-8 -*-
"""搞清 libplacebo 里 preset 与 downscaler 的优先级：谁覆盖谁。"""
import hashlib
import os
import subprocess

ROOT = r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web"
OUT = os.path.join(ROOT, "work", "placebo2")
SRC = os.path.join(ROOT, "work", "placebo", "src_1080_50f.mp4")
FF = r"D:\ffmpeg-9.0.1\ffmpeg.exe"

CASES = [
    ("A downscaler=e4s", "libplacebo=w=540:h=304:downscaler=ewa_lanczos4sharpest"),
    ("B extra:downscaler=e4s", "libplacebo=w=540:h=304:extra_opts=downscaler=ewa_lanczos4sharpest"),
    ("C extra:preset=hq", "libplacebo=w=540:h=304:extra_opts=preset=high_quality"),
    ("D extra:preset+downscaler", "libplacebo=w=540:h=304:extra_opts=preset=high_quality:downscaler=ewa_lanczos4sharpest"),
    ("E downscaler=e4s + extra:preset", "libplacebo=w=540:h=304:downscaler=ewa_lanczos4sharpest:extra_opts=preset=high_quality"),
    ("F downscaler=spline36", "libplacebo=w=540:h=304:downscaler=spline36"),
]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


os.makedirs(OUT, exist_ok=True)
res = []
for tag, filt in CASES:
    dst = os.path.join(OUT, tag.split()[0] + ".mp4")
    cmd = [FF, "-v", "error", "-y", "-init_hw_device", "vulkan=pl",
           "-filter_hw_device", "pl", "-i", SRC, "-frames:v", "20",
           "-vf", filt + ",hwdownload,format=yuv420p", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "6",
           "-pix_fmt", "yuv420p", dst]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        err = [l for l in (r.stderr or "").splitlines() if "rror" in l]
        print(f"{tag:34s} FAIL {(err or [''])[0][:80]}")
        continue
    res.append((tag, dst, os.path.getsize(dst), md5(dst)[:10]))

print(f"\n{'配置':34s} {'大小KB':>8s}  md5")
for tag, dst, size, h in res:
    print(f"{tag:34s} {size//1024:8d}  {h}")

print("\n重复（说明后者被覆盖/等效）：")
seen = {}
for tag, dst, size, h in res:
    seen.setdefault(h, []).append(tag)
for h, tags in seen.items():
    if len(tags) > 1:
        print("  " + " == ".join(tags))
