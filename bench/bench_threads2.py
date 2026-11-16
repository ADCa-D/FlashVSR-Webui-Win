# -*- coding: utf-8 -*-
"""补两项对照：
  1. x264 的 -threads 4 是否真的限住（对比 x265 的 pools 坑）；
  2. SVT-AV1 的 lp=4 实际用了多少逻辑核（抓 SVT 自己的日志）。
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import config  # noqa: E402

SRC = os.path.join(ROOT, "out", "4765_333d880c286c_2x.mp4")
OUT = os.path.join(ROOT, "work", "threads")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30

CASES = [
    ("x264 slow t4", ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-threads", "4"]),
    ("x264 slow t1", ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-threads", "1"]),
    ("av1 p8 lp4", ["-c:v", "libsvtav1", "-preset", "8", "-crf", "26",
                    "-svtav1-params", "lp=4"]),
]

for tag, args in CASES:
    dst = os.path.join(OUT, tag.replace(" ", "_") + ".mp4")
    t0 = time.perf_counter()
    r = subprocess.run(
        [config.FFMPEG, "-hide_banner", "-benchmark", "-v", "info", "-y", "-i", SRC,
         "-frames:v", str(N), "-an"] + args + ["-pix_fmt", "yuv420p", dst],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    err = r.stderr or ""
    m = re.search(r"bench: utime=([\d.]+)s stime=([\d.]+)s", err)
    cpu = (float(m.group(1)) + float(m.group(2))) if m else -1
    print(f"{tag:14s} wall {wall:5.1f}s  cpu {cpu:6.1f}s  并发 {cpu/wall:4.1f}x")
    for line in err.splitlines():
        low = line.lower()
        if any(k in low for k in ("thread", "logical", "ppcs", "pools", "core")):
            print("    ·", line.strip()[:120])
    print()
