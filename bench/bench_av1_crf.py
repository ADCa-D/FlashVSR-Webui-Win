# -*- coding: utf-8 -*-
"""验证 AV1 的 CRF 甜点：4K 素材，svtav1 preset4，crf 20/23/26 的 VMAF 与体积。

用法：python bench/bench_av1_crf.py
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
OUT = os.path.join(ROOT, "work", "av1_crf")
N = 50
CRFS = [18, 20, 23, 26, 30]


def encode(crf, dst):
    cmd = [config.FFMPEG, "-v", "error", "-y", "-i", SRC, "-frames:v", str(N), "-an",
           "-c:v", "libsvtav1", "-preset", "4", "-crf", str(crf),
           "-svtav1-params", "lp=4:tune=0", "-pix_fmt", "yuv420p", dst]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        return None, 0, (r.stderr or "").strip()[-100:]
    return os.path.getsize(dst), time.perf_counter() - t0, ""


def vmaf(path):
    r = subprocess.run([config.FFMPEG, "-v", "info", "-i", path, "-i", SRC,
                        "-frames:v", str(N),
                        "-lavfi", "[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];"
                                  "[a][b]libvmaf", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.findall(r"VMAF score:\s*([\d.]+)", r.stderr or "")
    return float(m[-1]) if m else None


os.makedirs(OUT, exist_ok=True)
print(f"4K/50 帧，svtav1 preset4 tune=VQ，{os.path.basename(SRC)}\n")
print(f"{'crf':>4s} {'体积MB':>8s} {'码率Mbps':>9s} {'VMAF':>7s} {'耗时':>6s}")
for crf in CRFS:
    dst = os.path.join(OUT, f"crf{crf}.mp4")
    size, dt, err = encode(crf, dst)
    if not size:
        print(f"{crf:4d} FAIL {err}")
        continue
    mb = size / 1048576
    mbps = size * 8 / (N / 25.0) / 1e6
    v = vmaf(dst)
    print(f"{crf:4d} {mb:8.2f} {mbps:9.1f} {(f'{v:.2f}' if v else 'n/a'):>7s} {dt:5.1f}s")
