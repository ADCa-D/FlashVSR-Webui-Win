# -*- coding: utf-8 -*-
"""预览帧格式对比：PNG / JPEG / AVIF（HEIC 浏览器基本不支持，先测能不能编）。

指标：单帧体积、编码耗时、与无损 PNG 的 PSNR/SSIM。
"""
import os
import subprocess
import time

ROOT = r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web"
OUT = os.path.join(ROOT, "work", "pv_fmt")
FF = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
SRC = os.path.join(ROOT, "out", "4765_333d880c286c_2x.mp4")
T = "12.0"
W = 1280


def grab(fmt, extra, name, src_args=("-ss", T, "-i", SRC)):
    dst = os.path.join(OUT, name)
    cmd = [FF, "-v", "error", "-y"] + list(src_args) + \
          ["-frames:v", "1", "-vf", f"scale='min({W},iw)':-2"] + extra + [dst]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    if r.returncode != 0 or not os.path.exists(dst):
        err = [l for l in (r.stderr or "").splitlines() if l.strip()]
        return None, dt, (err[-1][:90] if err else "")
    return (os.path.getsize(dst), dst), dt, ""


def q(path, ref):
    r = subprocess.run([FF, "-v", "info", "-i", path, "-i", ref,
                        "-lavfi", "[0:v][1:v]psnr;[0:v][1:v]ssim", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    ps = ss = "?"
    for l in (r.stderr or "").splitlines():
        if "PSNR" in l and "average:" in l:
            ps = l.split("average:")[-1].split()[0]
        if "SSIM" in l and "All:" in l:
            ss = l.split("All:")[-1].split()[0]
    return ps, ss


os.makedirs(OUT, exist_ok=True)
print("先看本机能不能编码这些格式：")
CASES = [
    ("png", ["-c:v", "png"], "pv.png"),
    ("jpeg q3", ["-c:v", "mjpeg", "-q:v", "3"], "pv_q3.jpg"),
    ("jpeg q2", ["-c:v", "mjpeg", "-q:v", "2"], "pv_q2.jpg"),
    ("webp", ["-c:v", "libwebp", "-quality", "90"], "pv.webp"),
    ("avif(svtav1)", ["-c:v", "libsvtav1", "-crf", "28", "-still-picture", "1"], "pv_svt.avif"),
    ("avif(libaom)", ["-c:v", "libaom-av1", "-crf", "28", "-still-picture", "1",
                      "-cpu-used", "6"], "pv_aom.avif"),
    ("heic", ["-c:v", "libx265", "-crf", "28"], "pv.heic"),
]

rows = []
for tag, extra, name in CASES:
    info, dt, err = grab(tag, extra, name)
    if not info:
        print(f"  {tag:14s} FAIL {err}")
        continue
    size, path = info
    print(f"  {tag:14s} OK   {size//1024:5d} KB  {dt:5.2f}s")
    rows.append((tag, path, size, dt))

ref = os.path.join(OUT, "pv.png")
print(f"\n{'格式':14s} {'体积KB':>7s} {'耗时':>6s} {'PSNR':>8s} {'SSIM':>7s}")
for tag, path, size, dt in rows:
    if path.endswith(".png"):
        print(f"{tag:14s} {size//1024:7d} {dt:5.2f}s {'(无损参考)':>8s}")
        continue
    ps, ss = q(path, ref)
    print(f"{tag:14s} {size//1024:7d} {dt:5.2f}s {ps:>8s} {ss:>7s}")
