# -*- coding: utf-8 -*-
"""缩放器保真度：540p 结果再放大回 1080p，与原始 1080p 比 PSNR/SSIM。

逻辑：好的降采样应当"可逆"——保留细节、不引入混叠。放大用固定 ewa_lanczos，
所以差异只来自降采样那一步。
"""
import os
import subprocess

ROOT = r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web"
SRC = os.path.join(ROOT, "work", "placebo", "src_1080_50f.mp4")
OUT = os.path.join(ROOT, "work", "placebo2")
RT = os.path.join(ROOT, "work", "placebo2", "rt")
FF = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
N = 20

FILES = ["A.mp4", "C.mp4", "F.mp4"]
LABEL = {"A.mp4": "ewa_lanczos4sharpest", "C.mp4": "preset=high_quality",
         "F.mp4": "spline36"}


def upscale_back(path, dst):
    cmd = [FF, "-v", "error", "-y", "-init_hw_device", "vulkan=pl",
           "-filter_hw_device", "pl", "-i", path, "-frames:v", str(N),
           "-vf", "libplacebo=w=1920:h=1080:upscaler=ewa_lanczos,hwdownload,format=yuv420p",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "4",
           "-pix_fmt", "yuv420p", dst]
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").returncode == 0


def psnr_ssim(a, b):
    """用 ffmpeg 的 psnr/ssim 滤镜，与源比（只在相同尺寸下比较）。"""
    r = subprocess.run([FF, "-v", "info", "-i", a, "-i", b, "-frames:v", str(N),
                        "-lavfi", "[0:v][1:v]psnr;[0:v][1:v]ssim",
                        "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = {}
    for l in (r.stderr or "").splitlines():
        if "PSNR" in l:
            out["psnr"] = l.strip().split("average:")[-1].split()[0] if "average:" in l else "?"
        if "SSIM" in l:
            out["ssim"] = l.strip().split("All:")[-1].split()[0] if "All:" in l else "?"
    return out


os.makedirs(RT, exist_ok=True)
print(f"{'降采样器':24s} {'体积KB':>7s} {'PSNR(往返)':>11s} {'SSIM':>7s}")
for f in FILES:
    p = os.path.join(OUT, f)
    if not os.path.isfile(p):
        print(f"{LABEL[f]:24s} 缺失")
        continue
    rt = os.path.join(RT, "rt_" + f)
    ok = upscale_back(p, rt)
    m = psnr_ssim(rt, SRC) if ok else {}
    print(f"{LABEL[f]:24s} {os.path.getsize(p)//1024:7d} "
          f"{m.get('psnr','?'):>11s} {m.get('ssim','?'):>7s}")
