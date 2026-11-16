# -*- coding: utf-8 -*-
"""探测 libplacebo 的 scaler 名称与 extra_opts 支持（本机 ffmpeg 9.0.1）。"""
import subprocess

FF = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_long_544.mp4"

SCALERS = ["spline36", "spline64", "lanczos", "ewa_lanczos", "ewa_lanczossharp",
           "ewa_lanczos4sharpest", "ewa_hanning", "sinc", "jinc", "ginseng",
           "bicubic", "mitchell", "catmull_rom", "nearest", "bilinear"]
PRESETS = ["", "preset=high_quality", "preset=fast"]

print("=== downscaler 名称支持 ===")
ok = []
for sc in SCALERS:
    cmd = [FF, "-hide_banner", "-v", "error", "-init_hw_device", "vulkan=pl",
           "-filter_hw_device", "pl", "-i", SRC,
           "-vf", f"libplacebo=w=480:h=272:downscaler={sc},hwdownload,format=yuv420p",
           "-frames:v", "2", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    good = r.returncode == 0
    if good:
        ok.append(sc)
    err = (r.stderr or "").strip().splitlines()
    print(f"  {sc:22s} {'OK' if good else 'FAIL'}  {'' if good else (err[0][:90] if err else '')}")

print("\n=== extra_opts=preset=... ===")
for pr in PRESETS:
    vf = "libplacebo=w=480:h=272:downscaler=spline36"
    if pr:
        vf += f":extra_opts={pr}"
    vf += ",hwdownload,format=yuv420p"
    cmd = [FF, "-hide_banner", "-v", "error", "-init_hw_device", "vulkan=pl",
           "-filter_hw_device", "pl", "-i", SRC, "-vf", vf,
           "-frames:v", "2", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    err = (r.stderr or "").strip().splitlines()
    print(f"  {pr or '(默认)':22s} {'OK' if r.returncode == 0 else 'FAIL'}  "
          f"{'' if r.returncode == 0 else (err[0][:100] if err else '')}")
