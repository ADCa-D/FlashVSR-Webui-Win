# -*- coding: utf-8 -*-
"""逐帧 PSNR：判断抽帧复用的画质损失是"插回的帧"还是"整体漂移"。

  - 只有插回帧低 → 映射/拷贝问题，可修
  - 所有帧都低   → 模型的时间上下文变了，是真实的画质代价
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = r"D:\ffmpeg-9.0.1\ffmpeg.exe"


def frames(path):
    # ffprobe 输出可能带空行/多余逗号，逐行解析
    txt = subprocess.run(
        [r"D:\ffmpeg-9.0.1\ffprobe.exe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
        capture_output=True, text=True, errors="replace").stdout
    nums = [int(x) for x in txt.replace("\n", ",").split(",") if x.strip().isdigit()]
    w, h = nums[0], nums[1]
    buf = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    n = len(buf) // (w * h * 3)
    return np.frombuffer(buf[: n * w * h * 3], np.uint8).reshape(n, h, w, 3).astype(np.float32)


def main():
    a = frames(os.path.join(ROOT, "work", "dropdemo", "off.mp4"))
    b = frames(os.path.join(ROOT, "work", "dropdemo", "on.mp4"))
    n = min(len(a), len(b))
    print(f"{len(a)} vs {len(b)} 帧，比较前 {n} 帧")
    mse = ((a[:n] - b[:n]) ** 2).mean(axis=(1, 2, 3))
    psnr = 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-9))
    for i in range(n):
        tag = ""
        if i % 2 == 1:
            tag = "  ← 插回帧"
        print(f"  f{i:02d}  PSNR {psnr[i]:6.2f} dB{tag}")
    even = psnr[::2].mean()
    odd = psnr[1::2].mean() if n > 1 else float("nan")
    print(f"\n原始帧平均 {even:.2f} dB   插回帧平均 {odd:.2f} dB")


if __name__ == "__main__":
    sys.exit(main())
