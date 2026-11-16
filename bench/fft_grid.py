# -*- coding: utf-8 -*-
"""用 2D 频谱找周期性网格：给出周期（px）与方向。

用法：python bench/fft_grid.py <视频> [帧号] [crop尺寸]
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import config  # noqa: E402
import engine as E  # noqa: E402
import numpy as np  # noqa: E402


def grab(path, idx, size):
    info = E.probe(path)
    x = (info["w"] - size) // 2
    y = (info["h"] - size) // 2
    q = subprocess.run([config.FFMPEG, "-v", "error", "-i", path,
                        "-vf", f"select=eq(n\\,{idx}),crop={size}:{size}:{x}:{y}",
                        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                       capture_output=True)
    a = np.frombuffer(q.stdout, np.uint8).astype("float32").reshape(size, size) / 255.0
    return a, info


def main():
    path = sys.argv[1]
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 512

    a, info = grab(path, idx, size)
    # 高通：减去 9×9 均值，只留细节
    k = 9
    pad = np.pad(a, k // 2, mode="edge")
    blur = sum(pad[i:i + size, j:j + size] for i in range(k) for j in range(k)) / (k * k)
    hp = (a - blur) * np.hanning(size)[:, None] * np.hanning(size)[None, :]

    F = np.abs(np.fft.fftshift(np.fft.fft2(hp)))
    c = size // 2
    F[c - 6:c + 7, c - 6:c + 7] = 0            # 去掉直流/低频
    F[:c - 200] = 0
    F[c + 200:] = 0
    F[:, :c - 200] = 0
    F[:, c + 200:] = 0

    flat = F.ravel()
    order = np.argsort(flat)[::-1][:400]
    peaks = []
    for i in order:
        fy, fx = divmod(int(i), size)
        if any(abs(fy - p[0]) < 3 and abs(fx - p[1]) < 3 for p in peaks):
            continue
        peaks.append((fy, fx, flat[i]))
        if len(peaks) >= 6:
            break

    print(f"{os.path.basename(path)}  {info['w']}x{info['h']}  第 {idx} 帧  中心 {size}×{size} 裁块")
    print(f"{'方向':8s} {'周期(px)':>9s} {'强度':>10s}")
    for fy, fx, v in peaks:
        dy, dx = fy - c, fx - c
        if dx == 0 and dy == 0:
            continue
        r = max(abs(dx), abs(dy))
        period = size / r if r else 0
        ori = "水平线" if abs(dy) > 2 * abs(dx) else ("垂直线" if abs(dx) > 2 * abs(dy) else "斜/二维")
        print(f"{ori:8s} {period:9.1f} {v:10.1f}   (dy={dy}, dx={dx})")


if __name__ == "__main__":
    main()
