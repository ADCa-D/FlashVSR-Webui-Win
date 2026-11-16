# -*- coding: utf-8 -*-
"""在成片上挑最纹理的一行，裁出跨接缝的局部图（3× 最近邻放大）。"""
import os
import subprocess
import sys

ROOT = r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web"
sys.path.insert(0, ROOT)
import config  # noqa: E402
import engine as E  # noqa: E402
import numpy as np  # noqa: E402

path = sys.argv[1]
seam_x = int(sys.argv[2]) if len(sys.argv) > 2 else 976
w = int(sys.argv[3]) if len(sys.argv) > 3 else 360
h = int(sys.argv[4]) if len(sys.argv) > 4 else 240
nframe = int(sys.argv[5]) if len(sys.argv) > 5 else 40

info = E.probe(path)
q = subprocess.run([config.FFMPEG, "-v", "error", "-i", path, "-frames:v", str(nframe),
                    "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
a = np.frombuffer(q.stdout, np.uint8).astype("float32").reshape(-1, info["h"], info["w"])
x = max(0, min(seam_x - w // 2, info["w"] - w))
best, bestv = 0, -1
for y in range(0, info["h"] - h, 120):
    v = float(np.abs(np.diff(a[:, y:y + h, x:x + w], axis=2)).mean())
    if v > bestv:
        bestv, best = v, y
out = os.path.join(ROOT, "work", "seam_crops",
                   os.path.splitext(os.path.basename(path))[0] + f"_x{seam_x}_y{best}_3x.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
subprocess.run([config.FFMPEG, "-v", "error", "-y", "-i", path,
                "-vf", f"crop={w}:{h}:{x}:{best},scale=iw*3:ih*3:flags=neighbor",
                "-frames:v", "1", out], check=True)
print(out, "| 纹理强度", round(bestv, 3), "| 缝在裁剪内 x =", seam_x - x, "px")
