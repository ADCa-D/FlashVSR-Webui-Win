# -*- coding: utf-8 -*-
"""看“羽化带糊成一条”的网格：列向高频能量 + 裁局部图给人眼核对。

用法：python bench/analyze_seams2.py <视频> [帧数] [stride]
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


def read(path, n):
    info = E.probe(path)
    p = subprocess.run([config.FFMPEG, "-v", "error", "-i", path, "-frames:v", str(n),
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    a = np.frombuffer(p.stdout, np.uint8).astype("float32").reshape(-1, info["h"], info["w"], 3) / 255.0
    return a, info


def crop(path, x, y, w, h, out, zoom=1):
    vf = f"crop={w}:{h}:{x}:{y}"
    if zoom > 1:
        vf += f",scale=iw*{zoom}:ih*{zoom}:flags=neighbor"
    subprocess.run([config.FFMPEG, "-v", "error", "-y", "-i", path, "-vf", vf,
                    "-frames:v", "1", out], check=True)


def main():
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 488

    a, info = read(path, n)
    H, W = a.shape[1], a.shape[2]
    g = a.mean(axis=3)
    # 列向二阶差分能量：横跨接缝的像素被"糊"过，这条曲线会在缝上掉下去
    d2 = np.abs(g[:, :, 2:] - 2 * g[:, :, 1:-1] + g[:, :, :-2]).mean(axis=(0, 1))
    prof = np.convolve(d2, np.ones(15) / 15, mode="same")
    print(f"{os.path.basename(path)} {info['w']}x{info['h']}  帧 {len(a)}")
    print(f"列向 HF 能量：中位数 {np.median(prof):.6f}  p10 {np.percentile(prof,10):.6f}")

    idx = [k * stride for k in range(1, (W - 1) // stride + 1)]
    if idx:
        seam = float(np.mean([prof[max(0, i - 8):i + 9].mean() for i in idx]))
        mask = np.ones(len(prof), bool)
        for i in idx:
            mask[max(0, i - 16):i + 17] = False
        base = float(prof[mask].mean())
        print(f"stride={stride}: 缝处 HF {seam:.6f}  非缝 {base:.6f}  比值 {seam/max(base,1e-9):.3f}  (1.0=无差异)")

    # 自相关（去掉均值）
    x = prof - prof.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0]
    lo, hi = 60, min(len(ac) - 1, 900)
    best = lo + int(np.argmax(ac[lo:hi]))
    print(f"HF 曲线自相关主周期 ≈ {best} px (峰 {ac[best]:.3f})")

    os.makedirs(os.path.join(ROOT, "work", "seam_crops"), exist_ok=True)
    outdir = os.path.join(ROOT, "work", "seam_crops")
    name = os.path.splitext(os.path.basename(path))[0]
    cx = idx[len(idx) // 2] if idx else W // 2
    crop(path, max(0, cx - 200), H // 2 - 150, 400, 300,
         os.path.join(outdir, f"{name}_seam_x{cx}.png"), zoom=2)
    crop(path, max(0, cx - 200 - stride // 2), H // 2 - 150, 400, 300,
         os.path.join(outdir, f"{name}_control.png"), zoom=2)
    print("裁图：", outdir, f"(seam x={cx})")


if __name__ == "__main__":
    main()
