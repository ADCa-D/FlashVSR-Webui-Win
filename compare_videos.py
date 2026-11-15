# -*- coding: utf-8 -*-
"""两段视频逐帧对比：PSNR / SSIM（可选 GT）。

用法：
  python compare_videos.py A.mp4 B.mp4 [--gt GT.mp4] [--width 640]

- A/B 尺寸不同会先缩放 B（或 GT）到 A 的尺寸
- 输出：平均/最差 PSNR、SSIM；若给 GT，再给两段各自对 GT 的 PSNR 与 ΔPSNR
"""
import argparse
import subprocess
import sys

import numpy as np

FFMPEG = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
FFPROBE = r"D:\ffmpeg-9.0.1\ffprobe.exe"


def probe(path):
    out = subprocess.check_output([
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0", path,
    ]).decode().strip().split(",")
    return int(out[0]), int(out[1])


def frames(path, width, height=None):
    h = height or int(round(probe(path)[1] * width / probe(path)[0] / 2)) * 2
    proc = subprocess.Popen(
        [FFMPEG, "-v", "error", "-i", path, "-vf", f"scale={width}:{h}:flags=bilinear,format=rgb24",
         "-f", "rawvideo", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fb = width * h * 3
    out = []
    while True:
        b = proc.stdout.read(fb)
        if not b or len(b) < fb:
            break
        out.append(np.frombuffer(b, np.uint8).reshape(h, width, 3))
    proc.wait()
    return np.stack(out) if out else None


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse <= 1e-9 else 10 * np.log10(255.0 ** 2 / mse)


def ssim_gray(a, b):
    a = a.mean(axis=2).astype(np.float32)
    b = b.mean(axis=2).astype(np.float32)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    A = frames(args.a, args.width)
    B = frames(args.b, args.width, A.shape[1])
    n = min(len(A), len(B))
    if n == 0:
        print("解码失败")
        return 2
    ps = [psnr(A[i], B[i]) for i in range(n)]
    ss = [ssim_gray(A[i], B[i]) for i in range(0, n, max(1, n // 60))]
    print(f"A={len(A)} 帧  B={len(B)} 帧  比较 {n} 帧")
    print(f"PSNR 平均 {np.mean(ps):6.2f} dB   最差 {np.min(ps):6.2f} dB   p5 {np.percentile(ps, 5):6.2f} dB")
    print(f"SSIM 平均 {np.mean(ss):.4f}")

    if args.gt:
        G = frames(args.gt, args.width, A.shape[1])
        m = min(n, len(G))
        pg_a = [psnr(A[i], G[i]) for i in range(m)]
        pg_b = [psnr(B[i], G[i]) for i in range(m)]
        print(f"对 GT：A {np.mean(pg_a):.2f} dB   B {np.mean(pg_b):.2f} dB   "
              f"ΔPSNR {np.mean(pg_b) - np.mean(pg_a):+.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
