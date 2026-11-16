# -*- coding: utf-8 -*-
"""空间复用可行性第一步：真实素材里有多少 128×128 块在相邻窗之间"几乎没变"。

流水线的事实：每个窗口只推进 2 帧，所以"同一空间位置的上一次结果"= 2 帧前。
这里直接量像素域：对每个 128×128 块算 |f(t) − f(t−2)| 的均值，给出分布和占比。

输出：不同阈值下"可复用块"占比 + 分位数。这只是收益上限（GEMM 仍要全算，
能省的是稀疏注意力里对这些块的 qk/pv 计算）。
"""
import os
import subprocess
import sys

import numpy as np

FFMPEG = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = [
    r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_long_544.mp4",
    os.path.join(ROOT, "work", "placebo", "src_1080_50f.mp4"),
]
W, H = 512, 288          # 统一降到 512×288 量（块 128 → 4×2 = 8 块/帧）
BLK = 64                 # 128px 块在降采样后是 64（因为 W 从 1024 降到 512）
FRAMES = 60


def grab(path, frames=FRAMES):
    cmd = [FFMPEG, "-v", "error", "-i", path, "-frames:v", str(frames),
           "-vf", f"scale={W}:{H}", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    out = subprocess.run(cmd, capture_output=True)
    buf = np.frombuffer(out.stdout, dtype=np.uint8)
    n = buf.size // (W * H)
    return buf[: n * W * H].reshape(n, H, W).astype(np.float32)


def main():
    for path in CLIPS:
        if not os.path.exists(path):
            print(f"skip (missing): {path}")
            continue
        fr = grab(path)
        if len(fr) <= 2:
            print(f"skip (too short): {path}")
            continue
        # 每个窗口推进 2 帧 → 逐对比 t 与 t-2，再按块统计（不要先做时间平均，
        # 那会把"每个窗口自己的判断"抹平）
        bh, bw = H // BLK, W // BLK
        per_pair = []
        for t in range(2, len(fr)):
            d = np.abs(fr[t] - fr[t - 2])
            blocks = d[: bh * BLK, : bw * BLK].reshape(bh, BLK, bw, BLK).mean(axis=(1, 3))
            per_pair.append(blocks.reshape(-1))
        flat = np.concatenate(per_pair)          # 所有 (窗口, 块) 样本
        print(f"\n{os.path.basename(path)}  frames={len(fr)}  pairs={len(per_pair)}  "
              f"blocks/frame={flat.size // len(per_pair)}")
        print(f"  (窗口,块) 样本: mean {flat.mean():.2f}  median {np.median(flat):.2f}  "
              f"p90 {np.percentile(flat, 90):.2f}  max {flat.max():.2f}  (0-255)")
        for thr in (0.25, 0.5, 1.0, 2.0, 4.0):
            frac = (flat < thr).mean() * 100
            per_win = np.mean([(p < thr).mean() for p in per_pair]) * 100
            print(f"  |Δt| < {thr:4.2f}/255 : 总占比 {frac:5.1f}%   "
                  f"逐窗口平均 {per_win:5.1f}%   (完全静止的块 {np.mean([(p < 0.05).mean() for p in per_pair])*100:4.1f}%)")


if __name__ == "__main__":
    sys.exit(main())
