# -*- coding: utf-8 -*-
"""对任意素材评估"静止复用"的上限，并按内容类型给不同判据。

两种内容性质完全不同：

- **动画（一拍二/一拍三）**：同一张画稿连拍 2–3 帧 → 存在**逐字节相同**的帧对。
  这类可以走"完全重复帧"路径：判据是精确相等（或 mpdecimate 的判定），
  复用后画质 **零损失**（PSNR = inf），不需要阈值、不需要迟滞。

- **实拍（电影）**：几乎没有完全重复帧，只有"低频区域 + 缓动"。要先看
  I/P/B 的分布（I 帧≈场景切换，不能复用）和 P/B 帧的残差大小，
  再决定阈值；复用是有损的，必须 VMAF 验收。

输出：
  1) 完全重复帧占比与被复用的帧区间
  2) ffprobe 的帧类型（I/P/B）分布 + 包大小分位
  3) 128px tile 的静止占比（当前 tile 级判据能吃到多少）
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

FFMPEG = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
FFPROBE = os.path.join(os.path.dirname(FFMPEG), "ffprobe.exe")
BLK = 128


def decode_gray(path, w, h, frames=200):
    cmd = [FFMPEG, "-v", "error", "-i", path, "-frames:v", str(frames),
           "-vf", f"scale={w}:{h}", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    out = subprocess.run(cmd, capture_output=True)
    buf = np.frombuffer(out.stdout, dtype=np.uint8)
    n = buf.size // (w * h)
    return buf[: n * w * h].reshape(n, h, w)


def frame_types(path, limit=200):
    cmd = [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_frames",
           "-show_entries", "frame=pict_type,pkt_size", "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    try:
        fr = json.loads(out.stdout).get("frames", [])[:limit]
    except Exception:
        return [], []
    types = [f.get("pict_type") for f in fr]
    sizes = [int(f.get("pkt_size") or 0) for f in fr]
    return types, sizes


def main():
    if len(sys.argv) < 2:
        print("用法: python bench/probe_reuse_potential.py <video> [宽 高]")
        return 2
    path = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    h = int(sys.argv[3]) if len(sys.argv) > 3 else 288
    if not os.path.exists(path):
        print("找不到:", path)
        return 2

    g = decode_gray(path, w, h)
    n = len(g)
    print(f"\n{os.path.basename(path)}  {w}x{h}  分析 {n} 帧")

    # ---- 1) 完全重复帧 + 近似重复帧 ---------------------------------------
    hashes = [hashlib.md5(frame.tobytes()).hexdigest() for frame in g]
    dup = sum(1 for i in range(1, n) if hashes[i] == hashes[i - 1])
    d1 = np.array([np.abs(g[i].astype(np.float32) - g[i - 1]).mean() for i in range(1, n)])
    runs, cur = [], 1
    for i in range(1, n):
        if hashes[i] == hashes[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    print(f"  逐字节相同帧: {dup}/{n-1} = {dup/max(1,n-1)*100:.1f}%   "
          f"连拍长度 {sorted(set(runs))}（最长 {max(runs)}）")
    print(f"  帧间平均差: median {np.median(d1):.2f}  p10 {np.percentile(d1,10):.2f}  "
          f"p90 {np.percentile(d1,90):.2f}  (0-255)")
    # 编码噪声会让"同一张画稿"不再逐字节相同，所以还要看近似重复
    for thr in (0.05, 0.5, 1.0, 2.0):
        f = (d1 < thr).mean() * 100
        print(f"  近似重复帧 |Δ|<{thr:4.2f}/255: {f:5.1f}%")

    # ---- 1b) 是否"一拍二 / 一拍三"（周期性重复）--------------------------
    for period in (2, 3):
        flags = d1 < 1.0
        # 每个相位上"重复帧"的占比：一拍二 应该是 1/2 的帧重复、且集中在固定相位
        ph = [flags[np.arange(len(flags)) % period == r].mean() * 100 for r in range(period)]
        spread = max(ph) - min(ph)
        print(f"  周期 {period} 检测: 各相位重复率 {[round(x,1) for x in ph]}%  "
              f"相位差 {spread:.1f}pp {'← 明显周期性（一拍二/三）' if spread > 20 else ''}")

    # ---- 2) 帧类型 / 包大小（实拍判据）------------------------------------
    types, sizes = frame_types(path)
    if types:
        from collections import Counter
        c = Counter(types)
        print(f"  帧类型: {dict(c)}   （I 帧≈场景切换，不可复用）")
        if sizes:
            s = np.array(sizes, dtype=np.float64)
            print(f"  包大小: median {np.median(s):.0f}B  p10 {np.percentile(s,10):.0f}B  "
                  f"p90 {np.percentile(s,90):.0f}B")
    else:
        print("  帧类型: 取不到（容器可能没有索引）")

    # ---- 3) tile 级静止占比（当前判据能吃到多少）-------------------------
    bh, bw = h // BLK, w // BLK
    if bh and bw:
        acc = []
        for t in range(2, n):
            d = np.abs(g[t].astype(np.float32) - g[t - 2])
            blk = d[: bh * BLK, : bw * BLK].reshape(bh, BLK, bw, BLK).mean(axis=(1, 3))
            acc.append(blk.reshape(-1))
        flat = np.concatenate(acc)
        for thr in (0.05, 0.5, 2.0):
            print(f"  tile(128px) |Δt|<{thr:4.2f}/255 占比: {(flat < thr).mean()*100:5.1f}%")
    else:
        print(f"  分辨率 {w}x{h} 小于 {BLK}px，tile 统计跳过")


if __name__ == "__main__":
    sys.exit(main())
