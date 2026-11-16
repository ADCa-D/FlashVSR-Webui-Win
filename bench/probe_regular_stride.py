# -*- coding: utf-8 -*-
"""规整步长抽帧：给定 (p, phase)，判断"丢掉的那些帧是不是前一张的复制"。

关键修正：一拍二里 |Δ(i, i−2)| 比的是**不同画稿**，永远不会小；
真正该看的是 |Δ(i, i−1)|（相邻帧）——被丢的帧必须与它的前一张几乎相同。

规则一旦是 (p, phase)，插回就是"每帧重复 p 次"，不需要 keep 表：
    keep = [phase, phase+p, phase+2p, ...]
    expand(out_frames) = 每个输出帧重复 p 次（末尾按原帧数截断）
"""
import os
import subprocess
import sys

import numpy as np

FFMPEG = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
THR = 0.5
CHUNK = 64


def decode_gray(path, w, h, frames=200):
    cmd = [FFMPEG, "-v", "error", "-i", path, "-frames:v", str(frames),
           "-vf", f"scale={w}:{h}", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    out = subprocess.run(cmd, capture_output=True)
    buf = np.frombuffer(out.stdout, dtype=np.uint8)
    n = buf.size // (w * h)
    return buf[: n * w * h].reshape(n, h, w).astype(np.float32)


def d1_of(g):
    return np.array([np.abs(g[i] - g[i - 1]).mean() for i in range(1, len(g))])  # d1[i-1]


def eval_regular(g, p, phase, thr=THR, d1=None):
    """返回 (被丢帧中'不是复制'的比例, 保留帧数)。d1 索引对应 (i-1)->(i)。"""
    n = len(g)
    if d1 is None:
        d1 = d1_of(g)
    keep = set(range(phase, n, p))
    bad = 0
    dropped = 0
    for i in range(1, n):
        if i in keep:
            continue
        dropped += 1
        if d1[i - 1] >= thr:      # 这一帧是真的新画面，丢了就丢信息
            bad += 1
    return (bad / dropped if dropped else 0.0), len(keep)


def arbitrary_keep(g, thr=THR):
    keep, last = [0], g[0]
    for i in range(1, len(g)):
        if np.abs(g[i] - last).mean() >= thr:
            keep.append(i)
            last = g[i]
    return keep


def main():
    if len(sys.argv) < 2:
        print("用法: python bench/probe_regular_stride.py <video>")
        return 2
    path = sys.argv[1]
    g = decode_gray(path, 512, 288)
    n = len(g)
    d1 = d1_of(g)
    print(f"\n{os.path.basename(path)}  {n} 帧   相邻帧 |Δ|<{THR}/255 的占比 "
          f"{(d1 < THR).mean()*100:.1f}%")

    ka = arbitrary_keep(g)
    print(f"  A 任意 keep 表: 保留 {len(ka)}  省 {(1-len(ka)/n)*100:5.1f}%")

    best = None
    for p in (2, 3):
        for ph in range(p):
            bad, kept = eval_regular(g, p, ph, d1=d1)
            tag = ""
            if best is None or bad < best[0]:
                best = (bad, p, ph, kept)
            print(f"  B p={p} phase={ph}: 保留 {kept:3d}（省 {100*(1-kept/n):5.1f}%）"
                  f"  丢错率 {bad*100:5.1f}%{tag}")

    bad, p, ph, kept = best
    print(f"\n  最佳规整规则: p={p}, phase={ph} → 保留 {kept}（省 {100*(1-kept/n):.1f}%）"
          f"  丢错率 {bad*100:.1f}%")
    print(f"  插回规则: 每个输出帧重复 {p} 次（最后按原帧数 {n} 截断），无需映射表")

    # 逐 chunk 稳定性（决定"整片能不能用同一个 p"）
    print(f"  逐 {CHUNK} 帧 chunk 的丢错率（门槛 10%）:")
    need_split = 0
    for s in range(0, n, CHUNK):
        seg = g[s:min(s + CHUNK, n)]
        if len(seg) < 4:
            continue
        b, _ = eval_regular(seg, p, ph)
        mark = "" if b < 0.10 else "  ← 这段要退回 p=1"
        if b >= 0.10:
            need_split += 1
        print(f"    [{s:4d},{s+len(seg):4d})  丢错率 {b*100:5.1f}%{mark}")
    print(f"  → {'需要分段切换 p' if need_split else '整片可以统一使用'}："
          f"{need_split} 个 chunk 超标")


if __name__ == "__main__":
    sys.exit(main())
