# -*- coding: utf-8 -*-
"""抽帧复用的收益上限：给一段视频算出 keep/drop 映射，并报告各项数字。

思路（用户提的）：把重复帧在**输入侧就抽走**，推理只跑"唯一画稿"，
输出时按映射把帧插回去。好处：
  - DiT 和 decoder 一起省（不止 decoder），
  - 动画一拍二 = 一半的帧根本不用算，
  - 不需要在流水线里做逐 tile 判定。

输出：
  keep 索引、drop 索引、可省帧占比、按 8n+5 窗口算的重排后帧数
"""
import json
import os
import subprocess
import sys

import numpy as np

FFMPEG = r"D:\ffmpeg-9.0.1\ffmpeg.exe"


def decode_gray(path, w, h, frames=400):
    cmd = [FFMPEG, "-v", "error", "-i", path, "-frames:v", str(frames),
           "-vf", f"scale={w}:{h}", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    out = subprocess.run(cmd, capture_output=True)
    buf = np.frombuffer(out.stdout, dtype=np.uint8)
    n = buf.size // (w * h)
    return buf[: n * w * h].reshape(n, h, w)


def build_map(g, thr=0.5):
    """返回 keep 列表：第一个帧永远保留；后续帧与"上一个保留帧"比较。"""
    keep = [0]
    last = g[0].astype(np.float32)
    for i in range(1, len(g)):
        cur = g[i].astype(np.float32)
        if np.abs(cur - last).mean() >= thr:
            keep.append(i)
            last = cur
        # 被丢的帧不更新 last：这样"连续 3 帧同画面"只保留 1 帧
    return keep


def main():
    if len(sys.argv) < 2:
        print("用法: python bench/probe_drop_map.py <video> [thr]")
        return 2
    path = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    g = decode_gray(path, 512, 288)
    n = len(g)
    keep = build_map(g, thr)
    drop = [i for i in range(n) if i not in set(keep)]
    print(f"\n{os.path.basename(path)}  帧数 {n}  阈值 |Δ|>{thr}/255")
    print(f"  保留 {len(keep)} 帧，抽掉 {len(drop)} 帧 = 省 {len(drop)/n*100:.1f}% 计算")
    # 8n+5 结构的窗口数（每窗 2 帧推进，首窗 6 帧）
    for tag, cnt in (("原始", n), ("抽帧后", len(keep))):
        proc = max(0, (cnt - 1) // 8 - 2)
        print(f"  {tag}: {cnt} 帧 → {proc} 个流式步（首窗 6 帧 + 每步 2 帧）")

    # 连拍长度分布
    runs, cur = [], 1
    for i in range(1, len(keep)):
        if keep[i] - keep[i - 1] == 1:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    print(f"  唯一画面连拍长度: {sorted(set(runs))}  最长 {max(runs)}")

    out = os.path.join(os.path.dirname(path), "drop_map.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n": n, "thr": thr, "keep": keep}, f)
    print(f"  映射已存: {out}")


if __name__ == "__main__":
    sys.exit(main())
