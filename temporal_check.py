# -*- coding: utf-8 -*-
"""时序一致性检查（纯 CPU，不占 GPU）

指标：
1. 帧间差分能量 diff[t] = mean|x_t - x_{t-1}|（灰度、缩到 --width 宽）
2. 静态区亮度抖动：全片 mean 亮度序列的高频分量 RMS
3. 分段边界检测：把 diff 序列按段边界切开，看边界处的 diff 是否显著高于段内中位数
4. 与参照对比：可传入源视频，按同尺寸 bicubic 放大后算同样的指标，给出倍率

用法：
  python temporal_check.py 成片.mp4 [--src 源.mp4] [--stride 24] [--width 480]
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
        "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
        "-of", "csv=p=0", path,
    ]).decode().strip().split(",")
    w, h = int(out[0]), int(out[1])
    try:
        n = int(out[2])
    except ValueError:
        n = 0
    return w, h, n


def gray_frames(path, width):
    """解码成灰度、宽度固定的小图序列（uint8，用于帧间差分）。"""
    proc = subprocess.Popen(
        [FFMPEG, "-v", "error", "-i", path, "-vf", f"scale={width}:-2:flags=area,format=gray",
         "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    w, h, _ = probe(path)
    hh = int(round(h * width / w / 2)) * 2
    fb = width * hh
    frames = []
    while True:
        buf = proc.stdout.read(fb)
        if not buf or len(buf) < fb:
            break
        frames.append(np.frombuffer(buf, np.uint8).reshape(hh, width))
    proc.wait()
    return np.stack(frames) if frames else None


def metrics(frames):
    f = frames.astype(np.float32)
    diff = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))   # 每对相邻帧
    lum = f.mean(axis=(1, 2))
    hf = np.abs(np.diff(lum, n=2)) if len(lum) > 2 else np.zeros(1)
    return diff, lum, float(np.sqrt((hf ** 2).mean()))


def boundary_report(diff, stride):
    """按 stride 分段，检查段边界处 diff 是否异常。diff[i] = 帧 i 与 i+1 之差。"""
    n = len(diff)
    if stride <= 0 or n <= stride * 2:
        return None
    med = float(np.median(diff))
    mad = float(np.median(np.abs(diff - med))) or 1e-6
    z = (diff - med) / (1.4826 * mad)
    bounds = list(range(stride, n, stride))
    edge_z = [float(z[b - 1]) for b in bounds if 0 <= b - 1 < n]
    return dict(n_segments=len(bounds) + 1, median=med,
                edge_z_mean=float(np.mean(edge_z)) if edge_z else 0.0,
                edge_z_max=float(np.max(edge_z)) if edge_z else 0.0,
                overall_z_max=float(np.max(z)), overall_z_p99=float(np.percentile(z, 99)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--src", default=None, help="源视频，用于倍率对比（可省略）")
    ap.add_argument("--stride", type=int, default=24, help="分段净步长（默认 24，对齐 chunk32-overlap8）")
    ap.add_argument("--width", type=int, default=480)
    a = ap.parse_args()

    fr = gray_frames(a.video, a.width)
    if fr is None:
        print("解码失败")
        return 2
    diff, lum, hf = metrics(fr)
    print(f"== {a.video}")
    print(f"帧数 {len(fr)}   平均帧间差分 {diff.mean():.3f}   亮度高频抖动 RMS {hf:.4f}")

    rep = boundary_report(diff, a.stride)
    if rep:
        print(f"分段参考 stride={a.stride}：共 {rep['n_segments']} 段")
        print(f"  段边界 z 均值 {rep['edge_z_mean']:.2f}  最大 {rep['edge_z_max']:.2f}")
        print(f"  全片 z 最大 {rep['overall_z_max']:.2f}  p99 {rep['overall_z_p99']:.2f}")
        verdict = "边界无异常" if rep['edge_z_mean'] < 1.0 else "边界偏大，建议加 overlap 或做 carry-state"
        print(f"  判定：{verdict}")

    if a.src:
        sf = gray_frames(a.src, a.width)
        if sf is not None:
            sd, sl, shf = metrics(sf)
            # 源通常帧数/时序对齐，直接比平均差分能量（内容相同则可比）
            ratio = diff.mean() / max(sd.mean(), 1e-6)
            print(f"== 参照 {a.src}")
            print(f"平均帧间差分 {sd.mean():.3f}   亮度高频 RMS {shf:.4f}")
            print(f"倍率（成片/源）= {ratio:.2f}   "
                  f"{'正常（≈1）' if 0.8 <= ratio <= 1.2 else '偏高，存在额外闪烁' if ratio > 1.2 else '偏低（可能过度平滑）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
