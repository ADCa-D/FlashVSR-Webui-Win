# -*- coding: utf-8 -*-
"""色带到底出在哪：我们自己那行 `(x*255).round().to(uint8)` 没有抖动。

流程（完全绕开 libplacebo，因为量化不在那儿）：
  float 渐变 → 三种量化（纯 round / 1LSB 随机三角抖动 / 8×8 有序抖动）
  → pipe 给 x264 CRF 23 压 8bit → 解码量色带

指标：不同灰阶取值数、每行最长同值连跑、>8 px 的长段占比。
色带 = 大段完全相同的像素，所以后两个越低越好。
"""
import os
import subprocess

import numpy as np

ROOT = r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web"
OUT = os.path.join(ROOT, "work", "banding2")
FF = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
W, H, FPS, N = 960, 540, 25, 10
CRF = 23


def make_float_ramp():
    # 水平慢渐变 + 一点竖直变化，模拟天空/墙面
    x = np.linspace(0.10, 0.32, W, dtype=np.float32)[None, :]
    y = np.linspace(-0.01, 0.01, H, dtype=np.float32)[:, None]
    plane = np.clip(x + y, 0, 1)
    return np.repeat(plane[None, :, :, None], 3, axis=3)      # [H,W,3]


def quant_round(v):
    return (np.clip(v, 0, 1) * 255.0).round().astype(np.uint8)


def quant_dither(v, seed=0):
    """1 LSB 三角分布抖动（TPDF）：标准做法，抑制色带且不引入偏置。"""
    rng = np.random.default_rng(seed)
    x = np.clip(v, 0, 1) * 255.0
    noise = rng.random(x.shape, dtype=np.float32) - rng.random(x.shape, dtype=np.float32)
    return np.clip(x + noise, 0, 255).round().astype(np.uint8)


def quant_ordered(v):
    """8×8 Bayer 有序抖动：确定性的，逐帧不变，不额外增加码率。"""
    bayer = np.array([[0, 32, 8, 40, 2, 34, 10, 42],
                      [48, 16, 56, 24, 50, 18, 58, 26],
                      [12, 44, 4, 36, 14, 46, 6, 38],
                      [60, 28, 52, 20, 62, 30, 54, 22],
                      [3, 35, 11, 43, 1, 33, 9, 41],
                      [51, 19, 59, 27, 49, 17, 57, 25],
                      [15, 47, 7, 39, 13, 45, 5, 37],
                      [63, 31, 55, 23, 61, 29, 53, 21]], dtype=np.float32) / 64.0 - 0.5
    h, w = v.shape[-3], v.shape[-2]
    tile = np.tile(bayer, (h // 8 + 1, w // 8 + 1))[:h, :w]
    x = np.clip(v, 0, 1) * 255.0
    noise = np.broadcast_to(tile[None, :, :, None], x.shape)
    return np.clip(x + noise, 0, 255).round().astype(np.uint8)


def encode(u8, dst):
    h, w = u8.shape[1], u8.shape[2]
    cmd = [FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", "yuv420p", dst]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(0, len(u8), 4):
        p.stdin.write(u8[i:i + 4].tobytes())
    p.stdin.close()
    err = p.stderr.read().decode("utf-8", "ignore")
    if p.wait() != 0:
        print("  编码失败:", err.strip()[-160:])
        return False
    return True


def stats(path):
    txt = subprocess.run([FF, "-v", "error", "-i", path, "-select_streams", "v:0",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0"],
                         capture_output=True, text=True, errors="replace").stdout
    num = [int(x) for x in txt.replace("\n", ",").split(",") if x.strip().isdigit()]
    w, h = num[0], num[1]
    buf = subprocess.run([FF, "-v", "error", "-i", path, "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True).stdout
    g = np.frombuffer(buf[: w * h], np.uint8).reshape(h, w).astype(np.int16)
    rows = g[h // 3: h // 3 + 8, :]
    uniq = len(np.unique(rows))
    maxrun, long_cnt, total = 0, 0, 0
    for r in rows:
        idx = np.flatnonzero(np.diff(r) != 0)
        runs = np.diff(np.concatenate(([-1], idx, [len(r) - 1])))
        maxrun = max(maxrun, int(runs.max()))
        long_cnt += int((runs > 8).sum())
        total += len(runs)
    return uniq, maxrun, long_cnt / max(1, total)


def main():
    os.makedirs(OUT, exist_ok=True)
    base = make_float_ramp()
    frames = np.repeat(base[None], N, axis=0)
    cases = {
        "round(现状)": quant_round(frames),
        "TPDF抖动": quant_dither(frames),
        "Bayer8x8": quant_ordered(frames),
    }
    print(f"{W}x{H} float 渐变 → 8bit x264 CRF {CRF}（{N} 帧）\n")
    print(f"{'量化方式':14s} {'不同取值':>8s} {'最长同值段':>10s} {'长段占比':>9s} {'码率':>8s}")
    for tag, u8 in cases.items():
        dst = os.path.join(OUT, f"{tag.replace('(', '_').replace(')', '')}.mp4")
        if not encode(u8, dst):
            continue
        uniq, maxrun, frac = stats(dst)
        kb = os.path.getsize(dst) / 1024
        print(f"{tag:14s} {uniq:8d} {maxrun:10d} {frac*100:8.1f}% {kb:7.0f}K")


if __name__ == "__main__":
    main()
