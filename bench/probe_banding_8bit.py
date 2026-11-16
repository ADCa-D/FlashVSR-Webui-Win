# -*- coding: utf-8 -*-
"""抖动到底该放在哪一环？——必须放在"float → 8bit"那一步才有效。

背景：预缩放阶段 libplacebo 输出的是 `gbrpf32le`（浮点），**没有 8bit 目标可抖**，
所以它的 dither 基本是空转；真正的量化发生在编码前那一跳（float → yuv420p 8bit）。

变体（10bit 渐变 → 960x540 → 8bit H.264 CRF 23）：
  A         ewa 缩放，8bit 直接转（无抖动）
  A_z1      + zscale dither=ordered
  A_z3      + zscale dither=error_diffusion
  D         ewa + libplacebo:deband=1:dithering=blue（浮点输出，抖动无落点）
  C         ewa → 1:1 再跑 extra_opts=preset=high_quality
"""
import os
import subprocess

import numpy as np

ROOT = r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web"
OUT = os.path.join(ROOT, "work", "banding")
FF = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
PNG = os.path.join(OUT, "ramp10.png")
W, H = 960, 540
CRF = 23

SCALE = f"libplacebo=w={W}:h={H}:downscaler=ewa_lanczos4sharpest"
VARIANTS = {
    "A_plain": SCALE + ",format=yuv420p",
    "A_z_ordered": SCALE + ",zscale=dither=ordered,format=yuv420p",
    "A_z_errordiff": SCALE + ",zscale=dither=error_diffusion,format=yuv420p",
    "D_lp_dither": SCALE + ":deband=1:dithering=blue,format=yuv420p",
    "C_chain_hq": (f"{SCALE},libplacebo=w={W}:h={H}:extra_opts=preset=high_quality,"
                   f"format=yuv420p"),
}


def encode(vf, dst):
    cmd = [FF, "-v", "error", "-y", "-init_hw_device", "vulkan=pl",
           "-filter_hw_device", "pl", "-i", PNG, "-frames:v", "1",
           "-vf", vf + ",hwdownload" if "libplacebo" in vf else vf,
           "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", "yuv420p", dst]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        err = (p.stderr or "").strip().splitlines()
        return False, err[-1][:90] if err else "?"
    return True, ""


def stats(path):
    txt = subprocess.run([FF, "-v", "error", "-i", path, "-select_streams", "v:0",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0"],
                         capture_output=True, text=True, errors="replace").stdout
    num = [int(x) for x in txt.replace("\n", ",").split(",") if x.strip().isdigit()]
    w, h = num[0], num[1]
    buf = subprocess.run([FF, "-v", "error", "-i", path, "-f", "rawvideo",
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
    if not os.path.exists(PNG):
        print("缺渐变素材:", PNG)
        return 1
    print(f"10bit 渐变 → {W}x{H} → 8bit CRF {CRF}")
    print(f"（色带 = 大段完全相同的灰阶；'长段占比' 越低越好）\n")
    print(f"{'变体':16s} {'不同取值':>8s} {'最长同值段':>10s} {'长段占比':>9s}")
    for tag, vf in VARIANTS.items():
        dst = os.path.join(OUT, f"{tag}.mp4")
        ok, err = encode(vf, dst)
        if not ok:
            print(f"{tag:16s} 失败: {err}")
            continue
        uniq, maxrun, frac = stats(dst)
        print(f"{tag:16s} {uniq:8d} {maxrun:10d} {frac * 100:8.1f}%")


if __name__ == "__main__":
    main()
