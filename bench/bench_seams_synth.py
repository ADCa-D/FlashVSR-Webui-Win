# -*- coding: utf-8 -*-
"""在「平滑合成素材」上量化 tile 网格伪影。

为什么不用实拍素材：真实画面里 p99.9 的梯度全都是内容边缘，接缝被淹没，
上一版指标（p99.9/中位数）在 ov24 和 ov64 上都是 ~30，分不出好坏。

换成缓慢平移的平滑渐变后，画面本身没有高频，任何**周期性**的梯度尖峰/亮暗台阶
都只能来自 tile 拼接。指标：
  1. 缝处高频梯度 / 其余平均      —— 硬接缝
  2. 缝处低频（x 方向平滑后）台阶   —— 羽化带造成的亮暗带（Mach band）
两者都按 stride×scale 的整数倍位置取样。

用法：python bench/bench_seams_synth.py [帧数]
"""
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import config  # noqa: E402
import engine as E  # noqa: E402

WORK = os.path.join(ROOT, "work", "seams_synth")
SRC = os.path.join(WORK, "gradient_960x544.mp4")


def make_src(nframes):
    os.makedirs(WORK, exist_ok=True)
    if os.path.isfile(SRC):
        return SRC
    # 平滑对角渐变（1120×624 画布）+ 0.6 px/帧 的缓慢横移，画面本身没有高频
    vf = ("geq=lum='(X+Y)*255/(W+H-2)':cb=128:cr=128,"
          "crop=960:544:x='(iw-960)/2+t*0.6':y='(ih-544)/2',format=yuv420p")
    cmd = [config.FFMPEG, "-v", "error", "-y", "-f", "lavfi",
           "-i", f"color=c=black:s=1120x624:r=25:d={max(1, nframes / 25.0):.2f}",
           "-vf", vf, "-frames:v", str(nframes),
           "-c:v", "libx264", "-crf", "4", "-pix_fmt", "yuv420p", SRC]
    subprocess.run(cmd, check=True)
    return SRC


def read_frames(path, n):
    import numpy as np
    info = E.probe(path)
    p = subprocess.run([config.FFMPEG, "-v", "error", "-i", path, "-frames:v", str(n),
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    a = np.frombuffer(p.stdout, np.uint8).reshape(-1, info["h"], info["w"], 3)
    return a.astype(np.float32) / 255.0, info


def seam_metrics(a, stride_out):
    """返回 (高频缝比, 低频台阶幅度) —— 越小越好。"""
    import numpy as np
    g = np.abs(np.diff(a, axis=2)).mean(axis=(0, 1, 3))          # 每列 |dx|
    g_s = np.convolve(g, np.ones(33) / 33.0, mode="same")        # 低频版
    W = a.shape[2]

    seam_cols = [x for x in range(stride_out - 1, W - 2, stride_out)]
    if not seam_cols:
        return 0.0, 0.0

    def at(profile, half=2):
        idx = []
        for x in seam_cols:
            idx += [max(0, min(W - 1, x + d)) for d in range(-half, half + 1)]
        m = np.zeros(len(profile), bool)
        m[idx] = True
        return float(profile[m].mean()), float(profile[~m].mean())

    hi_seam, hi_base = at(g, 1)
    lo_seam, lo_base = at(g_s, 2)
    # 低频台阶：缝处斜率相对整体斜率的变化倍数
    lo_ratio = lo_seam / max(lo_base, 1e-9)
    return hi_seam / max(hi_base, 1e-9), lo_ratio


def main():
    nframes = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    src = make_src(nframes)

    fields = dict(preset="custom", scale="2", tile="128", kv="3", chunk="0", mode="tiny-long",
                  attention="sparse", quant="mxfp8", encoder="x265", quality="10", bitdepth="8",
                  budget="auto", prescale="off", preview="0")

    eng = E.Engine()
    print("加载模型…", flush=True)
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")

    runs = [
        ("ov12 线性 (旧默认)", 12, "off", True),
        ("ov24 线性", 24, "off", True),
        ("ov24 smooth", 24, "smooth", True),
        ("ov24 smooth+align", 24, "align", True),
        ("ov48 参考", 48, "smooth", True),
    ]
    outs = []
    for tag, ov, seam, cf in runs:
        p = E.resolve_params(dict(fields, tile_overlap=str(ov), seam_fix=seam,
                                  color_fix="1" if cf else "0"))
        dst = os.path.join(WORK, f"s_{ov}_{seam}.mp4")
        if os.path.isfile(dst):
            print(f"\n=== {tag} === （已有输出，跳过）", flush=True)
            outs.append((tag, dst, ov))
            continue
        print(f"\n=== {tag} ===", flush=True)
        eng.upscale_video(src, dst, p, on_stage=lambda *a: None)
        outs.append((tag, dst, ov))

    print("\n=== 结果（两个比值都越小越好；理论最优 = 1.00）===")
    print(f"{'变体':20s} {'高频缝比':>8s} {'低频台阶':>8s}")
    for tag, dst, ov in outs:
        a, info = read_frames(dst, nframes)
        hi, lo = seam_metrics(a, (128 - ov) * 2)
        print(f"{tag:20s} {hi:8.2f} {lo:8.3f}   ({info['w']}x{info['h']}, stride={(128-ov)*2})")


if __name__ == "__main__":
    sys.exit(main())
