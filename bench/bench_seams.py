# -*- coding: utf-8 -*-
"""网格接缝（grid artifact）量化 + 羽化/color_fix 对比。

背景：节点把整帧切成 tile 分别推理，再按 `create_feather_mask` 的**线性**斜坡做
加权平均。tile 之间预测有细微亮度/相位差时，线性交叉淡入淡出会在接缝处留下
可见的规则网格（Mach band + DC 台阶）；`color_fix=True` 还会让每个 tile 各自
做一次颜色对齐，边界更容易出现色阶台阶。

本脚本：同一素材、同一参数跑 4 组，逐列/逐行统计 |梯度| 曲线，看 stride 整数倍
位置是否出现尖峰（尖峰比 = p99.9 / 中位数），并给与高重叠参考的 PSNR。

用法：python bench/bench_seams.py [帧数]
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

WORK = os.path.join(ROOT, "work", "seams")
SRC_CLIP = os.path.join(ROOT, "work", "ab_hi_prec", "src_32.mp4")


def smoothstep_mask(mod):
    """把节点的线性羽化换成 smoothstep（C1 连续，去掉接缝处的折线拐点）。"""
    import torch

    def create_feather_mask(size, overlap):
        H, W = size
        n = max(2, int(overlap))
        t = torch.linspace(0.0, 1.0, n)
        ramp = t * t * (3.0 - 2.0 * t)          # smoothstep
        mask = torch.ones(1, 1, H, W)
        mask[:, :, :, :n] = torch.minimum(mask[:, :, :, :n], ramp.view(1, 1, 1, -1))
        mask[:, :, :, -n:] = torch.minimum(mask[:, :, :, -n:], ramp.flip(0).view(1, 1, 1, -1))
        mask[:, :, :n, :] = torch.minimum(mask[:, :, :n, :], ramp.view(1, 1, -1, 1))
        mask[:, :, -n:, :] = torch.minimum(mask[:, :, -n:, :], ramp.flip(0).view(1, 1, -1, 1))
        return mask

    mod.create_feather_mask = create_feather_mask


def read_rgb(path, n):
    import numpy as np
    info = E.probe(path)
    p = subprocess.run([config.FFMPEG, "-v", "error", "-i", path, "-frames:v", str(n),
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    a = np.frombuffer(p.stdout, np.uint8).reshape(-1, info["h"], info["w"], 3)
    return a.astype(np.float32) / 255.0, info


def seam_score(a):
    """(列向尖峰比, 行向尖峰比)：|梯度| 的 p99.9 / 中位数，越大说明接缝越硬。"""
    import numpy as np
    gx = np.abs(np.diff(a, axis=2)).mean(axis=(0, 1, 3))
    gy = np.abs(np.diff(a, axis=1)).mean(axis=(0, 1, 3))

    def ratio(g):
        med = float(np.median(g)) or 1e-6
        return float(np.percentile(g, 99.9)) / med
    return ratio(gx), ratio(gy)


def psnr(a, b):
    import numpy as np
    n = min(len(a), len(b))
    mse = float(((a[:n] - b[:n]) ** 2).mean())
    return 99.0 if mse <= 0 else 10 * math.log10(1.0 / mse)


def main():
    import numpy as np
    nframes = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    if not os.path.isfile(SRC_CLIP):
        print("缺少素材", SRC_CLIP, "（先跑 bench/ab_hi_precision.py）")
        return 1
    os.makedirs(WORK, exist_ok=True)

    fields = dict(preset="custom", scale="2", tile="128", kv="3", chunk="0", mode="tiny-long",
                  attention="sparse", quant="mxfp8", encoder="x265", quality="18", bitdepth="8",
                  budget="auto", prescale="off", preview="0")

    eng = E.Engine()
    print("加载模型…", flush=True)
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
    mod = eng.mod
    orig_mask = mod.create_feather_mask

    # tag, overlap, smoothstep, color_fix
    runs = [
        ("ov24 线性", 24, False, True),
        ("ov24 smooth", 24, True, True),
        ("ov24 smooth+nofix", 24, True, False),
        ("ov64 参考", 64, True, True),
    ]
    outs = {}
    for tag, ov, use_smooth, cf in runs:
        mod.create_feather_mask = orig_mask
        if use_smooth:
            smoothstep_mask(mod)
        p = E.resolve_params(dict(fields, tile_overlap=str(ov),
                                  color_fix="1" if cf else "0"))
        dst = os.path.join(WORK, f"seam_{ov}_{'s' if use_smooth else 'l'}_{'cf' if cf else 'nc'}.mp4")
        print(f"\n=== {tag} ===", flush=True)
        eng.upscale_video(SRC_CLIP, dst, p, on_stage=lambda *a: None)
        outs[tag] = dst

    ref, _ = read_rgb(outs["ov64 参考"], nframes)
    print("\n=== 结果（尖峰比越小越好；PSNR 相对 ov64 参考）===")
    for tag in ("ov24 线性", "ov24 smooth", "ov24 smooth+nofix", "ov64 参考"):
        a, info = read_rgb(outs[tag], nframes)
        rx, ry = seam_score(a)
        ps = psnr(a, ref) if tag != "ov64 参考" else 0.0
        print(f"{tag:18s} 列尖峰比 {rx:5.2f}  行尖峰比 {ry:5.2f}  "
              f"PSNR {ps:6.2f} dB  ({info['w']}x{info['h']})")


if __name__ == "__main__":
    sys.exit(main())
