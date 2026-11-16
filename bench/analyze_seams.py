# -*- coding: utf-8 -*-
"""在**已产出的 4K 成片**上找网格接缝：带符号的列间台阶 + 自相关求周期。

用法：python bench/analyze_seams.py <成片.mp4> [帧数] [预期 stride]
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import config  # noqa: E402
import engine as E  # noqa: E402


def read(path, n, scale_down=1):
    import numpy as np
    info = E.probe(path)
    vf = f"scale=iw/{scale_down}:ih/{scale_down}" if scale_down > 1 else None
    cmd = [config.FFMPEG, "-v", "error", "-i", path, "-frames:v", str(n)]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.run(cmd, capture_output=True)
    w = info["w"] // scale_down
    h = info["h"] // scale_down
    a = np.frombuffer(p.stdout, np.uint8).astype("float32").reshape(-1, h, w, 3) / 255.0
    return a, info


def main():
    import numpy as np
    path = sys.argv[1]
    nframes = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    expect = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    a, info = read(path, nframes)
    H, W = a.shape[1], a.shape[2]
    print(f"{os.path.basename(path)}  {info['w']}x{info['h']}  取 {len(a)} 帧")

    # 带符号台阶：沿 x 的列间均值差（接缝是系统性亮暗跳变，符号不会被"平均掉"）
    s = (a[:, :, 1:, :] - a[:, :, :-1, :]).mean(axis=(0, 1, 3))     # 长度 W-1
    sa = np.abs(s)
    print(f"|台阶| 中位数 {np.median(sa):.5f}  均值 {sa.mean():.5f}  p99 {np.percentile(sa, 99):.5f}")

    # 自相关找周期（去掉均值再算）
    x = sa - sa.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0]
    lo, hi = 40, min(len(ac) - 1, 1200)
    seg = ac[lo:hi]
    period = lo + int(np.argmax(seg))
    print(f"自相关主周期 ≈ {period} px  (峰值 {ac[period]:.3f})")

    target = expect or period
    idx = [k * target for k in range(1, (W - 1) // target + 1)]
    if idx:
        seam = float(np.mean([sa[max(0, i - 2):i + 3].mean() for i in idx]))
        mask = np.ones(len(sa), bool)
        for i in idx:
            mask[max(0, i - 8):i + 9] = False
        base = float(sa[mask].mean())
        print(f"按 stride={target} 取样：缝处 |台阶| {seam:.5f}  非缝 {base:.5f}  比值 {seam / max(base,1e-9):.2f}")
        top = sorted(range(len(sa)), key=lambda i: -sa[i])[:8]
        print("|台阶| 最大的 8 列：", [int(i) for i in sorted(top)])


if __name__ == "__main__":
    main()
