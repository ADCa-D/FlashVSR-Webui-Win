# -*- coding: utf-8 -*-
"""1080p overlap sweep: time + PSNR vs ov32 + signed-column seam ratio.

Env: OVS=32,24,16  TILE=128  N=4  SRC=...
Prints ASCII only.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

LOG = os.path.join(ROOT, "work", "ovsweep.log")


def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except OSError:
        pass


import numpy as np  # noqa: E402
import torch  # noqa: E402
import engine as E  # noqa: E402

SRC = os.environ.get("SRC", os.path.join(ROOT, "work", "src_1080p.mp4"))
N = int(os.environ.get("N", "4"))
TILE = int(os.environ.get("TILE", "128"))
SCALE = int(os.environ.get("SCALE", "2"))
OVS = [int(x) for x in os.environ.get("OVS", "32,24,16").split(",")]


def sharp_ratio(img, stride, scale, band=16, guard=64):
    """缝处/非缝 清晰度比：band 内 Laplacian 能量 / band 外（留 guard）能量。

    这是 engine.py:1413 注释里 0.866 -> 0.997 那个口径。比值 <0.95 说明缝处被糊。
    """
    g = img.astype(np.float32).mean(axis=3)                     # (N,H,W)
    lap = (4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1]
           - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]) ** 2             # (N,H-2,W-2)
    H2, W2 = lap.shape[1], lap.shape[2]
    xs = [k * stride * scale - 1 for k in range(1, 300)]
    xs = [x for x in xs if 0 <= x < W2]
    seam = np.zeros_like(lap, dtype=bool)
    for x in xs:
        seam[:, :, max(0, x - band):min(W2, x + band)] = True
    guard_mask = np.zeros_like(lap, dtype=bool)
    for x in xs:
        guard_mask[:, :, max(0, x - guard):min(W2, x + guard)] = True
    other = ~guard_mask
    if seam.sum() == 0 or other.sum() == 0:
        return float("nan")
    a = float(lap[seam].mean())
    b = float(lap[other].mean())
    return a / b if b > 0 else float("nan")


def seam_ratio(img, stride, scale):
    """signed column mean diff; ratio of |d| at tile boundaries vs elsewhere."""
    g = img.astype(np.float32).mean(axis=(0, 3))          # (H,W)
    d = np.abs(np.diff(g, axis=1)).mean(axis=0)           # (W-1,)
    if d.size == 0:
        return float("nan")
    cols = [k * stride * scale for k in range(1, 200)]
    cols = [c for c in cols if 1 <= c < d.size]
    if not cols:
        return float("nan")
    edge = float(np.mean([d[c - 1] for c in cols]))
    base = float(np.median(d))
    return edge / base if base > 0 else float("nan")


eng = E.Engine()
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
frames = E.decode_frames_range(SRC, 0, N, pre_scale=1.0, fps=25.0)
log("=== %s" % time.strftime("%H:%M:%S"))
log("SRC=%dx%d N=%d tile=%d scale=%d" % (frames.shape[2], frames.shape[1], N, TILE, SCALE))

ref = None
for ov in OVS:
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = eng._run_window(frames, SCALE, TILE, 3, tile_overlap=ov)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    a = np.asarray(out)
    if ref is None:
        ref = a
        psnr = float("nan")
        note = "REF"
    else:
        mse = float(np.mean((a.astype(np.float32) - ref.astype(np.float32)) ** 2))
        psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else 99.0
        note = ""
    if os.environ.get("SAVE") == "1":
        try:
            from PIL import Image
            stride = (TILE - ov) * SCALE
            h, w = a.shape[1], a.shape[2]
            g = a.astype(np.float32).mean(axis=3).mean(axis=0)      # (H,W) 平均帧
            e = np.abs(np.diff(g, axis=1)).sum(axis=0)               # 每列能量
            xs = [k * stride for k in range(1, 200) if k * stride < w]
            cx = max(xs, key=lambda x: float(e[max(0, x - 40):x + 40].sum())) if xs else w // 2
            row_e = e[max(0, cx - 8):cx + 8].mean(axis=0)
            ry = int(np.argmax(g[:, max(0, cx - 40):cx + 40].std(axis=1)))
            cy = int(np.clip(ry, 200, h - 200))
            y0, y1 = cy - 200, cy + 200
            x0, x1 = max(0, cx - 260), min(w, cx + 260)
            d = os.path.join(ROOT, "ovcrops")
            os.makedirs(d, exist_ok=True)
            Image.fromarray(a[a.shape[0] // 2, y0:y1, x0:x1]).save(
                os.path.join(d, "ov%02d.png" % ov))
            Image.fromarray(a[a.shape[0] // 2, ::4, ::4]).save(
                os.path.join(d, "full_ov%02d.png" % ov))
            log("saved ovcrops/ov%02d.png crop x=%d..%d y=%d..%d boundary=%d"
                % (ov, x0, x1, y0, y1, cx))
        except Exception as e:  # noqa
            log("save failed: %r" % (e,))
    sr = seam_ratio(a, TILE - ov, SCALE)
    sh = sharp_ratio(a, TILE - ov, SCALE)
    log("ov=%2d  %6.2f s  psnr=%6.2f dB  seam_ratio=%5.2f  sharp_ratio=%5.3f  %s"
        % (ov, dt, psnr, sr, sh, note))
