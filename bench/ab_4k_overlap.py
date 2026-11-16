# -*- coding: utf-8 -*-
"""4K 素材上对比 tile256 的 overlap 12 vs 64：接缝处的"清晰度跌落"与人眼裁图。

用法：python bench/ab_4k_overlap.py <源视频> [帧数]
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import config  # noqa: E402
import engine as E  # noqa: E402

WORK = os.path.join(ROOT, "work", "ab_4k_overlap")


def run(eng, src, tag, ov, nframes):
    fields = dict(preset="custom", scale="2", tile="256", kv="3", chunk="16",
                  mode="tiny-long", attention="sparse", quant="nvfp4", encoder="x265",
                  quality="20", bitdepth="8", budget="auto", prescale="off", preview="0",
                  tile_overlap=str(ov))
    p = E.resolve_params(fields)
    dst = os.path.join(WORK, f"{tag}.mp4")
    eng.upscale_video(src, dst, p, on_stage=lambda *a: None)
    return dst, p


def hf_profile(path, n):
    import numpy as np
    info = E.probe(path)
    q = subprocess.run([config.FFMPEG, "-v", "error", "-i", path, "-frames:v", str(n),
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    a = np.frombuffer(q.stdout, np.uint8).astype("float32").reshape(-1, info["h"], info["w"], 3) / 255.0
    g = a.mean(axis=3)
    d2 = np.abs(g[:, :, 2:] - 2 * g[:, :, 1:-1] + g[:, :, :-2]).mean(axis=(0, 1))
    return np.convolve(d2, np.ones(15) / 15.0, mode="same"), (info["w"], info["h"])


def dip(prof, stride):
    import numpy as np
    idx = [k * stride for k in range(1, (len(prof) - 1) // stride + 1)]
    seam = float(np.mean([prof[max(0, i - 8):i + 9].mean() for i in idx]))
    mask = np.ones(len(prof), bool)
    for i in idx:
        mask[max(0, i - 16):i + 17] = False
    base = float(prof[mask].mean())
    return seam, base, seam / max(base, 1e-9), idx


def sharpest_crop(path, n, seam_idx, size=(420, 320)):
    """挑一张「局部最锐利」的裁图：纹理最多的位置接缝最显眼。"""
    import numpy as np
    info = E.probe(path)
    q = subprocess.run([config.FFMPEG, "-v", "error", "-i", path, "-frames:v", str(n),
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    a = np.frombuffer(q.stdout, np.uint8).astype("float32").reshape(-1, info["h"], info["w"], 3)
    g = a.mean(axis=3)
    w, h = size
    cx = seam_idx[len(seam_idx) // 2]
    best, bestv = None, -1
    for y in range(200, info["h"] - h - 200, 160):
        patch = g[:, y:y + h, max(0, cx - w // 2):max(0, cx - w // 2) + w]
        v = float(np.abs(np.diff(patch, axis=2)).mean())
        if v > bestv:
            bestv, best = v, y
    x = max(0, cx - w // 2)
    vf = f"crop={w}:{h}:{x}:{best},scale=iw*2:ih*2:flags=neighbor"
    out = os.path.join(WORK, os.path.splitext(os.path.basename(path))[0] + f"_seam_x{cx}.png")
    subprocess.run([config.FFMPEG, "-v", "error", "-y", "-i", path, "-vf", vf,
                    "-frames:v", "1", out], check=True)
    return out, bestv, (x, best)


def main():
    src = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    os.makedirs(WORK, exist_ok=True)
    print("源：", src, E.probe(src))

    eng = E.Engine()
    print("加载模型…", flush=True)
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="nvfp4")

    for tag, ov in (("ov12", 12), ("ov64", 64)):
        dst, p = run(eng, src, tag, ov, n)
        prof, (w, h) = hf_profile(dst, n)
        stride = 256 * p["scale"] - ov * p["scale"]
        seam, base, ratio, idx = dip(prof, p["scale"] * (256 - ov))
        print(f"\n{tag}: tile256 → stride(输出) {p['scale']*(256-ov)}  {w}x{h}")
        print(f"  缝处 HF {seam:.6f}  非缝 {base:.6f}  比值 {ratio:.3f}   ({len(idx)} 道缝)")
        png, v, xy = sharpest_crop(dst, n, idx)
        print(f"  裁图（纹理最多处 y={xy[1]}）: {png}  锐度 {v:.4f}")


if __name__ == "__main__":
    sys.exit(main())
