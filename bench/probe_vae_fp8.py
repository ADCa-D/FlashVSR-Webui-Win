# -*- coding: utf-8 -*-
"""TCDecoder 换 fp8 卷积：解码耗时 + 与 bf16 输出的 PSNR。

用法：python bench/probe_vae_fp8.py [帧数]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import engine as E  # noqa: E402
import fp8_conv  # noqa: E402

SRC = os.path.join(r"D:\ComfyUI_windows_portable\ComfyUI\input", "flashvsr_long_544.mp4")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
stats = {"t": 0.0, "n": 0}


def psnr(a, b):
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    return 99.0 if mse <= 0 else 10 * np.log10(255.0 ** 2 / mse)


print("fp8 卷积可用:", fp8_conv.available(), flush=True)
eng = E.Engine()
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
dec = eng.pipe.TCDecoder
frames = E.decode_frames_range(SRC, 0, N, pre_scale=0.5, fps=25.0)

orig_decode = dec.decode_video


def timed(*a, **kw):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = orig_decode(*a, **kw)
    torch.cuda.synchronize()
    stats["t"] += time.perf_counter() - t0
    stats["n"] += 1
    return out


def run(tag):
    stats["t"], stats["n"] = 0.0, 0
    eng._run_window(frames[:4], 2, 128, 3, tile_overlap=32)      # 预热
    stats["t"], stats["n"] = 0.0, 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = eng._run_window(frames, 2, 128, 3, tile_overlap=32)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    print(f"{tag:12s} 整体 {total:5.2f}s  TCDecoder {stats['t']:5.2f}s "
          f"({stats['t']/total*100:4.1f}%)  {stats['n']} 次", flush=True)
    return out, total, stats["t"]


dec.decode_video = timed
base, tot0, dt0 = run("bf16")

n_patch = fp8_conv.patch_module(dec, verbose=True)
if n_patch == 0:
    print("没有可替换的卷积（或平台不支持 fp8）")
    sys.exit(0)

out, tot1, dt1 = run("fp8")
print(f"\n解码加速 {(dt0-dt1)/dt0*100:+.1f}%（{dt0:.2f} → {dt1:.2f}s）  "
      f"整体 {(tot0-tot1)/tot0*100:+.1f}%")
print(f"输出 PSNR {psnr(out, base):.2f} dB  最大差 "
      f"{int(np.abs(out.astype(np.int16)-base.astype(np.int16)).max())}")
