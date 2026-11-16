# -*- coding: utf-8 -*-
"""VAE 唯一还没试过的一项：只对 TCDecoder.decode_video 上 torch.compile。

背景（都已默认开启，别再重复"发现"）：
  - channels_last（engine.py:1342）→ 实测 625→540 ms，−13.6%，逐位一致
  - cudnn.benchmark=True（engine.py:1307）
  - fp16 解码 → 只 −3%，不值得
VAE 占比已降不下来（24% 就含上面两项）。compile 是最后一项。

用法：python bench/probe_compile_decoder.py [帧数]
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

SRC = os.path.join(r"D:\ComfyUI_windows_portable\ComfyUI\input", "flashvsr_long_544.mp4")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
stats = {"t": 0.0, "n": 0}


def psnr(a, b):
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    return 99.0 if mse <= 0 else 10 * np.log10(255.0 ** 2 / mse)


eng = E.Engine()
print("加载模型…", flush=True)
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
dec = eng.pipe.TCDecoder
frames = E.decode_frames_range(SRC, 0, N, pre_scale=0.5, fps=25.0)
cur = dec.decode_video                     # 已带 channels_last 的包装
print(f"输入 {frames.shape}", flush=True)


def timed(fn, x, cond):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(x, cond)
    torch.cuda.synchronize()
    stats["t"] += time.perf_counter() - t0
    stats["n"] += 1
    return out


def run(tag, patch):
    stats["t"], stats["n"] = 0.0, 0
    dec.decode_video = patch
    try:
        eng._run_window(frames[:4], 2, 128, 3, tile_overlap=32)     # 预热
        stats["t"], stats["n"] = 0.0, 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = eng._run_window(frames, 2, 128, 3, tile_overlap=32)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
    finally:
        dec.decode_video = cur
    print(f"{tag:10s} 整体 {total:5.2f}s  TCDecoder {stats['t']:5.2f}s "
          f"({stats['t']/total*100:4.1f}%)  {stats['n']} 次", flush=True)
    return out, total, stats["t"]


base, tot0, dt0 = run("基线", cur)

# ---------- 只编解码器
tok_holder = {}


def plain(x, cond):
    return cur(x, parallel=False, show_progress_bar=False, cond=cond)


def compiled(x, cond):
    if "f" not in tok_holder:
        t0 = time.perf_counter()
        tok_holder["f"] = torch.compile(plain, mode="reduce-overhead", fullgraph=False)
        tok_holder["compile_s"] = time.perf_counter() - t0
    return tok_holder["f"](x, cond)


def patched(x, parallel=True, show_progress_bar=False, cond=None):
    return timed(compiled, x, cond)


print("首次调用会触发编译（可能要几分钟）…", flush=True)
t0 = time.perf_counter()
out, tot1, dt1 = run("compile", patched)
print(f"编译+首跑 {time.perf_counter()-t0:.1f}s，之后稳定 {dt1:.2f}s，"
      f"PSNR vs 基线 {psnr(out, base):.2f} dB", flush=True)
