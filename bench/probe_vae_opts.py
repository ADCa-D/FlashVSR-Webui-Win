# -*- coding: utf-8 -*-
"""VAE(TCDecoder) 优化对比：fp16 / channels_last / 叠加。

修正上一版两个 bug：
  1. patch 里没用计时器 → TCDecoder 时间一直是 0；
  2. `dec.float()` 把原本的 bf16 权重改成了 fp32，导致下一轮 channels_last
     报 "Input type (BFloat16) and bias type (float)"；必须还原成 bf16。
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
pipe = eng.pipe
dec = pipe.TCDecoder
frames = E.decode_frames_range(SRC, 0, N, pre_scale=0.5, fps=25.0)
print(f"输入 {frames.shape}", flush=True)
orig = dec.decode_video


def wrapped(fn):
    def inner(*a, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        torch.cuda.synchronize()
        stats["t"] += time.perf_counter() - t0
        stats["n"] += 1
        return out
    return inner


def run(patch, pre=None, post=None):
    stats["t"], stats["n"] = 0.0, 0
    if pre:
        pre()
    dec.decode_video = patch or wrapped(orig)
    try:
        eng._run_window(frames[:4], 2, 128, 3, tile_overlap=32)      # 预热
        stats["t"], stats["n"] = 0.0, 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = eng._run_window(frames, 2, 128, 3, tile_overlap=32)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
    finally:
        dec.decode_video = orig
        if post:
            post()
    return out, total, stats["t"], stats["n"]


base, tot, dt, n = run(None)
print(f"\n基线      整体 {tot:5.2f}s  TCDecoder {dt:5.2f}s ({dt/tot*100:4.1f}%)  {n} 次")
ref = base

# ---------- fp16
def fp16_fn(*a, **kw):
    a = tuple(x.half() if torch.is_tensor(x) and x.is_floating_point() else x for x in a)
    kw = {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
          for k, v in kw.items()}
    return orig(*a, **kw).to(torch.bfloat16)


out, tot2, dt2, n2 = run(wrapped(fp16_fn), pre=lambda: dec.half())
print(f"fp16      整体 {tot2:5.2f}s  TCDecoder {dt2:5.2f}s ({dt2/tot2*100:4.1f}%)  "
      f"PSNR {psnr(out, ref):5.2f} dB")
dec.to(torch.bfloat16)          # 还原 bf16，否则下一轮 bias dtype 不匹配

# ---------- channels_last
def cl_fn(*a, **kw):
    a = tuple(x.contiguous(memory_format=torch.channels_last)
              if torch.is_tensor(x) and x.dim() == 4 else x for x in a)
    kw = {k: (v.contiguous(memory_format=torch.channels_last)
              if torch.is_tensor(v) and v.dim() == 4 else v) for k, v in kw.items()}
    return orig(*a, **kw).contiguous()


out, tot3, dt3, n3 = run(wrapped(cl_fn),
                         pre=lambda: dec.to(memory_format=torch.channels_last))
print(f"channels  整体 {tot3:5.2f}s  TCDecoder {dt3:5.2f}s ({dt3/tot3*100:4.1f}%)  "
      f"PSNR {psnr(out, ref):5.2f} dB")

# ---------- fp16 + channels_last
dec.to(memory_format=torch.contiguous_format)
out, tot4, dt4, n4 = run(wrapped(cl_fn),
                         pre=lambda: dec.half().to(memory_format=torch.channels_last))
print(f"fp16+chL  整体 {tot4:5.2f}s  TCDecoder {dt4:5.2f}s ({dt4/tot4*100:4.1f}%)  "
      f"PSNR {psnr(out, ref):5.2f} dB")

print("\n还原 bf16 / NCHW…")
dec.to(torch.bfloat16).to(memory_format=torch.contiguous_format)
