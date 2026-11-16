# -*- coding: utf-8 -*-
"""先回答"值不值得动 VAE"：TCDecoder(=VAE 解码) 在一次推理里占多少时间？

背景：MXFP8 量化的是 nn.Linear（matmul 走 fp8 张量核）。TCDecoder 全是 Conv，没有 Linear，
所以 MXFP8 没法直接套；要判断是否值得为它换别的优化手段，先量占比。

用法：python bench/probe_vae_share.py [帧数] [quant]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import config  # noqa: E402
import engine as E  # noqa: E402

SRC = os.path.join(r"D:\ComfyUI_windows_portable\ComfyUI\input", "flashvsr_long_544.mp4")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
QUANT = sys.argv[2] if len(sys.argv) > 2 else "mxfp8"

eng = E.Engine()
print(f"加载模型（quant={QUANT}）…", flush=True)
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant=QUANT)
pipe = eng.pipe

calls = {"n": 0, "t": 0.0}
orig = pipe.TCDecoder.decode_video


def timed(*a, **kw):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = orig(*a, **kw)
    torch.cuda.synchronize()
    calls["n"] += 1
    calls["t"] += time.perf_counter() - t0
    return out


pipe.TCDecoder.decode_video = timed

frames = E.decode_frames_range(SRC, 0, N, pre_scale=0.5, fps=25.0)
print(f"输入 {frames.shape}（源/2）", flush=True)

# 预热一次，避免 cudnn autotune/首次编译算进来
eng._run_window(frames[:4], 2, 128, 3, tile_overlap=32)
calls["n"], calls["t"] = 0, 0.0

torch.cuda.synchronize()
t0 = time.perf_counter()
out = eng._run_window(frames, 2, 128, 3, tile_overlap=32)
torch.cuda.synchronize()
total = time.perf_counter() - t0

print(f"\n输出 {out.shape}")
print(f"整体        {total:6.2f} s")
print(f"TCDecoder   {calls['t']:6.2f} s  ({calls['n']} 次调用, 占 {calls['t']/total*100:.1f}%)")
print(f"其余(DiT等) {total - calls['t']:6.2f} s  ({(total-calls['t'])/total*100:.1f}%)")
