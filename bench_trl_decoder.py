# -*- coding: utf-8 -*-
"""FlashVSR TCDecoder：torch vs TensorRT 实测。

复用 bench/probe_tcdec_chunk.py 的 build/make_inputs/run_full（已验证可跑的调用路径），
然后用 TRTDecoder.wrap 替换 decode_video 走 TRT。
"""
import os
import sys
import time

ROOT = r"D:\flashvsr-web"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))

import torch  # noqa: E402
import probe_tcdec_chunk as P  # noqa: E402
from trt_decoder import TRTDecoder  # noqa: E402

torch.cuda.reset_peak_memory_stats()
tile = int(sys.argv[1]) if len(sys.argv) > 1 else 128
frames = int(sys.argv[2]) if len(sys.argv) > 2 else 6
lat = tile // 8

dec = P.build()
xt, cond = P.make_inputs(lat, frames)
print(f"tile={tile} latent={lat} T={frames} x{tuple(xt.shape)} cond{tuple(cond.shape)}", flush=True)

# 1) torch 基准
P.run_full(dec, xt, cond, False)                       # 预热
dt, ref, pk = P.run_full(dec, xt, cond, False)
print(f"torch bf16           {dt*1000:7.0f} ms   peak {pk:6.0f} MiB", flush=True)

# 2) TRT
cache = os.path.join(ROOT, "work", "trt_cache")
trt = TRTDecoder(dec, cache_dir=cache, log=lambda m: print("  " + str(m), flush=True))
t0 = time.perf_counter()
trt.prewarm(lat, lat)
print(f"prewarm(build)       {time.perf_counter()-t0:7.1f} s", flush=True)

trt.wrap(dec)                                          # 替换 dec.decode_video
out = dec.decode_video(xt, parallel=False, cond=cond)  # 走 TRT
if isinstance(out, (tuple, list)):
    out = out[0]
d = float((out.float() - ref.float()).abs().max())
print(f"TRT vs torch maxdiff {d:.3e}", flush=True)

# 3) TRT 计时
if torch.cuda.is_available():
    torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(3):
    dec.decode_video(xt, parallel=False, cond=cond)
if torch.cuda.is_available():
    torch.cuda.synchronize()
dt2 = (time.perf_counter() - t0) / 3
print(f"TRT                  {dt2*1000:7.0f} ms   加速 {dt/dt2:.2f}x", flush=True)
