# -*- coding: utf-8 -*-
"""TRT 解码器「带状态逐帧」验证：数值对齐 + 计时（对应真实 parallel=False 路径）。

真实路径里 TGrow 会把 1 帧裂成 4 个子帧，所以「并行图」的加速比不能直接引用，
必须和同一张逐帧图对比。

用法：
    python bench/probe_trt_step.py 48
    python bench/probe_trt_step.py 96      # 4× tile192 的真实 latent 尺寸
"""
import os
import site
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

import engine as E
import trt_decoder as T

H = int(sys.argv[1]) if len(sys.argv) > 1 else 48
NF = 3


def main():
    cache = os.path.join(os.environ.get("TEMP", "."), "flashvsr_trt_cache")
    os.makedirs(cache, exist_ok=True)
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", quant="none")
    dec = eng.pipe.TCDecoder.to("cuda")
    try:
        eng.pipe.dit.to("cpu")     # 只测解码：5.4 GB DiT 挪走
    except Exception:
        pass
    torch.cuda.empty_cache()
    dec.clean_mem()

    torch.manual_seed(0)
    latent = torch.randn(1, NF, 16, H, H, device="cuda", dtype=torch.bfloat16) * 0.5
    cond = torch.randn(1, 3, NF * 4, H * 8, H * 8, device="cuda", dtype=torch.bfloat16).clamp(-1, 1)

    # ---- torch 参考
    dec.clean_mem()
    with torch.no_grad():
        ref = dec.decode_video(latent, parallel=False, cond=cond).float()
    print(f"torch 输出 {tuple(ref.shape)}（裁掉前 {dec.frames_to_trim} 帧）", flush=True)

    # ---- 启 TRT
    trt = T.maybe_enable(dec, cache, log=lambda s: print(s, flush=True))
    if trt is None:
        print("TRT 不可用")
        return

    dec.clean_mem()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = dec.decode_video(latent, parallel=False, cond=cond).float()
    print(f"TRT 首跑 {time.perf_counter()-t0:.1f}s（含引擎构建）输出 {tuple(out.shape)}", flush=True)

    n = min(out.shape[1], ref.shape[1])
    md = float((out[0, :n] - ref[0, :n]).abs().max())
    print(f"maxdiff = {md:.4f}（bf16 权重 + fp16 引擎，<0.05 视为等价）", flush=True)

    # ---- 计时
    def t_torch():
        dec.clean_mem()
        trt._orig(latent, parallel=False, cond=cond)

    def t_trt():
        dec.clean_mem()
        dec.decode_video(latent, parallel=False, cond=cond)

    for _ in range(2):
        t_torch()
    torch.cuda.synchronize()
    a = time.perf_counter()
    for _ in range(3):
        t_torch()
    torch.cuda.synchronize()
    ta = (time.perf_counter() - a) / 3 / NF * 1000

    for _ in range(2):
        t_trt()
    torch.cuda.synchronize()
    b = time.perf_counter()
    for _ in range(3):
        t_trt()
    torch.cuda.synchronize()
    tb = (time.perf_counter() - b) / 3 / NF * 1000

    print(f"latent {H}: torch {ta:.1f} ms/帧 · TRT {tb:.1f} ms/帧 → {ta / tb:.2f}×", flush=True)


if __name__ == "__main__":
    main()
