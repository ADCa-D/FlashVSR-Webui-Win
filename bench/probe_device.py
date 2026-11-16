# -*- coding: utf-8 -*-
"""诊断：pipe(...) / tensor2video 返回的张量在哪个设备。"""
import os
import sys

sys.path.insert(0, r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web")
import torch  # noqa: E402
import engine as E  # noqa: E402

SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_long_544.mp4"
eng = E.Engine()
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
mod, pipe = eng.mod, eng.pipe
print("pipe.device =", pipe.device, " dtype =", pipe.torch_dtype)

import numpy as np
frames = torch.from_numpy(E.decode_frames_range(SRC, 0, 8, pre_scale=0.5, fps=25.0)).float().div_(255.0)
print("frames.device =", frames.device)

tile = (128, 128)
coords = mod.calculate_tile_coords(frames.shape[1], frames.shape[2], tile, 32)
print("tiles =", len(coords))
x1, y1, x2, y2 = coords[0]
input_tile = frames[:, y1:y2, x1:x2, :]
LQ, th, tw, F = mod.prepare_input_tensor(input_tile, pipe.device, scale=2, dtype=pipe.torch_dtype)
print("LQ.device =", LQ.device, LQ.shape, " th/tw/F =", th, tw, F)
out = pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
           tiled=True, LQ_video=LQ, num_frames=F, height=th, width=tw,
           is_full_block=False, if_buffer=True,
           topk_ratio=2.0 * 768 * 1280 / (th * tw), kv_ratio=3.0, local_range=11,
           color_fix=True, unload_dit=False, force_offload=False)
print("pipe 返回:", type(out), getattr(out, "device", None), getattr(out, "shape", None))
v = mod.tensor2video(out)
print("tensor2video:", v.device, v.shape, v.dtype, float(v.min()), float(v.max()))
