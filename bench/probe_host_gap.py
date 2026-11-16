# -*- coding: utf-8 -*-
"""为什么 kernel 只占 3.94s、墙钟却 8.09s？逐项拆掉可疑项：

  A 基线（现状：每 8 次 clean_vram 就 gc+empty_cache；TCDecoder channels_last）
  B 去掉 empty_cache（只保留 gc）
  C 去掉 channels_last（纯 NCHW）
  D B+C 都去掉

用法：python bench/probe_host_gap.py [帧数]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import engine as E  # noqa: E402

SRC = os.path.join(r"D:\ComfyUI_windows_portable\ComfyUI\input", "flashvsr_long_544.mp4")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8

eng = E.Engine()
print("加载模型…", flush=True)
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
frames = E.decode_frames_range(SRC, 0, N, pre_scale=0.5, fps=25.0)
print(f"输入 {frames.shape}", flush=True)

cur_clean = eng.mod.clean_vram
dec = eng.pipe.TCDecoder
cur_decode = dec.decode_video
orig_decode = getattr(cur_decode, "__wrapped__", None)


def run(tag):
    eng._run_window(frames[:4], 2, 128, 3, tile_overlap=32)     # 预热
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = eng._run_window(frames, 2, 128, 3, tile_overlap=32)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"{tag:26s} {dt:5.2f}s", flush=True)
    return dt


def clean_nogc():
    torch.cuda.empty_cache()


def clean_noop():
    pass


import gc  # noqa: E402


def clean_gconly():
    gc.collect()


res = {}
res["A 基线"] = run("A 基线（gc+empty_cache）")

eng.mod.clean_vram = clean_noop
res["B 无 clean_vram"] = run("B 无 clean_vram")
res["C 无 clean_vram+无cl"] = run("C 无 clean_vram")
eng.mod.clean_vram = clean_gconly
res["B2 只 gc"] = run("B2 只 gc（无 empty_cache）")

eng.mod.clean_vram = cur_clean
print("\n对比：")
base = res["A 基线"]
for k, v in res.items():
    print(f"  {k:24s} {v:5.2f}s  {(base-v)/base*100:+5.1f}%")
