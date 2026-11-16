# -*- coding: utf-8 -*-
"""显存峰值实测：tile128 vs tile192（mxfp8 / nvfp4），同时给 torch 统计与设备级轮询。

用法：python bench/probe_peak_tile.py <quant> <tile> [源视频] [帧数]
"""
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import engine as E  # noqa: E402

quant = sys.argv[1] if len(sys.argv) > 1 else "mxfp8"
tile = int(sys.argv[2]) if len(sys.argv) > 2 else 128
SRC = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    r"D:\ComfyUI_windows_portable\ComfyUI\input", "flashvsr_long_544.mp4")
N = int(sys.argv[4]) if len(sys.argv) > 4 else 8

stop = threading.Event()
peak = {"mib": 0}


def poll():
    while not stop.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, timeout=5).strip()
            val = int(out.splitlines()[0])
            peak["mib"] = max(peak["mib"], val)
        except Exception:
            pass
        time.sleep(0.05)


eng = E.Engine()
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant=quant)
pre = float(os.environ.get("FLASHVSR_PROBE_PRE", "0.5"))
frames = E.decode_frames_range(SRC, 0, N, pre_scale=pre, fps=25.0)
ov = max(24, tile // 4)

torch.cuda.synchronize()
base = torch.cuda.memory_reserved() / 2 ** 20
eng._run_window(frames[:4], 2, tile, 3, tile_overlap=ov)      # 预热

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
t = threading.Thread(target=poll, daemon=True)
t.start()
t0 = time.perf_counter()
eng._run_window(frames, 2, tile, 3, tile_overlap=ov)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
stop.set()
t.join(timeout=2)

peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 20
peak_res = torch.cuda.max_memory_reserved() / 2 ** 20
total = torch.cuda.mem_get_info()[1] / 2 ** 20
print(f"{quant:6s} tile={tile:3d} ov={ov:2d}  {dt:5.2f}s  "
      f"torch峰值 alloc={peak_alloc:6.0f} reserved={peak_res:6.0f} MiB  "
      f"设备峰值={peak['mib']:6.0f}/{total:.0f} MiB  "
      f"（加载后基线 reserved {base:.0f}）", flush=True)
