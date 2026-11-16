# -*- coding: utf-8 -*-
"""NVFP4 到底快不快：同素材背靠背 A/B（量化 × tile）。

注意：办公模式下绝对时间偏慢，但背靠背相对比较仍有效。
"""
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_real.mp4"
WORKER = os.path.join(ROOT, "worker.py")
WORK = os.path.join(ROOT, "work")
OUT = os.path.join(ROOT, "out")

CASES = [
    ("mxfp8_t128", "mxfp8", 128),
    ("nvfp4_t128", "nvfp4", 128),
    ("mxfp8_t192", "mxfp8", 192),
    ("nvfp4_t192", "nvfp4", 192),
]


def sample(stop, out):
    peak = 0
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            peak = max(peak, int(r.stdout.strip().splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.3)
    out.append(peak)


def run(tag, quant, tile):
    dst = os.path.join(OUT, f"q4_{tag}.mp4")
    cfg = os.path.join(WORK, f"q4_{tag}.json")
    params = dict(scale=4, tile=tile, kv=3.0 if tile == 128 else 2.0, chunk=0, overlap=8,
                  mode="tiny-long", attention="sparse", encoder="cpu", quality=15,
                  quant=quant, budget="auto", pre_scale=1.0, tile_overlap=12,
                  seg_dir=os.path.join(WORK, "segments"))
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"src": SRC, "dst": dst, "params": params}, f, ensure_ascii=False)
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["FLASHVSR_DEC_BACKEND"] = "torch"
    stop, peaks = threading.Event(), []
    th = threading.Thread(target=sample, args=(stop, peaks), daemon=True)
    th.start()
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-u", WORKER, cfg], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    stop.set()
    th.join(timeout=2)
    ok = any(l.startswith("@@DONE ") for l in (p.stdout or "").splitlines())
    print(f"[{tag:12s}] {wall:6.1f}s  峰值 {peaks[0] if peaks else 0:5d} MiB  "
          f"{'OK' if ok else 'FAIL'}", flush=True)
    return wall if ok else None


def main():
    res = {}
    for tag, q, tile in CASES:
        res[tag] = run(tag, q, tile)
        time.sleep(5)
    print("\n=== 汇总 ===")
    for k, v in res.items():
        print(f"{k:12s} {'—' if v is None else f'{v:.1f}s'}")


if __name__ == "__main__":
    main()
