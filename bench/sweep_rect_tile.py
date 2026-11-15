# -*- coding: utf-8 -*-
"""矩形 tile 对比：128×128 vs 128×192（overlap 12），比 tile 数 / 接缝 / 峰值。

640×360：128×128 → 3×6=18；128×192 → 3×4=12（−33%）
"""
import json
import math
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
OV = 12

CASES = [
    ("sq128", dict(tile=128)),
    ("rect128x192", dict(tile=128, tile_h=128, tile_w=192)),
    ("rect128x256", dict(tile=128, tile_h=128, tile_w=256)),
]


def tiles(h, w, th, tw, ov):
    return (math.ceil((h - ov) / (th - ov)) * math.ceil((w - ov) / (tw - ov)))


def params(extra):
    p = dict(scale=4, tile=128, kv=3.0, chunk=0, overlap=8, mode="tiny-long",
             attention="sparse", encoder="cpu", quality=15, quant="mxfp8",
             budget="auto", pre_scale=1.0, tile_overlap=OV,
             seg_dir=os.path.join(WORK, "segments"))
    p.update(extra)
    return p


def sample_peak(stop, out):
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


def run(tag, extra):
    dst = os.path.join(OUT, f"rect_{tag}.mp4")
    cfg = os.path.join(WORK, f"rect_{tag}.json")
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"src": SRC, "dst": dst, "params": params(extra)}, f, ensure_ascii=False)
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["FLASHVSR_DEC_BACKEND"] = "torch"
    stop, peaks = threading.Event(), []
    th = threading.Thread(target=sample_peak, args=(stop, peaks), daemon=True)
    th.start()
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-u", WORKER, cfg], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    stop.set()
    th.join(timeout=2)
    ok = any(l.startswith("@@DONE ") for l in (p.stdout or "").splitlines())
    err = next((l for l in (p.stdout or "").splitlines() if l.startswith("@@ERR ")), "")
    th_, tw_ = extra.get("tile_h", 128), extra.get("tile_w", 128)
    print(f"[{tag:13s}] tile {th_}x{tw_} → {tiles(360, 640, th_, tw_, OV):3d} 个  "
          f"峰值 {peaks[0] if peaks else 0:5d} MiB  {'OK' if ok else 'FAIL ' + err[:110]}"
          f"  [耗时 {wall:.0f}s 参考]", flush=True)
    return dst if ok else None


def main():
    outs = {}
    for tag, extra in CASES:
        o = run(tag, extra)
        if o:
            outs[tag] = o
        time.sleep(5)
    if "sq128" in outs:
        print("\n=== 与 128×128 的差异 ===")
        for tag, path in outs.items():
            if tag == "sq128":
                continue
            p = subprocess.run([sys.executable, os.path.join(ROOT, "compare_videos.py"),
                                outs["sq128"], path], cwd=ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            tail = [l for l in (p.stdout or "").splitlines() if l.strip()][-2:]
            print(f"[{tag}] " + " | ".join(tail), flush=True)


if __name__ == "__main__":
    main()
