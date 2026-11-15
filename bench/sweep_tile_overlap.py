# -*- coding: utf-8 -*-
"""DiT tile_overlap 扫描：羽化带宽度 vs tile 数 vs 接缝。

固定 tile=128 / kv3 / mxfp8 / 25 帧 640×360，比较 overlap = 24 / 16 / 12 / 8：
- tile 数（由节点 calculate_tile_coords 的公式直接算）
- 与 overlap=24 基线的 PSNR/SSIM（接缝差异会拉低它）
- 显存峰值

注意：低噪音模式下耗时不作性能数据，这里只看接缝与显存。
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

TILE = 128
OVERLAPS = [24, 16, 12, 8]


def tiles(h, w, tile, ov):
    stride = tile - ov
    return math.ceil((h - ov) / stride) * math.ceil((w - ov) / stride)


def params(ov):
    return dict(scale=4, tile=TILE, kv=3.0, chunk=0, overlap=8, mode="tiny-long",
                attention="sparse", encoder="cpu", quality=15, quant="mxfp8",
                budget="auto", pre_scale=1.0, tile_overlap=ov,
                seg_dir=os.path.join(WORK, "segments"))


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


def run(ov):
    dst = os.path.join(OUT, f"ov_{ov}.mp4")
    cfg = os.path.join(WORK, f"ov_{ov}.json")
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"src": SRC, "dst": dst, "params": params(ov)}, f, ensure_ascii=False)
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
    print(f"[overlap={ov:2d}] tile={tiles(360, 640, TILE, ov):3d}  峰值 {peaks[0] if peaks else 0:5d} MiB  "
          f"{'OK' if ok else 'FAIL ' + err[:100]}  [耗时 {wall:.0f}s 参考]", flush=True)
    return dst if ok else None


def main():
    outs = {}
    for ov in OVERLAPS:
        o = run(ov)
        if o:
            outs[ov] = o
        time.sleep(5)
    if 24 in outs:
        print("\n=== 与 overlap=24 的差异 ===")
        for ov, path in outs.items():
            if ov == 24:
                continue
            p = subprocess.run([sys.executable, os.path.join(ROOT, "compare_videos.py"),
                                outs[24], path], cwd=ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            tail = [l for l in (p.stdout or "").splitlines() if l.strip()][-2:]
            print(f"[overlap={ov:2d}] " + " | ".join(tail), flush=True)


if __name__ == "__main__":
    main()
