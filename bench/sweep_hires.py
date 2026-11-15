# -*- coding: utf-8 -*-
"""高分辨率复现：960×544 / 100 帧，比较 tile 与「分段与否」的显存。

假设：OOM 的主因是整段 LQ 缓冲（帧数 × 输出分辨率），不是 tile。
"""
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_long_544.mp4"
WORKER = os.path.join(ROOT, "worker.py")
WORK = os.path.join(ROOT, "work")
OUT = os.path.join(ROOT, "out")

CASES = [
    ("544_t192_kv3_chunk0", dict(tile=192, kv=3.0, chunk=0)),
    ("544_t128_kv3_chunk0", dict(tile=128, kv=3.0, chunk=0)),
    ("544_t128_kv3_chunk40", dict(tile=128, kv=3.0, chunk=40)),
]


def params(tile, kv, chunk):
    return dict(scale=4, tile=tile, kv=kv, chunk=chunk, overlap=8, mode="tiny-long",
                attention="sparse", encoder="cpu", quality=15, quant="mxfp8",
                budget="auto", pre_scale=1.0, seg_dir=os.path.join(WORK, "segments"))


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


def run(tag, kw):
    dst = os.path.join(OUT, f"hi_{tag}.mp4")
    cfg = os.path.join(WORK, f"hi_{tag}.json")
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"src": SRC, "dst": dst, "params": params(**kw)}, f, ensure_ascii=False)
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
    info = err = None
    for line in (p.stdout or "").splitlines():
        if line.startswith("@@DONE "):
            info = json.loads(line[7:])
        elif line.startswith("@@ERR "):
            err = json.loads(line[6:]).get("error", "")[:200]
    peak = peaks[0] if peaks else 0
    print(f"[{tag}] 峰值 {peak:5d} MiB（余 {8151-peak:4d}）  "
          f"{'OK' if info else 'OOM: ' + str(err)}  [耗时 {wall:.0f}s 参考]", flush=True)
    return info


def main():
    for tag, kw in CASES:
        run(tag, kw)
        time.sleep(5)


if __name__ == "__main__":
    main()
