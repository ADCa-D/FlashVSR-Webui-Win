# -*- coding: utf-8 -*-
"""补测：① 复现旧默认档 t192/kv3 是否 OOM；② 分段跑，比较 kv3 vs kv4 的段边界时序。

只关心稳定性/时序，耗时仅参考（低噪音模式）。
"""
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_long.mp4"
WORKER = os.path.join(ROOT, "worker.py")
WORK = os.path.join(ROOT, "work")
OUT = os.path.join(ROOT, "out")

CASES = [
    ("t192_kv3", dict(tile=192, kv=3.0, chunk=0), False),
    ("seg_t128_kv3", dict(tile=128, kv=3.0, chunk=40), True),
    ("seg_t128_kv4", dict(tile=128, kv=4.0, chunk=40), True),
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


def run(tag, kw, segmented):
    dst = os.path.join(OUT, f"sweep2_{tag}.mp4")
    cfg = os.path.join(WORK, f"sweep2_{tag}.json")
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
            err = json.loads(line[6:]).get("error", "")[:170]
    peak = peaks[0] if peaks else 0
    print(f"[{tag}] 峰值 {peak:5d} MiB（余 {8151-peak:4d}）  "
          f"{'OK' if info else 'OOM/FAIL: ' + str(err)}  [耗时 {wall:.0f}s 参考]", flush=True)
    return dict(tag=tag, peak=peak, ok=bool(info), out=dst, err=err, seg=segmented)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for tag, kw, seg in CASES:
        rows.append(run(tag, kw, seg))
        time.sleep(5)
    print("\n=== 时序一致性 ===")
    for r in rows:
        if not r["ok"]:
            continue
        p = subprocess.run([sys.executable, os.path.join(ROOT, "temporal_check.py"), r["out"]],
                           cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        tail = [l for l in (p.stdout or "").splitlines() if l.strip()][-3:]
        print(f"[{r['tag']}] " + " | ".join(tail), flush=True)


if __name__ == "__main__":
    main()
