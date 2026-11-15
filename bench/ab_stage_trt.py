# -*- coding: utf-8 -*-
"""试分段 TRT 解码（StageTRTDecoder）：是否能建 4 段引擎 + 与 torch 共存 + 速度。"""
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


def params():
    return dict(scale=4, tile=128, kv=3.0, chunk=0, overlap=8, mode="tiny-long",
                attention="sparse", encoder="cpu", quality=15, quant="mxfp8",
                budget="auto", pre_scale=1.0, tile_overlap=12,
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


def run(tag, backend, extra_env=None):
    dst = os.path.join(OUT, f"stage_{tag}.mp4")
    cfg = os.path.join(WORK, f"stage_{tag}.json")
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"src": SRC, "dst": dst, "params": params()}, f, ensure_ascii=False)
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["FLASHVSR_DEC_BACKEND"] = backend
    if extra_env:
        env.update(extra_env)
    stop, peaks = threading.Event(), []
    th = threading.Thread(target=sample_peak, args=(stop, peaks), daemon=True)
    th.start()
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-u", WORKER, cfg], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    stop.set()
    th.join(timeout=2)
    out = p.stdout or ""
    info = next((l for l in out.splitlines() if l.startswith("@@DONE ")), None)
    err = next((l for l in out.splitlines() if l.startswith("@@ERR ")), None)
    print(f"\n[{tag}] wall {wall:.1f}s  峰值 {peaks[0] if peaks else 0} MiB  "
          f"{'OK' if info else 'FAIL'}", flush=True)
    if err:
        print("   ERR:", err[:200], flush=True)
    for l in out.splitlines():
        if "[TRT]" in l or "channels_last" in l:
            print("   ", l[:170], flush=True)
    return dst, info is not None


def main():
    a, ok_a = run("torch", "torch")
    b, ok_b = run("stage", "stage")
    if ok_a and ok_b:
        p = subprocess.run([sys.executable, os.path.join(ROOT, "compare_videos.py"), a, b],
                           cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        tail = [l for l in (p.stdout or "").splitlines() if l.strip()][-2:]
        print("\n=== stage vs torch ===")
        print(" | ".join(tail))


if __name__ == "__main__":
    main()
