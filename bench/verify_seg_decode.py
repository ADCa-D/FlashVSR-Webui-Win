# -*- coding: utf-8 -*-
"""验证：分段时只解当前窗口的帧（内存不再随总帧数增长）+ 输出帧数正确。"""
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


def main():
    params = dict(scale=4, tile=128, kv=3.0, chunk=30, overlap=8, mode="tiny-long",
                  attention="sparse", encoder="cpu", quality=15, quant="mxfp8",
                  budget="auto", pre_scale=1.0, tile_overlap=12,
                  seg_dir=os.path.join(WORK, "segments"))
    dst = os.path.join(OUT, "seg30.mp4")
    cfg = os.path.join(WORK, "seg30.json")
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
    out = p.stdout or ""
    info = next((l for l in out.splitlines() if l.startswith("@@DONE ")), None)
    err = next((l for l in out.splitlines() if l.startswith("@@ERR ")), None)
    dec = [l for l in out.splitlines() if "[decode" in l or "解码" in l][:4]
    print(f"wall {wall:.0f}s  峰值 {peaks[0] if peaks else 0} MiB  {'OK' if info else 'FAIL'}")
    if info:
        print(info[:200])
    if err:
        print(err[:200])
    for l in dec:
        print("  ", l[:120])
    if os.path.isfile(dst):
        r = subprocess.run([r"D:\ffmpeg-9.0.1\ffprobe.exe", "-v", "error", "-select_streams", "v:0",
                            "-count_frames", "-show_entries", "stream=nb_read_frames,width,height",
                            "-of", "csv=p=0", dst], capture_output=True, text=True)
        print("输出:", r.stdout.strip())


if __name__ == "__main__":
    main()
