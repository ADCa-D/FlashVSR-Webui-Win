# -*- coding: utf-8 -*-
"""引擎已缓存的情况下，只看 stage 解码的实际耗时（不含构建）。"""
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


def run(tag, backend):
    dst = os.path.join(OUT, f"sp_{tag}.mp4")
    cfg = os.path.join(WORK, f"sp_{tag}.json")
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"src": SRC, "dst": dst, "params": params()}, f, ensure_ascii=False)
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["FLASHVSR_DEC_BACKEND"] = backend
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-u", WORKER, cfg], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    ok = any(l.startswith("@@DONE ") for l in (p.stdout or "").splitlines())
    build = [l for l in (p.stdout or "").splitlines() if "段引擎" in l or "就绪" in l]
    print(f"[{tag}] wall {wall:.1f}s  {'OK' if ok else 'FAIL'}  引擎行数={len(build)}", flush=True)
    for l in build:
        print("   ", l[:120], flush=True)


if __name__ == "__main__":
    run("stage2", "stage")
