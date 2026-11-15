# -*- coding: utf-8 -*-
"""混合精度 vs 纯 MXFP8 的画质对比（办公模式下只测质量，不看速度）。"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_real.mp4"
WORKER = os.path.join(ROOT, "worker.py")
WORK = os.path.join(ROOT, "work")
OUT = os.path.join(ROOT, "out")


def run(tag, quant):
    dst = os.path.join(OUT, f"qa_{tag}.mp4")
    cfg = os.path.join(WORK, f"qa_{tag}.json")
    params = dict(scale=4, tile=128, kv=3.0, chunk=0, overlap=8, mode="tiny-long",
                  attention="sparse", encoder="cpu", quality=15, quant=quant,
                  budget="auto", pre_scale=1.0, tile_overlap=12,
                  seg_dir=os.path.join(WORK, "segments"))
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"src": SRC, "dst": dst, "params": params}, f, ensure_ascii=False)
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["FLASHVSR_DEC_BACKEND"] = "torch"
    p = subprocess.run([sys.executable, "-u", WORKER, cfg], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = any(l.startswith("@@DONE ") for l in (p.stdout or "").splitlines())
    mix = [l for l in (p.stdout or "").splitlines() if "混合精度" in l]
    print(f"[{tag}] {'OK' if ok else 'FAIL'}", flush=True)
    for l in mix:
        print("   ", l[:120], flush=True)
    return dst if ok else None


def main():
    a = run("mxfp8", "mxfp8")
    b = run("mixed", "mixed")
    if a and b:
        p = subprocess.run([sys.executable, os.path.join(ROOT, "compare_videos.py"), a, b],
                           cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        tail = [l for l in (p.stdout or "").splitlines() if l.strip()][-2:]
        print("=== 混合 vs MXFP8 ===")
        print(" | ".join(tail))


if __name__ == "__main__":
    main()
