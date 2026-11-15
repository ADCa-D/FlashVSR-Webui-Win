# -*- coding: utf-8 -*-
"""2× 路径（source /2 + 4× 模型，latent 48）的 TRT vs torch 端到端对比。"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_real.mp4"
WORKER = os.path.join(ROOT, "worker.py")
WORK = os.path.join(ROOT, "work")
OUT = os.path.join(ROOT, "out")

PARAMS = dict(scale=4, tile=96, kv=3.0, chunk=0, overlap=8, mode="tiny-long",
              attention="sparse", encoder="cpu", quality=15, quant="mxfp8",
              budget="auto", pre_scale=0.5, seg_dir=os.path.join(WORK, "segments"))


def run(tag, trt):
    dst = os.path.join(OUT, f"ab2x_{tag}.mp4")
    cfg = os.path.join(WORK, f"ab2x_{tag}.json")
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"src": SRC, "dst": dst, "params": PARAMS}, f, ensure_ascii=False)
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["FLASHVSR_TRT"] = "1" if trt else "0"
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-u", WORKER, cfg], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    info = None
    for line in (p.stdout or "").splitlines():
        if line.startswith("@@DONE "):
            info = json.loads(line[7:])
        if "[TRT]" in line:
            print("   ", line[:150], flush=True)
    print(f"[{tag}] wall {wall:.1f}s info={info}", flush=True)
    return wall


def main():
    run("warm", True)
    a = run("trt", True)
    b = run("torch", False)
    print(f"\n2× 路径：torch {b:.1f}s → TRT {a:.1f}s = {b / a:.2f}×", flush=True)


if __name__ == "__main__":
    main()
