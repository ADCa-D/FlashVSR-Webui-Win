# -*- coding: utf-8 -*-
"""端到端 A/B：TRT 解码 vs torch 解码（同一素材/参数）。

跑法：python bench/ab_trt.py
产出 out/ab_trt_<tag>.mp4，并打印 wall time（含模型加载）与纯推理耗时。
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_real.mp4"
PY = sys.executable
WORKER = os.path.join(ROOT, "worker.py")
WORK = os.path.join(ROOT, "work")
OUT = os.path.join(ROOT, "out")

PARAMS = dict(scale=4, tile=192, kv=3.0, chunk=0, overlap=8, mode="tiny-long",
              attention="sparse", encoder="cpu", quality=15, quant="mxfp8",
              budget="auto", pre_scale=1.0, seg_dir=os.path.join(WORK, "segments"))


def run(tag, trt):
    dst = os.path.join(OUT, f"ab_trt_{tag}.mp4")
    cfg = os.path.join(WORK, f"ab_{tag}.json")
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"src": SRC, "dst": dst, "params": PARAMS}, f, ensure_ascii=False)
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["FLASHVSR_TRT"] = "1" if trt else "0"
    t0 = time.perf_counter()
    p = subprocess.run([PY, "-u", WORKER, cfg], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    info = None
    for line in (p.stdout or "").splitlines():
        if line.startswith("@@DONE "):
            info = json.loads(line[7:])
        elif line.startswith("@@ERR "):
            print(tag, "ERR", line[:200])
    print(f"[{tag}] wall {wall:.1f}s  info={info}", flush=True)
    if p.returncode != 0:
        print((p.stdout or "")[-600:], (p.stderr or "")[-600:], flush=True)
    for line in (p.stdout or "").splitlines():
        if "[TRT]" in line or "channels_last" in line:
            print("   ", line[:160], flush=True)
    return wall, dst


def main():
    os.makedirs(OUT, exist_ok=True)
    # 第一次：可能包含引擎构建，只用于预热
    run("warm", True)
    w_trt, dst_trt = run("trt", True)
    w_torch, dst_torch = run("torch", False)
    print(f"\n结论：torch {w_torch:.1f}s → TRT {w_trt:.1f}s  = {w_torch / w_trt:.2f}×", flush=True)
    print(f"对比两段：python compare_videos.py \"{dst_torch}\" \"{dst_trt}\"", flush=True)


if __name__ == "__main__":
    main()
