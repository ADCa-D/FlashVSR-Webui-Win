# -*- coding: utf-8 -*-
"""A/B：节点 patch 前后（缓存常量张量）的端到端耗时。"""
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast\src\models\wan_video_dit.py"
SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_real.mp4"
WORKER = os.path.join(ROOT, "worker.py")
WORK = os.path.join(ROOT, "work")
OUT = os.path.join(ROOT, "out")


def run(tag):
    dst = os.path.join(OUT, f"ab_patch_{tag}.mp4")
    cfg = os.path.join(WORK, f"ab_patch_{tag}.json")
    params = dict(scale=4, tile=128, kv=3.0, chunk=0, overlap=8, mode="tiny-long",
                  attention="sparse", encoder="cpu", quality=15, quant="mxfp8",
                  budget="auto", pre_scale=1.0, tile_overlap=12,
                  seg_dir=os.path.join(WORK, "segments"))
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"src": SRC, "dst": dst, "params": params}, f, ensure_ascii=False)
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["FLASHVSR_DEC_BACKEND"] = "torch"
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-u", WORKER, cfg], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    ok = any(l.startswith("@@DONE ") for l in (p.stdout or "").splitlines())
    print(f"[{tag}] {wall:6.1f}s  {'OK' if ok else 'FAIL'}", flush=True)
    return wall


def main():
    shutil.copyfile(NODE, NODE + ".patched")
    shutil.copyfile(NODE + ".bak", NODE)
    print("== 原始版本 ==", flush=True)
    a = run("base")
    shutil.copyfile(NODE + ".patched", NODE)
    print("== patch 后 ==", flush=True)
    b = run("patched")
    print(f"\n端到端：{a:.1f}s → {b:.1f}s = {a / b:.2f}×", flush=True)


if __name__ == "__main__":
    main()
