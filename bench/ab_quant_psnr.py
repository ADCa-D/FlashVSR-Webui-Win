# -*- coding: utf-8 -*-
"""三种量化 vs bf16 真值的绝对 PSNR/SSIM（同一素材、同参数）。

办公模式下只测画质，不看速度。
"""
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
    if os.path.isfile(dst):
        print(f"[{tag}] 已存在，跳过", flush=True)
        return dst
    p = subprocess.run([sys.executable, "-u", WORKER, cfg], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = any(l.startswith("@@DONE ") for l in (p.stdout or "").splitlines())
    err = next((l for l in (p.stdout or "").splitlines() if l.startswith("@@ERR ")), "")
    print(f"[{tag}] {'OK' if ok else 'FAIL ' + err[:120]}", flush=True)
    return dst if ok else None


def psnr(a, b):
    p = subprocess.run([sys.executable, os.path.join(ROOT, "compare_videos.py"), a, b],
                       cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return " | ".join([l for l in (p.stdout or "").splitlines() if l.strip()][-2:])


def main():
    ref = run("bf16", "none")
    if not ref:
        return
    for tag, q in (("mxfp8", "mxfp8"), ("mixed", "mixed")):
        f = os.path.join(OUT, f"qa_{tag}.mp4")
        if os.path.isfile(f):
            print(f"\n{tag} vs bf16: {psnr(ref, f)}", flush=True)


if __name__ == "__main__":
    main()
