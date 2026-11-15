# -*- coding: utf-8 -*-
"""短探针：每个配置跑 ~80 秒就杀，只看显存峰值与是否 OOM（不跑完整片）。

峰值出现在「加载 + 解码整段 LQ + 前几个 tile」，跑满没必要。
用法：python bench/peak_probe.py
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
SECONDS = 80

CASES = [
    ("t192_chunk28", dict(tile=192, kv=3.0, chunk=28)),
    ("t192_chunk20", dict(tile=192, kv=3.0, chunk=20)),
    ("t192_chunk16", dict(tile=192, kv=3.0, chunk=16)),
]


def params(tile, kv, chunk):
    return dict(scale=4, tile=tile, kv=kv, chunk=chunk, overlap=8, mode="tiny-long",
                attention="sparse", encoder="cpu", quality=15, quant="mxfp8",
                budget="auto", pre_scale=1.0, tile_overlap=12,
                seg_dir=os.path.join(WORK, "segments"))


def probe(tag, kw):
    dst = os.path.join(OUT, f"pk_{tag}.mp4")
    cfg = os.path.join(WORK, f"pk_{tag}.json")
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"src": SRC, "dst": dst, "params": params(**kw)}, f, ensure_ascii=False)
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["FLASHVSR_DEC_BACKEND"] = "torch"
    p = subprocess.Popen([sys.executable, "-u", WORKER, cfg], cwd=ROOT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    peak, verdict = 0, None
    t0 = time.time()
    while time.time() - t0 < SECONDS:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            peak = max(peak, int(r.stdout.strip().splitlines()[0]))
        except Exception:
            pass
        line = p.stdout.readline() if p.poll() is None else ""
        if line.startswith("@@DONE "):
            verdict = "DONE"
            break
        if line.startswith("@@ERR "):
            verdict = "OOM/ERR: " + json.loads(line[6:]).get("error", "")[:80]
            break
        if p.poll() is not None:
            verdict = "exit=" + str(p.returncode)
            break
        time.sleep(0.3)
    if p.poll() is None:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        verdict = f"跑满 {SECONDS}s 未崩（峰值已采到）"
    print(f"[{tag}] 峰值 {peak:5d} MiB（余 {8151 - peak:4d}）  {verdict}", flush=True)
    time.sleep(3)


def main():
    for tag, kw in CASES:
        probe(tag, kw)


if __name__ == "__main__":
    main()
