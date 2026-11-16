# -*- coding: utf-8 -*-
"""W8A16 实测：与 bf16 / MXFP8 比画质（PSNR/SSIM）与耗时。

显存常驻用 bench/quant_mem.py 单独测；这里只看"画质掉了多少、快了多少"。

用法：python bench/bench_w8.py [帧数]
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORKER = os.path.join(ROOT, "worker.py")
WORK = os.path.join(ROOT, "work")
OUT = os.path.join(ROOT, "out")
SRC = os.path.join(r"D:\ComfyUI_windows_portable\ComfyUI\input", "flashvsr_long_544.mp4")


def run(tag, quant, nframes):
    dst = os.path.join(OUT, f"w8_{tag}.mp4")
    cfg = os.path.join(WORK, f"w8_{tag}.json")
    params = dict(scale=2, tile=128, kv=3.0, chunk=0, overlap=8, mode="tiny-long",
                  attention="sparse", encoder="x265", quality=12, bitdepth=8,
                  quant=quant, budget="fixed", pre_scale=1.0, tile_overlap=32,
                  max_frames=nframes, color_fix=True, seam_fix="smooth",
                  hi_precision=False, restart=True,
                  seg_dir=os.path.join(WORK, "segments"))
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"src": SRC, "dst": dst, "params": params}, f, ensure_ascii=False)
    if os.path.isfile(dst):
        os.remove(dst)
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-u", WORKER, cfg], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    ok = any(l.startswith("@@DONE ") for l in (p.stdout or "").splitlines())
    err = next((l for l in (p.stdout or "").splitlines() if l.startswith("@@ERR ")), "")
    print(f"[{tag:6s}] {'OK ' if ok else 'FAIL'} {dt:6.1f}s  {('' if ok else err[:120])}", flush=True)
    return (dst if ok else None), dt


def psnr(a, b):
    p = subprocess.run([sys.executable, os.path.join(ROOT, "compare_videos.py"), a, b],
                       cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    lines = [l.strip() for l in (p.stdout or "").splitlines() if l.strip()]
    return lines[-1] if lines else "?"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(f"素材 {os.path.basename(SRC)}  {n} 帧  2×  tile128/ov32")
    ref, t_ref = run("bf16", "none", n)
    if not ref:
        return 1
    rows = [("bf16", t_ref, "—")]
    for tag, q in (("mxfp8", "mxfp8"), ("w8", "w8")):
        f, dt = run(tag, q, n)
        if f:
            rows.append((tag, dt, psnr(ref, f)))

    print("\n=== 结果（画质相对 bf16）===")
    for tag, dt, ps in rows:
        print(f"{tag:7s} {dt:6.1f}s   {ps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
