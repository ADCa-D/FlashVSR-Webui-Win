# -*- coding: utf-8 -*-
"""端到端 A/B：DiT 整步 CUDA Graph 开 vs 关。

同一段短片段、同一组参数，比墙钟 + 峰值显存 + 输出是否一致。
用法：python bench/ab_dit_graph_e2e.py [frames]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import engine as E  # noqa: E402

SRC = os.environ.get("FLASHVSR_AB_SRC", r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_long_544.mp4")
WORK = os.path.join(ROOT, "work")
MAXF = int(sys.argv[1]) if len(sys.argv) > 1 else 9


def run(eng, dst, graph_on):
    if eng.dit_graph is not None:
        eng.dit_graph.enabled = graph_on
        eng.dit_graph.graphs.clear()
        eng.dit_graph.seen.clear()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    params = dict(scale=2, tile=128, kv=2.0, chunk=0, overlap=1, mode="tiny-long",
                  attention="sparse", quant="none", pre_scale=0.5, max_frames=MAXF,
                  encoder="cpu", quality=15, bitdepth=8, tile_overlap=12,
                  color_fix=True, seam_fix="smooth", restart=True)
    info = eng.upscale_video(SRC, dst, params, lambda *a, **k: None)
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    return dt, peak, info


def psnr(a, b):
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    if mse <= 0:
        return 99.0
    return 10 * np.log10((255.0 ** 2) / mse)


def main():
    os.makedirs(WORK, exist_ok=True)
    out_g = os.path.join(WORK, "e2e_graph.mp4")
    out_e = os.path.join(WORK, "e2e_eager.mp4")

    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a, **k: None, quant="none")

    # 第一次跑负责一次性初始化（ffmpeg/编码器探测、cudnn autotune），不计入对比
    dt_w, _, _ = run(eng, out_e, False)
    print(f"[warmup] {dt_w:.1f}s", flush=True)

    dt_g, pk_g, info_g = run(eng, out_g, True)
    print(f"[graph] 墙钟 {dt_g:.1f}s  峰值 {pk_g:.2f} GiB  {json.dumps(info_g, ensure_ascii=False)[:120]}",
          flush=True)
    if eng.dit_graph is not None:
        print("  " + eng.dit_graph.describe(), flush=True)
        eng.dit_graph.total_s = 0.0
        eng.dit_graph.calls = 0
    dt_e, pk_e, info_e = run(eng, out_e, False)
    print(f"[eager] 墙钟 {dt_e:.1f}s  峰值 {pk_e:.2f} GiB", flush=True)
    print(f"加速 {dt_e / dt_g:.2f}x", flush=True)
    if eng.dit_graph is not None:
        print("  " + eng.dit_graph.describe(), flush=True)

    fa = E.decode_frames_range(out_g, 0, MAXF, pre_scale=1.0)
    fb = E.decode_frames_range(out_e, 0, MAXF, pre_scale=1.0)
    n = min(len(fa), len(fb))
    print(f"帧数 graph={len(fa)} eager={len(fb)}  逐帧 PSNR 最小 {min(psnr(fa[i], fb[i]) for i in range(n)):.2f} dB",
          flush=True)


if __name__ == "__main__":
    main()
