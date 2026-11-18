# -*- coding: utf-8 -*-
"""单配置跑一遍复用：tag / 模式(off|seq|pos) / 阈值，落盘视频 + 打印统计。

用法：
  python bench/reuse_run.py off  seq 0.002
  python bench/reuse_run.py pos  pos 0.002
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_long_544.mp4"
OUT = os.path.join(ROOT, "work", "reuse_key")
FRAMES = 33
TILE = 128


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "off"
    mode = sys.argv[2] if len(sys.argv) > 2 else "off"
    thr = sys.argv[3] if len(sys.argv) > 3 else "0.002"
    os.environ["FLASHVSR_TILE_REUSE"] = "0" if mode == "off" else "1"
    os.environ["FLASHVSR_REUSE_KEY"] = "seq" if mode == "off" else mode
    os.environ["FLASHVSR_REUSE_THR"] = thr
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    import engine as E

    eng = E.Engine()
    t0 = time.perf_counter()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
    load_s = time.perf_counter() - t0
    # reuse 走 params（engine.py:2311 读 params["reuse"]），不是环境变量
    params = dict(preset="quality", quant="mxfp8", tile=TILE, kv=3, chunk=0,
                  overlap=0, scale=2, pre_scale=1.0, max_frames=FRAMES,
                  use_float16=False, reuse=("off" if mode == "off" else "tile"))
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, f"{tag}.mp4")
    t0 = time.perf_counter()
    eng.upscale_video(SRC, dst, params, lambda *a: None)
    dt = time.perf_counter() - t0
    st = getattr(eng, "_reuse_stats", {}) or {}
    n, skip = st.get("n", 0), st.get("skip", 0)
    print(f"[{tag}] mode={mode} thr={thr}  load {load_s:.1f}s  upscale {dt:.2f}s  "
          f"decode {n} 次 跳过 {skip} 次（{100.0 * skip / max(1, n):.1f}%）  "
          f"position 缓存 {len(st.get('pos', {}) or {})}", flush=True)


if __name__ == "__main__":
    main()
