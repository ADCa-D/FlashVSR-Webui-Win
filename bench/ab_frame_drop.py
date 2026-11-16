# -*- coding: utf-8 -*-
"""单次跑：抽帧复用开/关（分进程跑，避免两个模型同时在显存里）。

  python bench/ab_frame_drop.py off      # FLASHVSR_FRAME_DROP=0
  python bench/ab_frame_drop.py on       # FLASHVSR_FRAME_DROP=1
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

SRC = os.path.join(ROOT, "work", "anime_on_twos.mp4")
OUT = os.path.join(ROOT, "work", "dropdemo")
FRAMES = int(os.environ.get("AB_FRAMES", "33"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", choices=["off", "on"])
    a = ap.parse_args()
    os.environ["FLASHVSR_FRAME_DROP"] = "1" if a.tag == "on" else "0"

    import engine as E

    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *x: None, quant="mxfp8")
    params = dict(preset="quality", quant="mxfp8", tile=128, kv=3, chunk=0,
                  overlap=0, scale=2, pre_scale=1.0, max_frames=FRAMES,
                  use_float16=False, restart=True)
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, f"{a.tag}.mp4")
    t0 = time.perf_counter()
    eng.upscale_video(SRC, dst, params, lambda *x: None)
    dt = time.perf_counter() - t0
    print(f"RESULT {a.tag}: {dt:.2f}s -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
