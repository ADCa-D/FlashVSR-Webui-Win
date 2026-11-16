# -*- coding: utf-8 -*-
"""收尾验证：面板参数 reuse=window 是否真的生效（不再依赖环境变量）。"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

SRC = os.path.join(ROOT, "work", "anime_on_twos.mp4")
OUT = os.path.join(ROOT, "work", "final_check")
FFPROBE = r"D:\ffmpeg-9.0.1\ffprobe.exe"


def count_frames(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
                         capture_output=True, text=True, errors="replace").stdout
    return int("".join(c for c in out if c.isdigit()) or 0)


def main():
    for env_key in ("FLASHVSR_TILE_REUSE", "FLASHVSR_WIN_REUSE", "FLASHVSR_FRAME_DROP",
                    "FLASHVSR_REUSE"):
        os.environ.pop(env_key, None)

    import engine as E

    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
    os.makedirs(OUT, exist_ok=True)

    for mode in ("off", "window"):
        dst = os.path.join(OUT, f"reuse_{mode}.mp4")
        params = dict(preset="quality", quant="mxfp8", tile=128, kv=3, chunk=0, overlap=0,
                      scale=2, pre_scale=1.0, max_frames=33, use_float16=False,
                      restart=True, reuse=mode)
        stages = []
        t0 = time.perf_counter()
        eng.upscale_video(SRC, dst, params,
                          lambda *a: stages.append(" ".join(str(x) for x in a if x is not None)))
        dt = time.perf_counter() - t0
        n = count_frames(dst)
        hint = [s for s in stages if "复用" in s]
        print(f"reuse={mode:6s} {dt:6.2f}s  帧数 {n}  {' / '.join(hint) if hint else ''}")
        assert n == 33, f"帧数不对: {n}"


if __name__ == "__main__":
    sys.exit(main())
