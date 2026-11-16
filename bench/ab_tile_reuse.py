# -*- coding: utf-8 -*-
"""A/B：静止 tile 复用开/关，比时间与画质（PSNR against 关闭时的输出）。

用法：
  python bench/ab_tile_reuse.py            # 跑 A（关）和 B（开），再算 PSNR
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FFMPEG = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_long_544.mp4"
OUT = os.path.join(ROOT, "work", "reuse")
FRAMES = 33          # 8n+5 的倍数附近，够 3 个窗口
TILE = 128


def run_once(tag, reuse, thr=None):
    os.environ["FLASHVSR_TILE_REUSE"] = "1" if reuse else "0"
    if thr is not None:
        os.environ["FLASHVSR_REUSE_THR"] = str(thr)
    import importlib

    import engine as E
    importlib.reload(E)          # 让环境变量生效
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
    params = dict(preset="quality", quant="mxfp8", tile=TILE, kv=3, chunk=0,
                  overlap=0, scale=2, pre_scale=1.0, max_frames=FRAMES,
                  use_float16=False)
    dst = os.path.join(OUT, f"{tag}.mp4")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    eng.upscale_video(SRC, dst, params, lambda *a: None)
    dt = time.perf_counter() - t0
    st = getattr(eng, "_reuse_stats", {}) or {}
    print(f"{tag}: {dt:6.2f}s   decode 调用 {st.get('n',0)} 次，跳过 {st.get('skip',0)} 次"
          f"（{100.0*st.get('skip',0)/max(1,st.get('n',1)):.1f}%）")
    return dst, dt


def psnr(a, b):
    cmd = [FFMPEG, "-v", "error", "-i", a, "-i", b,
           "-lavfi", "psnr", "-f", "null", "-"]
    out = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    for line in (out.stderr or "").splitlines():
        if "average:" in line:
            return line.strip()
    return "PSNR 计算失败"


def main():
    a, ta = run_once("off", False)
    b, tb = run_once("on", True, thr=0.002)
    print(f"\n时间: 关 {ta:.2f}s → 开 {tb:.2f}s  ({(tb-ta)/ta*100:+.1f}%)")
    print(psnr(a, b))


if __name__ == "__main__":
    sys.exit(main())
