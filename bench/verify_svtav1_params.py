# -*- coding: utf-8 -*-
"""确认 svtav1 固定参数（preset 10 + lp=4）在 4K 上能跑，且不受环境变量影响。"""
import os
import subprocess
import sys
import time

sys.path.insert(0, r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web")
import engine as E  # noqa: E402

TMP = os.path.join(os.environ.get("TEMP", "."), "av1_bench")
SRC = os.path.join(TMP, "src4k.mp4")


def main():
    print("encoder_args:", " ".join(E.encoder_args("svtav1", 30)))
    if not os.path.isfile(SRC):
        print("没有 4K 测试源，跳过")
        return
    out = os.path.join(TMP, "svt_fixed.mp4")
    t0 = time.perf_counter()
    r = subprocess.run([E.FFMPEG, "-hide_banner", "-v", "error", "-i", SRC]
                       + E.encoder_args("svtav1", 30) + ["-y", out],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    if r.returncode == 0 and os.path.isfile(out):
        print(f"4K 编码成功 {dt:.1f}s  {os.path.getsize(out)/2**20:.2f} MB")
    else:
        print(f"失败 exit={r.returncode}: {(r.stderr or '').strip()[-200:]}")


if __name__ == "__main__":
    main()
