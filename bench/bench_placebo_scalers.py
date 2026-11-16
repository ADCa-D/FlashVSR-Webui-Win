# -*- coding: utf-8 -*-
"""libplacebo 缩放器对比：耗时 + 抗锯齿（mozzie）+ 细节保留。

场景就是本项目的"输入预缩放"：1080p → 540p（/2），随后交给 4× 模型。
用法：python bench/bench_placebo_scalers.py [帧数]
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import config  # noqa: E402

REAL = os.path.join(ROOT, "work", "placebo", "src_1080_50f.mp4")
OUT = os.path.join(ROOT, "work", "placebo")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50

CASES = [
    ("spline36（现在）", "downscaler=spline36", None),
    ("lanczos", "downscaler=lanczos", None),
    ("ewa_lanczos", "downscaler=ewa_lanczos", None),
    ("ewa_lanczos4sharpest", "downscaler=ewa_lanczos4sharpest", None),
    ("ewa_lanczos4sharpest + HQ", "downscaler=ewa_lanczos4sharpest", "preset=high_quality"),
    ("spline36 + HQ", "downscaler=spline36", "preset=high_quality"),
]


def run(src, filt, dst, extra=None):
    vf = "libplacebo=w=540:h=304:" + filt
    if extra:
        vf += ":extra_opts=" + extra
    vf += ",hwdownload,format=yuv420p"
    cmd = [config.FFMPEG, "-hide_banner", "-benchmark", "-v", "info", "-y",
           "-init_hw_device", "vulkan=pl", "-filter_hw_device", "pl",
           "-i", src, "-frames:v", str(N), "-vf", vf, "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "6",
           "-pix_fmt", "yuv420p", dst]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    m = re.search(r"bench: utime=([\d.]+)s stime=([\d.]+)s", r.stderr or "")
    cpu = (float(m.group(1)) + float(m.group(2))) if m else -1
    return r.returncode, wall, cpu, (r.stderr or "")


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"1080p → 540p，{N} 帧，核显 Vulkan 缩放 + 软编 x264（只用来落盘做比较）")
    print(f"{'配置':26s} {'wall':>7s} {'cpu':>7s} {'并发':>5s}  输出")
    for tag, filt, extra in CASES:
        dst = os.path.join(OUT, tag.split("（")[0].replace(" ", "_").replace("+", "hq") + ".mp4")
        code, wall, cpu, err = run(REAL, filt, dst, extra)
        if code != 0:
            lines = [l for l in (err or "").splitlines() if l.strip()]
            last = next((l for l in reversed(lines) if "Error" in l or "error" in l),
                        lines[-1] if lines else "")
            print(f"{tag:26s} FAIL {last[:90]}")
            continue
        print(f"{tag:26s} {wall:6.1f}s {cpu:6.1f}s {cpu/wall:4.1f}x  {os.path.basename(dst)}")
    print("\n产物都在", OUT)


if __name__ == "__main__":
    main()
