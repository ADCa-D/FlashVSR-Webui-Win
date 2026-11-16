# -*- coding: utf-8 -*-
"""两个质疑的实测回答：
  1. x265 到底有没有被限制到 4 线程？（看 ffmpeg -benchmark 的 cpu/wall 与 x265 自己的 pool 日志）
  2. SVT-AV1 的 lp=4 / 6 / 8 各是什么代价与画质？

用法：python bench/bench_threads.py [帧数]
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

SRC = os.path.join(ROOT, "out", "4765_333d880c286c_2x.mp4")
OUT = os.path.join(ROOT, "work", "threads")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50

CASES = [
    # tag, 编码参数, 是否打印 x265 线程信息
    ("x265 t4", ["-c:v", "libx265", "-preset", "medium", "-crf", "20", "-threads", "4"]),
    ("x265 default", ["-c:v", "libx265", "-preset", "medium", "-crf", "20"]),
    ("x265 pools4", ["-c:v", "libx265", "-preset", "medium", "-crf", "20",
                     "-x265-params", "pools=4"]),
    ("av1 p8 lp4", ["-c:v", "libsvtav1", "-preset", "8", "-crf", "26",
                    "-svtav1-params", "lp=4"]),
    ("av1 p8 lp6", ["-c:v", "libsvtav1", "-preset", "8", "-crf", "26",
                    "-svtav1-params", "lp=6"]),
    ("av1 p8 lp8", ["-c:v", "libsvtav1", "-preset", "8", "-crf", "26",
                    "-svtav1-params", "lp=8"]),
    ("av1 p6 lp6", ["-c:v", "libsvtav1", "-preset", "6", "-crf", "25",
                    "-svtav1-params", "lp=6"]),
]


def bench(src, args, dst):
    cmd = [config.FFMPEG, "-hide_banner", "-benchmark", "-v", "info", "-y",
           "-i", src, "-frames:v", str(N), "-an"] + args + ["-pix_fmt", "yuv420p", dst]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    err = r.stderr or ""
    m = re.search(r"bench: utime=([\d.]+)s stime=([\d.]+)s", err)
    cpu = (float(m.group(1)) + float(m.group(2))) if m else -1
    pool = [l.strip() for l in err.splitlines()
            if "Thread pool" in l or "frame threads" in l or "pools" in l.lower()
            or "Threads:" in l]
    return r.returncode, wall, cpu, err, pool


def vmaf(enc):
    r = subprocess.run([config.FFMPEG, "-v", "info", "-i", enc, "-i", SRC,
                        "-frames:v", str(N),
                        "-lavfi", "[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];"
                                  "[a][b]libvmaf", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.findall(r"VMAF score:\s*([\d.]+)", r.stderr or "")
    return float(m[-1]) if m else None


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"源 {os.path.basename(SRC)}  {N} 帧 4K\n")
    print(f"{'配置':14s} {'wall':>6s} {'cpu':>7s} {'并发':>5s} {'MB':>6s} {'VMAF':>6s}")
    for tag, args in CASES:
        dst = os.path.join(OUT, tag.replace(" ", "_") + ".mp4")
        code, wall, cpu, err, pool = bench(SRC, args, dst)
        if code != 0:
            print(f"{tag:14s} 失败 {(err.strip().splitlines() or [''])[-1][:80]}")
            continue
        mb = os.path.getsize(dst) / 1048576
        s = vmaf(dst)
        conc = (cpu / wall) if cpu > 0 and wall > 0 else 0
        print(f"{tag:14s} {wall:5.1f}s {cpu:6.1f}s {conc:4.1f}x {mb:6.2f} "
              f"{f'{s:.2f}' if s else 'n/a':>6s}")
        for l in pool[:3]:
            print(f"    · {l[:110]}")


if __name__ == "__main__":
    main()
