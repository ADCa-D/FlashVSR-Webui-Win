# -*- coding: utf-8 -*-
"""编码器矩阵基准：同源、同时长，比 速度 / 体积 / VMAF。

用法：
    python bench/bench_av1_matrix.py [源视频]

没有任何源时自动用 lavfi 生成 4K 测试片（testsrc2，含高频细节）。
VMAF 以「源」为参考（无损源 → 编码结果），越高越好。
"""
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine as E  # noqa: E402

TMP = os.path.join(os.environ.get("TEMP", "."), "av1_bench")
CASES = [
    ("x264 slow crf18", ["-c:v", "libx264", "-preset", "slow", "-crf", "18"]),
    ("x264 medium crf18", ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]),
    ("x265 medium crf20", ["-c:v", "libx265", "-preset", "medium", "-crf", "20"]),
    ("svtav1 p10 lp4 crf30", ["-c:v", "libsvtav1", "-preset", "10", "-crf", "30",
                              "-svtav1-params", "lp=4"]),
    ("svtav1 p10 lp8 crf30", ["-c:v", "libsvtav1", "-preset", "10", "-crf", "30",
                              "-svtav1-params", "lp=8"]),
    ("svtav1 p8 lp4 crf30", ["-c:v", "libsvtav1", "-preset", "8", "-crf", "30",
                             "-svtav1-params", "lp=4"]),
    ("svtav1 p8 lp8 crf30", ["-c:v", "libsvtav1", "-preset", "8", "-crf", "30",
                             "-svtav1-params", "lp=8"]),
]


def make_src(path):
    subprocess.run([E.FFMPEG, "-hide_banner", "-v", "error", "-f", "lavfi",
                    "-i", "testsrc2=s=3840x2160:r=25:d=3", "-c:v", "libx264",
                    "-preset", "ultrafast", "-crf", "0", "-pix_fmt", "yuv420p", "-y", path],
                   check=True)
    return path


def vmaf(ref, test):
    p = subprocess.run([E.FFMPEG, "-hide_banner", "-v", "info", "-i", test, "-i", ref,
                        "-lavfi", "libvmaf", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"VMAF score:\s*([\d.]+)", p.stderr or "")
    return float(m.group(1)) if m else None


def main():
    os.makedirs(TMP, exist_ok=True)
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TMP, "src4k.mp4")
    if not os.path.isfile(src):
        print("生成 4K 测试源…", flush=True)
        make_src(src)
    info = E.probe(src)
    print(f"源 {os.path.basename(src)} {info['w']}x{info['h']} {info['frames']} 帧\n", flush=True)

    rows = []
    for tag, args in CASES:
        out = os.path.join(TMP, tag.replace(" ", "_") + ".mp4")
        t0 = time.perf_counter()
        r = subprocess.run([E.FFMPEG, "-hide_banner", "-v", "error", "-i", src]
                           + args + ["-pix_fmt", "yuv420p", "-y", out],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        dt = time.perf_counter() - t0
        if r.returncode != 0 or not os.path.isfile(out):
            print(f"{tag:22s} FAIL   {r.stderr.strip().splitlines()[-1][:80] if r.stderr else ''}")
            continue
        mb = os.path.getsize(out) / 2 ** 20
        v = vmaf(src, out) if "--no-vmaf" not in sys.argv else None
        rows.append((tag, dt, mb, v))
        print(f"{tag:22s} {dt:7.1f}s  {mb:6.2f} MB  VMAF {v if v is None else round(v, 2)}", flush=True)

    print("\n=== 汇总（按 VMAF/体积 排序）===")
    for tag, dt, mb, v in sorted(rows, key=lambda x: (-(x[3] or 0), x[2])):
        print(f"{tag:22s} {dt:7.1f}s  {mb:6.2f} MB  VMAF {v if v is None else round(v, 2)}")


if __name__ == "__main__":
    main()
