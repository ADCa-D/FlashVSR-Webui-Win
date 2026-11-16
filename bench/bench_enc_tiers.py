# -*- coding: utf-8 -*-
"""编码器「同等努力 / 同等质量」对比：4K 素材，测耗时、体积、VMAF。

上一版结论（x265 medium 3.9 s 比 svtav1 p4 8.1 s 快）是**错的对比**：
SVT 的 preset 4 是高档位（0=最慢最好，13=最快），x265 的 medium 只是中档。
必须按「同代努力档」或「同画质」比，否则结论没意义。

用法：python bench/bench_enc_tiers.py [源] [帧数]
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import config  # noqa: E402

OUT = os.path.join(ROOT, "work", "enc_tiers")


def run(cmd, timeout=3600):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def encode(src, args, dst, nframes):
    r = run([config.FFMPEG, "-v", "error", "-y", "-i", src, "-frames:v", str(nframes),
             "-an"] + args + ["-pix_fmt", "yuv420p", dst])
    if r.returncode != 0:
        raise RuntimeError((r.stderr or "")[-300:])
    return os.path.getsize(dst) / 1048576


def vmaf(src, enc, nframes):
    """用 libvmaf 比 源 vs 编码后（脚本固定分辨率，用 -frames 限制帧数）。"""
    r = run([config.FFMPEG, "-v", "error", "-i", enc, "-i", src, "-frames:v", str(nframes),
             "-lavfi", "[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];"
                       "[a][b]libvmaf=n_threads=8",
             "-f", "null", "-"], timeout=3600)
    m = re.search(r"VMAF score:\s*([\d.]+)", r.stderr or "")
    if not m:
        return None, (r.stderr or "")[-160:]
    return float(m.group(1)), ""


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "out", "4765_333d880c286c_2x.mp4")
    nframes = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    os.makedirs(OUT, exist_ok=True)

    cases = [
        ("x264 medium", ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-threads", "4"]),
        ("x264 slow", ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-threads", "4"]),
        ("x265 fast", ["-c:v", "libx265", "-preset", "fast", "-crf", "20", "-threads", "4"]),
        ("x265 medium", ["-c:v", "libx265", "-preset", "medium", "-crf", "20", "-threads", "4"]),
        ("x265 slow", ["-c:v", "libx265", "-preset", "slow", "-crf", "20", "-threads", "4"]),
        ("av1 p10", ["-c:v", "libsvtav1", "-preset", "10", "-crf", "28",
                     "-svtav1-params", "lp=4"]),
        ("av1 p8", ["-c:v", "libsvtav1", "-preset", "8", "-crf", "26",
                    "-svtav1-params", "lp=4"]),
        ("av1 p6", ["-c:v", "libsvtav1", "-preset", "6", "-crf", "25",
                    "-svtav1-params", "lp=4"]),
        ("av1 p4", ["-c:v", "libsvtav1", "-preset", "4", "-crf", "24",
                    "-svtav1-params", "lp=4"]),
    ]

    print(f"源 {os.path.basename(src)}  取 {nframes} 帧  4 线程/lp=4")
    print(f"{'编码器':14s} {'耗时':>7s} {'体积MB':>7s} {'VMAF':>7s} {'kbps':>7s}")
    rows = []
    for tag, args in cases:
        dst = os.path.join(OUT, tag.replace(" ", "_") + ".mp4")
        import time
        t0 = time.perf_counter()
        try:
            mb = encode(src, args, dst, nframes)
        except RuntimeError as e:
            print(f"{tag:14s} 失败 {str(e)[:60]}")
            continue
        dt = time.perf_counter() - t0
        score, err = vmaf(src, dst, nframes)
        info = subprocess.run([config.FFPROBE, "-v", "error", "-select_streams", "v:0",
                               "-show_entries", "stream=bit_rate", "-of", "csv=p=0", dst],
                              capture_output=True, text=True)
        try:
            kbps = int(info.stdout.strip() or 0) // 1000
        except ValueError:
            kbps = 0
        rows.append((tag, dt, mb, score, kbps))
        print(f"{tag:14s} {dt:6.1f}s {mb:7.2f} "
              f"{(f'{score:.2f}' if score else 'n/a'):>7s} {kbps:7d}"
              + ("" if score else f"  {err[:60]}"))

    print("\n按 VMAF 从高到低（同画质比较看耗时/体积）：")
    for tag, dt, mb, score, kbps in sorted(rows, key=lambda r: -(r[3] or 0)):
        print(f"  {tag:14s} VMAF {score if score else 0:6.2f}  {dt:6.1f}s  {mb:6.2f} MB  {kbps} kbps")


if __name__ == "__main__":
    main()
