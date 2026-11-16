# -*- coding: utf-8 -*-
"""解码后端对比：CPU(4线程) vs D3D11VA vs CUDA(NVDEC)。

输出统一 rawvideo rgb24 丢掉（NUL），只比解码+下载+转格式的总耗时与 CPU 占用。
"""
import os
import subprocess
import sys
import time

FF = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
SRC = r"C:\Users\MECHREVO\AppData\Local\Temp\opencode\src4k.mp4"   # 4K h264, 50 帧
IF = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_long_544.mp4"

CASES = [
    ("cpu threads4", ["-threads", "4"]),
    ("cpu threads32", ["-threads", "32"]),
    ("d3d11va", ["-hwaccel", "d3d11va"]),
    ("cuda(nvdec)", ["-hwaccel", "cuda"]),
]


def run(src, args, tag):
    cmd = [FF, "-v", "error"] + args + ["-i", src, "-f", "rawvideo",
                                        "-pix_fmt", "rgb24", "-"]
    t0 = time.perf_counter()
    cp0 = time.process_time()
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    dt = time.perf_counter() - t0
    cpu = time.process_time() - cp0
    ok = p.returncode == 0
    err = (p.stderr or b"").decode("utf-8", "ignore").strip().splitlines()
    return ok, dt, cpu, (err[-1][:100] if err else "")


def main():
    for src, name in ((SRC, "4K/h264/50帧"), (IF, "544p/h264/100帧")):
        if not os.path.isfile(src):
            continue
        print(f"\n=== {name} ===")
        for tag, args in CASES:
            ok, dt, cpu, err = run(src, args, tag)
            print(f"{tag:15s} {'OK ' if ok else 'FAIL'} wall {dt:5.2f}s  CPU {cpu:5.2f}s"
                  f"  {'' if ok else err}")


if __name__ == "__main__":
    main()
