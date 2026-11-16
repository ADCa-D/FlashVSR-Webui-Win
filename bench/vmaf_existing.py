# -*- coding: utf-8 -*-
"""给已有编码产物补算 VMAF。

坑（都踩过了）：
  1. `-v error` 会把 VMAF 分数（info 级）一起吞掉 → 必须 `-v info`；
  2. `libvmaf=log_path=C:\\...` 里的冒号会被滤镜解析器当选项分隔符，路径又含中文
     → `Invalid argument`。直接用 stderr 里的 "VMAF score:" 更省事。
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import config  # noqa: E402

SRC = os.path.join(ROOT, "out", "4765_333d880c286c_2x.mp4")
OUT = os.path.join(ROOT, "work", "enc_tiers")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50


def vmaf(enc):
    r = subprocess.run(
        [config.FFMPEG, "-v", "info", "-i", enc, "-i", SRC, "-frames:v", str(N),
         "-lavfi", "[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]libvmaf",
         "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.findall(r"VMAF score:\s*([\d.]+)", r.stderr or "")
    return (float(m[-1]) if m else None)


rows = []
for fn in sorted(os.listdir(OUT)):
    if not fn.endswith(".mp4"):
        continue
    p = os.path.join(OUT, fn)
    tag = fn[:-4]
    size = os.path.getsize(p) / 1048576
    s = vmaf(p)
    rows.append((tag, size, s))
    print(f"{tag:16s} {size:6.2f} MB  VMAF {f'{s:.2f}' if s else 'n/a'}")

print("\n按 VMAF 排序：")
for tag, size, s in sorted(rows, key=lambda r: -(r[2] or 0)):
    print(f"  {tag:16s} VMAF {s if s else 0:6.2f}  {size:5.2f} MB")
