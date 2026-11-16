# -*- coding: utf-8 -*-
"""实测本机 SVT-AV1 到底接受哪些 tune（数值与名字），并把它自己打印的 tune 行抓出来。"""
import re
import subprocess
import sys

FF = r"D:\ffmpeg-9.0.1\ffmpeg.exe"


def probe(param):
    cmd = [FF, "-hide_banner", "-v", "info", "-f", "lavfi",
           "-i", "color=c=gray:s=320x240:r=5:d=0.4", "-frames:v", "2",
           "-c:v", "libsvtav1", "-crf", "32", "-svtav1-params", f"lp=4:{param}",
           "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    err = r.stderr or ""
    tune_lines = [l.strip() for l in err.splitlines()
                  if re.search(r"tune|Tune", l) and "overrides" not in l]
    err_lines = [l.strip() for l in err.splitlines() if "error" in l.lower() or "invalid" in l.lower()]
    return r.returncode, tune_lines[:3], err_lines[:2]


print("=== 数值 ===")
for n in range(0, 7):
    code, tl, el = probe(f"tune={n}")
    print(f"tune={n} exit={code}  {(' | '.join(tl))[:150]}")
    if code != 0 and el:
        print(f"          err: {el[0][:120]}")

print("\n=== 名字 ===")
for name in ("vq", "ssim", "psnr", "animation", "vmaf", "iq"):
    code, tl, el = probe(f"tune={name}")
    print(f"tune={name:10s} exit={code}  {(' | '.join(tl))[:150]}")
    if code != 0 and el:
        print(f"          err: {el[0][:120]}")
