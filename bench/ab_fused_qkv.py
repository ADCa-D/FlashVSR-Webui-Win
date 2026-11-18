# -*- coding: utf-8 -*-
"""Fused QKV 的干净 A/B：6 个独立进程交替跑（避免同进程两份权重 OOM），取中位数。

每次进程都命中同一份量化缓存，therefore 启动状态一致。
"""
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = r"D:\ComfyUI_windows_portable\python_embeded\python.exe"
N_ROUNDS = 3


def one(fuse: bool) -> float:
    env = dict(os.environ)
    env.update({
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
        "BENCH_QUANT": "mxfp8",
        "BENCH_G": "16",
        "BENCH_TOPK": "22.5",
        "BENCH_KV": "2.0",
        "FLASHVSR_FUSE_QKV": "1" if fuse else "0",
    })
    out = subprocess.run([PY, os.path.join(HERE, "bench_dit_step.py"), "10"],
                         cwd=ROOT, env=env, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    m = re.search(r"blocks=\d+\s+([\d.]+)\s*ms", out.stdout)
    if not m:
        tail = (out.stdout or "")[-400:] + (out.stderr or "")[-400:]
        raise RuntimeError("没解析到耗时:\n" + tail)
    return float(m.group(1))


def main():
    on, off = [], []
    for i in range(N_ROUNDS):
        a = one(True)
        on.append(a)
        b = one(False)
        off.append(b)
        print("round%d  fused=%.1f ms   unfused=%.1f ms" % (i + 1, a, b), flush=True)
    mo, mf = statistics.median(on), statistics.median(off)
    print("\nfused   %s  → 中位 %.1f ms" % (["%.1f" % v for v in on], mo))
    print("unfused %s  → 中位 %.1f ms" % (["%.1f" % v for v in off], mf))
    print("变化 %+.1f%%（%.2fx）" % (100 * (mo / mf - 1), mf / mo))


if __name__ == "__main__":
    main()
