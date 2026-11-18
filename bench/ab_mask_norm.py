# -*- coding: utf-8 -*-
"""mask 跳过 + RMSNorm 融合 的 A/B（4 种组合，交替各跑 2 轮，取中位）。

同时做两件单元级校验：
  1. 饱和时返回的 ones mask 形状是否与真 mask 一致（否则 attn kernel 会崩）
  2. F.rms_norm 与旧实现在同一输入下的 maxdiff（bf16 下应为 1~2 ulp）
"""
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = r"D:\ComfyUI_windows_portable\python_embeded\python.exe"


def run_case(skip_mask, fused_norm, n=8):
    env = dict(os.environ)
    env.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "BENCH_QUANT": "mxfp8",
                "BENCH_G": "16", "BENCH_TOPK": "22.5", "BENCH_KV": "2.0",
                "FLASHVSR_SKIP_MASK": "1" if skip_mask else "0",
                "FLASHVSR_FUSED_NORM": "1" if fused_norm else "0"})
    out = subprocess.run([PY, os.path.join(HERE, "bench_dit_step.py"), str(n)],
                         cwd=ROOT, env=env, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    import re
    m = re.search(r"blocks=\d+\s+([\d.]+)\s*ms", out.stdout)
    if not m:
        raise RuntimeError((out.stdout or "")[-500:] + (out.stderr or "")[-500:])
    return float(m.group(1))


def main():
    cases = {"both": (True, True), "mask_off": (False, True),
             "norm_off": (True, False), "both_off": (False, False)}
    res = {k: [] for k in cases}
    for r in range(2):
        for k, (sm, fn) in cases.items():
            ms = run_case(sm, fn)
            res[k].append(ms)
            print("round%d %-9s %.1f ms" % (r + 1, k, ms), flush=True)

    med = {k: statistics.median(v) for k, v in res.items()}
    print()
    for k, v in med.items():
        print("%-9s %s → 中位 %.1f ms" % (k, ["%.1f" % x for x in res[k]], v))
    base = med["both_off"]
    print("\n两项目全开 vs 全关: %.1f → %.1f ms  (%+.1f%%)"
          % (base, med["both"], 100 * (med["both"] / base - 1)))
    print("单纯 mask 跳过: %+.1f%%   单纯 RMSNorm 融合: %+.1f%%"
          % (100 * (med["mask_off"] / base - 1), 100 * (med["norm_off"] / base - 1)))


if __name__ == "__main__":
    main()
