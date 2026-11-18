# -*- coding: utf-8 -*-
"""C：modulate 融合（addcmul）的 A/B —— 2 轮交替，各 8 步。

同时做单元级校验：同一输入下 addcmul 版与旧的 x*(1+scale)+shift 的 maxdiff。
"""
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = r"D:\ComfyUI_windows_portable\python_embeded\python.exe"


def run_case(fused, n=8):
    env = dict(os.environ)
    env.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "BENCH_QUANT": "mxfp8",
                "BENCH_G": "16", "BENCH_TOPK": "22.5", "BENCH_KV": "2.0",
                "FLASHVSR_FUSED_MODULATE": "1" if fused else "0"})
    out = subprocess.run([PY, os.path.join(HERE, "bench_dit_step.py"), str(n)],
                         cwd=ROOT, env=env, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    m = re.search(r"blocks=\d+\s+([\d.]+)\s*ms", out.stdout)
    if not m:
        raise RuntimeError((out.stdout or "")[-400:] + (out.stderr or "")[-400:])
    return float(m.group(1))


def unit():
    import torch
    sys.path.insert(0, r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes")
    import importlib
    import types
    if "fvnode" not in sys.modules:
        pkg = types.ModuleType("fvnode")
        pkg.__path__ = [r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast"]
        sys.modules["fvnode"] = pkg
    W = importlib.import_module("fvnode.src.models.wan_video_dit")
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(1, 1024, 1536, generator=g, device="cuda").to(torch.bfloat16)
    shift = torch.randn(1, 1, 1536, generator=g, device="cuda").to(torch.bfloat16)
    scale = torch.randn(1, 1, 1536, generator=g, device="cuda").to(torch.bfloat16)
    with torch.no_grad():
        W._FUSED_MODULATE = True
        a = W.modulate(x, shift, scale)
        W._FUSED_MODULATE = False
        b = W.modulate(x, shift, scale)
    d = (a.float() - b.float()).abs()
    print("单元级 modulate：equal=%s maxdiff %.6f 相对 %.2e"
          % (torch.equal(a, b), d.max().item(),
             (d.norm() / b.float().norm()).item()))


def main():
    unit()
    on, off = [], []
    for r in range(2):
        a = run_case(True)
        b = run_case(False)
        on.append(a)
        off.append(b)
        print("round%d  融合 %.1f ms   旧写法 %.1f ms" % (r + 1, a, b), flush=True)
    mo, mf = statistics.median(on), statistics.median(off)
    print("\n融合 %s → 中位 %.1f ms" % (["%.1f" % v for v in on], mo))
    print("旧版 %s → 中位 %.1f ms" % (["%.1f" % v for v in off], mf))
    print("变化 %+.1f%%" % (100 * (mo / mf - 1)))


if __name__ == "__main__":
    main()
