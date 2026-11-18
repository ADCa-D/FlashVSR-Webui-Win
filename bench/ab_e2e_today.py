# -*- coding: utf-8 -*-
"""B：端到端验证今天的三个开关是否真的兑现（全开 vs 全关，2 轮交替，取中位）。

每个 case 用独立子进程（开关是 import 时读取的），固定同一段片子与参数。
"""
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = r"D:\ComfyUI_windows_portable\python_embeded\python.exe"
SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_long_544.mp4"
MAXF = 100
ROUNDS = 2

# 单个 case 的 worker：直接调 engine
WORKER = r'''
import json, os, sys, time
sys.path.insert(0, r"%(root)s")
import torch
import engine as E
eng = E.Engine()
eng.ensure_loaded("tiny-long", "sparse", lambda *a, **k: None, quant="mxfp8")
params = dict(scale=2, tile=128, kv=2.0, chunk=0, overlap=1, mode="tiny-long",
              attention="sparse", quant="mxfp8", pre_scale=0.5, max_frames=%(maxf)d,
              encoder="cpu", quality=15, bitdepth=8, tile_overlap=12,
              color_fix=True, seam_fix="smooth", restart=True)
# 预热（编码器探测等一次性开销）
eng.upscale_video(r"%(src)s", r"%(root)s\work\e2e_warm.mp4", params, lambda *a, **k: None)
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
t0 = time.time()
info = eng.upscale_video(r"%(src)s", os.environ["OUT"], params, lambda *a, **k: None)
torch.cuda.synchronize()
print("RESULT " + json.dumps({"wall": time.time() - t0,
                              "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                              "frames": info.get("frames")}))
'''


def run_case(flags, out):
    env = dict(os.environ)
    env.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "OUT": out})
    env.update(flags)
    script = WORKER % {"root": ROOT.replace("\\", "\\\\"), "src": SRC.replace("\\", "\\\\"),
                       "maxf": MAXF}
    path = os.path.join(ROOT, "work", "_e2e_worker.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(script)
    r = subprocess.run([PY, path], cwd=ROOT, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    m = re.search(r"RESULT (\{.*\})", r.stdout or "")
    if not m:
        raise RuntimeError((r.stdout or "")[-600:] + (r.stderr or "")[-600:])
    return json.loads(m.group(1))


def main():
    on = {"FLASHVSR_FUSE_QKV": "1", "FLASHVSR_SKIP_MASK": "1", "FLASHVSR_FUSED_NORM": "1"}
    off = {"FLASHVSR_FUSE_QKV": "0", "FLASHVSR_SKIP_MASK": "0", "FLASHVSR_FUSED_NORM": "0"}
    a, b = [], []
    for r in range(ROUNDS):
        x = run_case(on, os.path.join(ROOT, "work", "e2e_on.mp4"))
        a.append(x)
        y = run_case(off, os.path.join(ROOT, "work", "e2e_off.mp4"))
        b.append(y)
        print("round%d  on %.1fs (%.2f GiB)   off %.1fs (%.2f GiB)"
              % (r + 1, x["wall"], x["peak_gib"], y["wall"], y["peak_gib"]), flush=True)
    wa = statistics.median([x["wall"] for x in a])
    wb = statistics.median([x["wall"] for x in b])
    print("\n全开 %s → 中位 %.1f s" % (["%.1f" % x["wall"] for x in a], wa))
    print("全关 %s → 中位 %.1f s" % (["%.1f" % x["wall"] for x in b], wb))
    print("端到端变化 %+.1f%%（%.2fx）；帧数 %s" %
          (100 * (wa / wb - 1), wb / wa, [x["frames"] for x in a + b]))


if __name__ == "__main__":
    main()
