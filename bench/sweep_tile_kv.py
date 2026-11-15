# -*- coding: utf-8 -*-
"""tile × kv 扫描：**只测稳定性**（是否 OOM + 显存峰值 + 时序一致性）。

注意：低噪音/balance 电源模式下 GPU 频率被压，**这里的耗时不可用作性能数据**，
只用来判断「这套参数在这张卡上装不装得下」。性能数据要在性能模式另测。

每个配置跑一遍 worker（100 帧 / 4× / MXFP8，不分段 = 最坏情况），
300 ms 采样显存峰值，最后跑 temporal_check.py 看时序一致性。

用法：python bench/sweep_tile_kv.py
"""
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_long.mp4"
WORKER = os.path.join(ROOT, "worker.py")
WORK = os.path.join(ROOT, "work")
OUT = os.path.join(ROOT, "out")

# 从最激进到最保守：找到「装得下」的分界线
CONFIGS = [
    ("t192_kv2", dict(tile=192, kv=2.0)),   # 之前默认档，已知会 OOM → 复现
    ("t160_kv3", dict(tile=160, kv=3.0)),
    ("t128_kv4", dict(tile=128, kv=4.0)),   # 时序更稳，可能 OOM
    ("t128_kv3", dict(tile=128, kv=3.0)),   # 用户实测可用
    ("t96_kv4", dict(tile=96, kv=4.0)),     # 进一步降显存
]


def base(tile, kv):
    return dict(scale=4, tile=tile, kv=kv, chunk=0, overlap=8, mode="tiny-long",
                attention="sparse", encoder="cpu", quality=15, quant="mxfp8",
                budget="auto", pre_scale=1.0, seg_dir=os.path.join(WORK, "segments"))


def sample_peak(stop, out):
    peak = 0
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            v = int(r.stdout.strip().splitlines()[0])
            peak = max(peak, v)
        except Exception:
            pass
        time.sleep(0.3)
    out.append(peak)


def run(tag, params):
    dst = os.path.join(OUT, f"sweep_{tag}.mp4")
    cfg = os.path.join(WORK, f"sweep_{tag}.json")
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"src": SRC, "dst": dst, "params": params}, f, ensure_ascii=False)
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["FLASHVSR_DEC_BACKEND"] = "torch"
    stop, peaks = threading.Event(), []
    th = threading.Thread(target=sample_peak, args=(stop, peaks), daemon=True)
    th.start()
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-u", WORKER, cfg], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    stop.set()
    th.join(timeout=2)
    info = None
    err = None
    for line in (p.stdout or "").splitlines():
        if line.startswith("@@DONE "):
            info = json.loads(line[7:])
        elif line.startswith("@@ERR "):
            err = json.loads(line[6:]).get("error", "")[:160]
    peak = peaks[0] if peaks else 0
    left = 8151 - peak
    print(f"[{tag}] 峰值 {peak:5d} MiB（余 {left:4d}）  {'OK' if info else 'OOM/FAIL: ' + str(err)}"
          f"   [耗时 {wall:.0f}s 仅供参考]", flush=True)
    return dict(tag=tag, wall=wall, peak=peak, ok=bool(info), out=dst, err=err)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for tag, kw in CONFIGS:
        rows.append(run(tag, base(**kw)))
        time.sleep(5)      # 让上一个进程的显存彻底释放
    print("\n=== 稳定性汇总（按显存峰值排序）===")
    for r in sorted(rows, key=lambda x: x["peak"]):
        flag = "✅ 可跑" if r["ok"] else "❌ OOM"
        print(f"{r['tag']:10s} 峰值 {r['peak']:5d} MiB  {flag}")
    ok = [r for r in rows if r["ok"]]
    if ok:
        print("\n=== 时序一致性（temporal_check，越小越稳）===")
        for r in ok:
            p = subprocess.run([sys.executable, os.path.join(ROOT, "temporal_check.py"), r["out"]],
                               cwd=ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            tail = [l for l in (p.stdout or "").splitlines() if l.strip()][-3:]
            print(f"[{r['tag']}] " + " | ".join(tail))


if __name__ == "__main__":
    main()
