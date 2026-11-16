# -*- coding: utf-8 -*-
"""改造验收：GPU 版 prepare / GPU 画布 / 异步写盘的等价性与耗时（真实模型，544p）。

三段对比，同一个 seed、同一批帧：
  A 新路径（GPU prepare + GPU 画布）
  B 新画布 + 节点原版 prepare（CPU 往返）——用来证明 A 与 B 数值等价
  C 旧路径（节点原版 prepare + CPU 画布）

用法：python bench/smoke_opt.py [帧数] [tile]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import engine as E  # noqa: E402

# 同时写日志文件：tqdm 进度条会淹没 job 输出尾部，结果必须落盘才可靠
_LOG = open(os.path.join(ROOT, "work", "smoke_opt.log"), "w", encoding="utf-8", buffering=1)


class _Tee(object):
    def write(self, s):
        try:
            _LOG.write(s)
        except Exception:
            pass
        try:
            sys.__stdout__.write(s)
        except Exception:
            pass
        return len(s)

    def flush(self):
        pass


sys.stdout = _Tee()

SRC = os.path.join(r"D:\ComfyUI_windows_portable\ComfyUI\input", "flashvsr_long_544.mp4")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 9
TILE = int(sys.argv[2]) if len(sys.argv) > 2 else 128
OV = max(24, TILE // 4)

eng = E.Engine()
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
frames = E.decode_frames_range(SRC, 0, N, pre_scale=0.5, fps=25.0)
print(f"输入 {frames.shape} dtype={frames.dtype} tile={TILE} ov={OV}", flush=True)


def run(tag, prep, canvas):
    os.environ["FLASHVSR_GPU_PREPARE"] = "1" if prep else "0"
    os.environ["FLASHVSR_GPU_CANVAS"] = "1" if canvas else "0"
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = eng._run_window(frames, 2, TILE, 3, tile_overlap=OV)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"{tag:28s} {dt:5.2f}s  峰值 reserved={torch.cuda.max_memory_reserved()/2**20:6.0f} MiB",
          flush=True)
    torch.cuda.reset_peak_memory_stats()
    return np.asarray(out, dtype=np.float32)


run("预热", True, True)          # 让 cudnn/triton 预热，别把编译时间算进 A
a = run("A GPU prepare+canvas", True, True)
b = run("B 节点 prepare+canvas", False, True)
c = run("C 节点 prepare+CPU画布", False, False)


def stats(x, y, tag):
    if x.shape != y.shape:
        print(f"{tag}: 形状不同 {x.shape} vs {y.shape}")
        return
    d = np.abs(x - y)
    mse = float((d ** 2).mean())
    # 数据是 0..255 的 uint8 尺度，PSNR 的峰值必须用 255，否则"dB"数没有意义
    psnr = 99.0 if mse <= 0 else 10 * np.log10(255.0 * 255.0 / mse)
    print(f"{tag}: max|Δ|={d.max():.3e} mean|Δ|={d.mean():.3e} PSNR={psnr:.2f} dB", flush=True)


stats(a, b, "A vs B（只差 prepare 在哪跑）")
stats(a, c, "A vs C（整条新路径 vs 旧路径）")
