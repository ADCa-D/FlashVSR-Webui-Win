# -*- coding: utf-8 -*-
"""抓一个 tile 的 kernel 级 profile：判断瓶颈是 GEMM、attention、elementwise 还是 launch 空隙。

用法：
  python profile_tile.py [tile] [frames]
  set PROF_OFFLOAD=1   # 复现旧的 force_offload=True 行为（默认 0 = 现状）
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch  # noqa: E402
import engine as E  # noqa: E402

SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_real.mp4"
TILE = int(sys.argv[1]) if len(sys.argv) > 1 else 128
FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 25
OFFLOAD = os.environ.get("PROF_OFFLOAD", "0") == "1"


def classify(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("gemm", "cutlass", "cublas", "sgemm", "hgmma", "nvjet")):
        return "GEMM"
    if "conv" in n or "implicit" in n:
        return "CONV"
    if any(k in n for k in ("attn", "attention", "sage", "flash", "softmax")):
        return "ATTN"
    if any(k in n for k in ("memcpy", "elementwise", "vectorized", "fill", "copy", "cat")):
        return "MEM/ELEM"
    return "OTHER"


def main():
    mod = E.load_node_module()
    frames_u8, info = E.decode_frames(SRC)
    frames = torch.from_numpy(frames_u8).float().div_(255.0)
    print(f"frames {tuple(frames.shape)} force_offload={OFFLOAD}", flush=True)

    t0 = time.time()
    pipe = mod.init_pipeline("FlashVSR-v1.1", "tiny", "cuda:0", torch.bfloat16)
    print(f"load {time.time() - t0:.1f}s", flush=True)

    tf = frames[:, :min(TILE, frames.shape[1]), :min(TILE, frames.shape[2]), :]
    LQ, th, tw, F = mod.prepare_input_tensor(tf, "cuda:0", scale=4, dtype=torch.bfloat16)
    LQ = LQ.to("cuda:0")
    print(f"tile {th}x{tw} x {F} frames", flush=True)

    kw = dict(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
              tiled=True, LQ_video=LQ, num_frames=F, height=th, width=tw,
              is_full_block=False, if_buffer=True,
              topk_ratio=2.0 * 768 * 1280 / (th * tw), kv_ratio=3.0, local_range=11,
              color_fix=True, unload_dit=False, force_offload=OFFLOAD)

    with torch.no_grad():
        _ = pipe(**kw)          # 预热：cudnn autotune / triton 编译都排除在 profile 之外
    torch.cuda.synchronize()
    print("warmup done", flush=True)

    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    wall0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            _ = pipe(**kw)
        torch.cuda.synchronize()
    wall = time.time() - wall0

    buckets, total_us, launches = {}, 0.0, 0
    for e in prof.key_averages():
        cuda = getattr(e, "self_device_time_total", 0.0) or 0.0
        if cuda <= 0:
            continue
        buckets[classify(e.key)] = buckets.get(classify(e.key), 0.0) + cuda
        total_us += cuda
        launches += e.count

    print("\n=== summary ===")
    print(f"wall {wall:.2f}s   cuda kernels {total_us / 1e6:.2f}s   "
          f"idle/gap {wall - total_us / 1e6:.2f}s ({(wall - total_us / 1e6) / wall * 100:.1f}%)")
    print(f"kernel entries {launches}")
    for k, v in sorted(buckets.items(), key=lambda x: -x[1]):
        print(f"  {k:9s} {v / 1e6:6.2f}s  {v / total_us * 100:5.1f}%")

    print("\n=== top 15 ===")
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=15))
    print(f"peak allocated {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
