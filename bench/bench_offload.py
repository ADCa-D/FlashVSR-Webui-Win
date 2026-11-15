# -*- coding: utf-8 -*-
"""对比 force_offload True/False 对单 tile 时间的影响（PCIe 往返开销验证）。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch  # noqa: E402
import engine as E  # noqa: E402

SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_real.mp4"


def main():
    mod = E.load_node_module()
    frames_u8, info = E.decode_frames(SRC)
    frames = torch.from_numpy(frames_u8).float().div_(255.0)
    pipe = mod.init_pipeline("FlashVSR-v1.1", "tiny", "cuda:0", torch.bfloat16)
    print("pipeline ready", flush=True)

    for tile, kv, offload in ((128, 3.0, True), (128, 3.0, False), (112, 2.0, False), (96, 2.0, False)):
        tf = frames[:, :min(tile, frames.shape[1]), :min(tile, frames.shape[2]), :]
        LQ, th, tw, F = mod.prepare_input_tensor(tf, "cuda:0", scale=4, dtype=torch.bfloat16)
        LQ = LQ.to("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():
                t0 = time.time()
                _ = pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
                         tiled=True, LQ_video=LQ, num_frames=F, height=th, width=tw,
                         is_full_block=False, if_buffer=True,
                         topk_ratio=2.0 * 768 * 1280 / (th * tw), kv_ratio=kv, local_range=11,
                         color_fix=True, unload_dit=False, force_offload=offload)
                torch.cuda.synchronize()
                dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"tile={tile:3d} kv={kv} force_offload={str(offload):5s} -> {dt:5.2f}s  peak {peak:.2f} GiB", flush=True)
        except Exception as e:  # noqa
            print(f"tile={tile:3d} kv={kv} force_offload={str(offload):5s} -> FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
        finally:
            del LQ
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
