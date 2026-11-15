# -*- coding: utf-8 -*-
"""修正版：先预热（含 cudnn autotune），reset 峰值后再测，避免把 autotune workspace 算进去。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
import engine as E

torch.backends.cudnn.benchmark = True


def main():
    mod = E.load_node_module()
    pipe = mod.init_pipeline("FlashVSR-v1.1", "tiny", "cuda:0", torch.bfloat16)
    dec = pipe.TCDecoder.to("cuda")
    B, C, H, W = 1, 16, 64, 64
    print(f"{'T':>3s} {'mode':>9s} {'耗时':>9s} {'峰值(预热后)':>13s}")
    for T in [int(x) for x in (sys.argv[1:] or ["2", "4", "8"])]:
        lat = torch.randn(B, C, T, H, W, device="cuda", dtype=torch.bfloat16)
        cond = torch.randn(B, 3, T * 4, H * 8, W * 8, device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
        for par in (False, True):
            # 预热：完成 autotune + 让分配器缓存稳定
            dec.clean_mem()
            with torch.no_grad():
                dec.decode_video(lat.transpose(1, 2).contiguous(), parallel=par,
                                 show_progress_bar=False, cond=cond.contiguous())
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()      # ← 关键：autotune 之后再 reset
            base = torch.cuda.memory_allocated()
            dec.clean_mem()
            t0 = time.time()
            with torch.no_grad():
                dec.decode_video(lat.transpose(1, 2).contiguous(), parallel=par,
                                 show_progress_bar=False, cond=cond.contiguous())
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated()
            print(f"{T:3d} {'parallel' if par else 'seq':>9s} {dt * 1000:8.0f}ms "
                  f"{peak / 1024 ** 3:10.2f} GiB  (增量 {(peak - base) / 1024 ** 3:+.2f})", flush=True)
        del lat, cond
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
