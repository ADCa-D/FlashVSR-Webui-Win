# -*- coding: utf-8 -*-
"""看 ck.scaled_mm_nvfp4 到底调用了哪个 kernel：真有 fp4 MMA，还是偷偷 dequant 回 bf16？"""
import torch
import comfy_kitchen as ck
from torch.profiler import ProfilerActivity, profile

K, N = 8960, 1536


def ts(t):
    return (t.abs().amax() / (6.0 * 448.0)).clamp(min=1e-8).float().reshape(1)


w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
wq4, wb4 = ck.quantize_nvfp4(w, ts(w), pad_16x=True)
wq8, wb8 = ck.quantize_mxfp8(w)

for M in (256, 2304):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
    xq4, xb4 = ck.quantize_nvfp4(x, ts(x), pad_16x=True)
    xq8, xb8 = ck.quantize_mxfp8(x)
    for tag, fn in (
        ("nvfp4", lambda: ck.scaled_mm_nvfp4(xq4, wq4, ts(x), ts(w), xb4, wb4, out_dtype=torch.bfloat16)),
        ("mxfp8", lambda: ck.scaled_mm_mxfp8(xq8, wq8, xb8, wb8, out_dtype=torch.bfloat16)),
    ):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
        evts = [e for e in prof.key_averages() if getattr(e, "self_device_time_total", 0) > 0]
        evts.sort(key=lambda e: -e.self_device_time_total)
        print(f"\nM={M} {tag}: {len(evts)} 个 kernel")
        for e in evts[:4]:
            print(f"   {e.self_device_time_total/1e3/10:7.3f} ms/次  {e.count:3d}x  {e.key[:80]}")
    del x, xq4, xb4, xq8, xb8
    torch.cuda.empty_cache()
