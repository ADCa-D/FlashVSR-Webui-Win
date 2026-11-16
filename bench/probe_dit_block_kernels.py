# -*- coding: utf-8 -*-
"""kernel 级 profile 单个 DiT block：看到底谁在吃时间。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

import engine as E

SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_real.mp4"
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "kern_probe.mp4")
IDX = 4


def main():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", quant="mxfp8")
    block = eng.pipe.dit.blocks[IDX]
    cap = {}

    def hook(m, args, kwargs):
        if not cap:
            cap["args"] = tuple(a.detach() if torch.is_tensor(a) else a for a in args)
            cap["kwargs"] = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in kwargs.items()}

    h = block.register_forward_pre_hook(hook, with_kwargs=True)
    params = dict(scale=2, tile=128, kv=3.0, chunk=0, overlap=8, mode="tiny-long",
                  attention="sparse", encoder="cpu", quality=15, quant="mxfp8",
                  budget="auto", pre_scale=1.0, tile_overlap=12,
                  seg_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", "segments"))
    try:
        eng.upscale_video(SRC, DST, params, lambda *a, **k: None)
    finally:
        h.remove()
    if not cap:
        print("hook 未触发")
        return

    from torch.profiler import ProfilerActivity, profile
    with torch.no_grad():
        for _ in range(3):
            block(*cap["args"], **cap["kwargs"])
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                block(*cap["args"], **cap["kwargs"])
            torch.cuda.synchronize()

    rows = [(e.device_time_total / 3 / 1000.0, e.key) for e in prof.key_averages()
            if e.device_time_total > 0]
    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows)
    print(f"\n单 block 设备总时间 {tot:.2f} ms（前 10 个 kernel）")
    for ms, k in rows[:10]:
        print(f"  {ms:7.2f} ms  {100*ms/tot:5.1f}%  {k[:80]}")


if __name__ == "__main__":
    main()
