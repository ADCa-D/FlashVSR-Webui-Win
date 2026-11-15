# -*- coding: utf-8 -*-
"""TCDecoder 解码：NCHW vs channels_last(NHWC) 对比。"""
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

    B, C, Fl, H, W = 1, 16, 8, 64, 64
    lat = torch.randn(B, C, Fl, H, W, device="cuda", dtype=torch.bfloat16)
    cond = torch.randn(B, 3, Fl * 4, H * 8, W * 8, device="cuda", dtype=torch.bfloat16).clamp(-1, 1)

    def run(tag, cl=False, reps=3):
        if cl:
            dec.to(memory_format=torch.channels_last)
            x = lat.transpose(1, 2).contiguous(memory_format=torch.channels_last_3d)
            c = cond.contiguous(memory_format=torch.channels_last_3d)
        else:
            dec.to(memory_format=torch.contiguous_format)
            x = lat.transpose(1, 2).contiguous()
            c = cond.contiguous()
        best = 1e9
        out = None
        for _ in range(reps):
            dec.clean_mem()
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                out = dec.decode_video(x, parallel=False, show_progress_bar=False, cond=c)
            torch.cuda.synchronize()
            best = min(best, time.time() - t0)
        print(f"{tag:18s} {best * 1000:7.0f} ms  out {tuple(out.shape)}", flush=True)
        return out.float().clone()

    ref = run("NCHW")
    cl = run("channels_last", cl=True)
    print(f"最大绝对差 {float((cl - ref).abs().max()):.4f}  均值 {float((cl - ref).abs().mean()):.5f}", flush=True)


if __name__ == "__main__":
    main()
