# -*- coding: utf-8 -*-
"""解码器 eager：bf16 vs fp16 vs fp32 的单帧耗时（同一张图，只换精度）。

用来判断「TRT 的 1.79× 到底是算法更快，还是 fp16 比 bf16 快」。
用法：python bench/probe_dec_dtype.py 96
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

import engine as E
from trt_decoder import _Step

H = int(sys.argv[1]) if len(sys.argv) > 1 else 96


def run(dec, step, x, zeros, n=4, warm=2):
    for _ in range(warm):
        step(x, *zeros)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        step(x, *zeros)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", quant="none")
    dec = eng.pipe.TCDecoder.to("cuda")
    try:
        eng.pipe.dit.to("cpu")
    except Exception:
        pass
    torch.cuda.empty_cache()
    dec.clean_mem()

    layers = [i for i, b in enumerate(dec.decoder) if type(b).__name__ == "MemBlock"]

    # 形状探测（按当前 bf16）
    shapes, queue = {}, []
    xb = torch.zeros(1, dec.decoder[1].in_channels, H, H, device="cuda", dtype=torch.bfloat16)
    queue = [(xb, 0)]
    with torch.no_grad():
        while queue:
            xt, i = queue.pop(0)
            if i == len(dec.decoder):
                continue
            b = dec.decoder[i]
            nm = type(b).__name__
            if nm == "MemBlock":
                shapes.setdefault(i, tuple(xt.shape))
                xt = b(xt, torch.zeros_like(xt))
                queue.insert(0, (xt, i + 1))
            elif nm == "TGrow":
                xt = b(xt)
                _nt, c_, h_, w_ = xt.shape
                for sub in reversed(xt.view(1, b.stride * c_, h_, w_).chunk(b.stride, 1)):
                    queue.insert(0, (sub, i + 1))
            else:
                xt = b(xt)
                queue.insert(0, (xt, i + 1))
    del xb
    torch.cuda.empty_cache()
    order = sorted(shapes)

    for dt in (torch.bfloat16, torch.float16, torch.float32):
        dec.to(dt)
        step = _Step(dec.decoder, order).eval()
        x = torch.randn(1, dec.decoder[1].in_channels, H, H, device="cuda", dtype=dt) * 0.3
        zeros = [torch.zeros(shapes[L], device="cuda", dtype=dt) for L in order]
        with torch.no_grad():
            ms = run(dec, step, x, zeros)
        print(f"{str(dt).replace('torch.',''):9s} {ms:7.1f} ms/帧", flush=True)
        del step, zeros, x
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
