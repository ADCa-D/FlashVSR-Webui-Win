# -*- coding: utf-8 -*-
"""解码器「逐帧图」用 torch.compile 融合，对比 eager。

动机：同一张图 TRT 能到 1.79×，说明约 45% 时间花在逐层 kernel 启动 / 没融合
（MemBlock 的 conv(cat([x,past])) + skip + relu 之间全是 elementwise）。
inductor 能把它们融掉，且**不额外占显存**（不像 TRT 要 3 GB workspace）。

用法：python bench/probe_compile_step.py 48 [max-autotune]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

import engine as E
from trt_decoder import _Step

H = int(sys.argv[1]) if len(sys.argv) > 1 else 48
MODE = sys.argv[2] if len(sys.argv) > 2 else "default"
NF = 3


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
    step = _Step(dec.decoder, layers).eval()

    torch.manual_seed(0)
    latent = torch.randn(1, 16, H, H, device="cuda", dtype=torch.bfloat16) * 0.5
    cond = torch.randn(1, 768, H, H, device="cuda", dtype=torch.bfloat16)
    x = torch.cat([cond, latent], dim=1)

    # past 形状：按真实数据流走一遍拿
    shapes = {}
    queue = [(x, 0)]
    N0 = 1
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
                for sub in reversed(xt.view(N0, b.stride * c_, h_, w_).chunk(b.stride, 1)):
                    queue.insert(0, (sub, i + 1))
            else:
                xt = b(xt)
                queue.insert(0, (xt, i + 1))
    zeros = [torch.zeros(shapes[L], device="cuda", dtype=torch.bfloat16) for L in sorted(shapes)]
    print(f"latent {H}  past={[tuple(z.shape) for z in zeros][:3]}...", flush=True)

    def run(fn, n=4, warm=2):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1000

    with torch.no_grad():
        t_eager = run(lambda: step(x, *zeros))
    print(f"[eager] {t_eager:7.1f} ms/帧", flush=True)

    t0 = time.perf_counter()
    cstep = torch.compile(step, mode=MODE, dynamic=False)
    with torch.no_grad():
        out_c = cstep(x, *zeros)[0]
    print(f"[compile {MODE}] 首帧（含编译）{time.perf_counter()-t0:.1f}s", flush=True)

    with torch.no_grad():
        t_comp = run(lambda: cstep(x, *zeros))
    with torch.no_grad():
        out_e = step(x, *zeros)[0]
    md = max(float((a - b).abs().max()) for a, b in zip(out_c, out_e))
    print(f"[compile] {t_comp:7.1f} ms/帧  →  {t_eager / t_comp:.2f}×   maxdiff={md:.4f}", flush=True)


if __name__ == "__main__":
    main()
