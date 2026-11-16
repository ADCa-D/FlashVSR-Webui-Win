# -*- coding: utf-8 -*-
"""DiT block 里到底是哪一步不能进 CUDA Graph：分别试 FFN 和 attention。

结论用途：如果 FFN 能捕获，即使 attention 不行，也可以只把 FFN（NVFP4 开销所在）图上。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

import engine as E

SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_real.mp4"
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "graph_probe2.mp4")
BLOCK_IDX = 4


def bench(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def try_graph(fn, tag):
    """返回 (是否成功, 图回放耗时 or None)"""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        t = bench(lambda: g.replay(), n=20)
        print(f"  {tag}: ✅ 可捕获，回放 {t:.2f} ms", flush=True)
        return True, t
    except Exception as e:  # noqa
        print(f"  {tag}: ❌ {type(e).__name__}: {str(e)[:150]}", flush=True)
        return False, None


def main():
    attn = sys.argv[1] if len(sys.argv) > 1 else "sparse"
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", attn, quant="mxfp8")
    block = eng.pipe.dit.blocks[BLOCK_IDX]
    cap = {}

    def hook(m, args, kwargs):
        if cap:
            return
        cap["x"] = args[0].detach()
        cap["args"] = tuple(a.detach() if torch.is_tensor(a) else a for a in args)
        cap["kwargs"] = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in kwargs.items()}

    h = block.register_forward_pre_hook(hook, with_kwargs=True)
    params = dict(scale=2, tile=128, kv=3.0, chunk=0, overlap=8, mode="tiny-long",
                  attention=attn, encoder="cpu", quality=15, quant="mxfp8",
                  budget="auto", pre_scale=1.0, tile_overlap=12,
                  seg_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", "segments"))
    try:
        eng.upscale_video(SRC, DST, params, lambda *a, **k: None)
    finally:
        h.remove()

    if not cap:
        print("hook 没触发")
        return
    x = cap["x"]
    print(f"block {BLOCK_IDX} 输入 {tuple(x.shape)}", flush=True)

    with torch.no_grad():
        t_all = bench(lambda: block(*cap["args"], **cap["kwargs"]))
        print(f"整块 eager {t_all:.2f} ms", flush=True)
        try_graph(lambda: block(*cap["args"], **cap["kwargs"]), "整块")

        # FFN 单独：block.ffn 是 Sequential(Linear, GELU, Linear)
        t_ffn = bench(lambda: block.ffn(x))
        print(f"FFN eager {t_ffn:.2f} ms", flush=True)
        try_graph(lambda: block.ffn(x), "FFN")

        # attention 单独：用 block 的 norm + qkv
        def attn_only():
            y = block.norm1(x) if hasattr(block, "norm1") else x
            return block.self_attn(y, cap["args"][3], cap["args"][4] if len(cap["args"]) > 4 else None,
                                   cap["args"][5] if len(cap["args"]) > 5 else None)
        try:
            t_a = bench(attn_only)
            print(f"attn eager {t_a:.2f} ms", flush=True)
            try_graph(attn_only, "attn")
        except Exception as e:  # noqa
            print(f"attn 单独测不了: {type(e).__name__}: {str(e)[:120]}", flush=True)


if __name__ == "__main__":
    main()
