# -*- coding: utf-8 -*-
"""单个 DiT block 能否进 CUDA Graph？能的话收益多少？

做法：真实跑一次推理，用 hook 抓住某个 block 的真实输入，然后：
  1) eager 计时
  2) CUDA Graph 捕获（失败会打印原因）
  3) replay 计时 + 输出一致性
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

import engine as E

SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_real.mp4"
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "graph_probe.mp4")
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


def main():
    eng = E.Engine()
    attn = sys.argv[1] if len(sys.argv) > 1 else "sparse"
    eng.ensure_loaded("tiny-long", attn, quant="mxfp8")
    dit = eng.pipe.dit
    block = dit.blocks[BLOCK_IDX]

    captured = {}

    def pre_hook(module, args, kwargs):
        if captured:
            return
        captured["args"] = tuple(a.detach() if torch.is_tensor(a) else a for a in args)
        captured["kwargs"] = {k: (v.detach() if torch.is_tensor(v) else v)
                              for k, v in kwargs.items()}

    h = block.register_forward_pre_hook(pre_hook, with_kwargs=True)
    params = dict(scale=2, tile=128, kv=3.0, chunk=0, overlap=8, mode="tiny-long",
                  attention="sparse", encoder="cpu", quality=15, quant="mxfp8",
                  budget="auto", pre_scale=1.0, tile_overlap=12,
                  seg_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", "segments"))
    try:
        eng.upscale_video(SRC, DST, params, lambda *a, **k: None)
    finally:
        h.remove()

    if not captured:
        print("没抓到 block 输入（hook 未触发）")
        return
    args, kwargs = captured["args"], captured["kwargs"]
    print(f"抓到 block {BLOCK_IDX}：args={[tuple(a.shape) if torch.is_tensor(a) else a for a in args][:4]}")
    print(f"kwargs keys={list(kwargs)[:6]}")

    with torch.no_grad():
        y_ref = block(*args, **kwargs)
        torch.cuda.synchronize()
        t_eager = bench(lambda: block(*args, **kwargs))
        print(f"eager {t_eager:7.2f} ms / block", flush=True)

        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    block(*args, **kwargs)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y_g = block(*args, **kwargs)
            torch.cuda.synchronize()
            t_graph = bench(lambda: g.replay(), n=20)
            a = y_g[0] if isinstance(y_g, tuple) else y_g
            b = y_ref[0] if isinstance(y_ref, tuple) else y_ref
            md = float((a.float() - b.float()).abs().max())
            print(f"graph {t_graph:7.2f} ms / block  (提速 {t_eager / t_graph:.2f}×)  maxdiff={md:.5f}",
                  flush=True)
        except Exception as e:  # noqa
            print(f"CUDA Graph 捕获失败: {type(e).__name__}: {str(e)[:300]}", flush=True)


if __name__ == "__main__":
    main()
