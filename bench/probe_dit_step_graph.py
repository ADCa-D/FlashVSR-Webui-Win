# -*- coding: utf-8 -*-
"""DiT 整步（30 block）能否整段 CUDA Graph 捕获？能的话 launch 开销基本归零。

只做可行性判断：
  eager 计时 → 捕获 → replay 计时 → 与 eager 输出比 maxdiff
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

import engine as E  # noqa: E402


def main():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="none")
    pipe = eng.pipe
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    wan = sys.modules["flashvsr_node.src.models.wan_video_dit"]
    pl = sys.modules["flashvsr_node.src.pipelines.flashvsr_tiny_long"]

    g = torch.Generator().manual_seed(1234)
    x = torch.randn(1, 16, 6, 16, 16, generator=g).to(dtype=torch.bfloat16, device="cuda")
    ts = torch.tensor([500.0], dtype=torch.bfloat16, device="cuda")
    dit.reinit_cross_kv(pipe.prompt_emb_posi["context"])
    with torch.no_grad():
        t = dit.time_embedding(wan.sinusoidal_embedding_1d(dit.freq_dim, ts))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    nb = len(dit.blocks)

    def one():
        with torch.no_grad():
            return pl.model_fn_wan_video(
                dit, x, ts, None, LQ_latents=None, is_stream=True,
                pre_cache_k=[None] * nb, pre_cache_v=[None] * nb,
                topk_ratio=2.0, kv_ratio=3.0, cur_process_idx=0,
                t_mod=t_mod, t=t, local_range=9)

    def bench(fn, n=10, warm=3):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000

    y_ref = one()[0]
    torch.cuda.synchronize()
    print("eager %.1f ms/步" % bench(one), flush=True)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            one()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = one()
        torch.cuda.synchronize()
        print("捕获成功", flush=True)
        graph.replay()
        torch.cuda.synchronize()
        y_g = out[0]
        print("graph replay %.1f ms/步" % bench(graph.replay), flush=True)
        print("maxdiff graph vs eager: %.4f" % (y_g.float() - y_ref.float()).abs().max().item())
    except Exception as e:
        print("捕获失败: %s: %s" % (type(e).__name__, str(e)[:300]))


if __name__ == "__main__":
    main()
