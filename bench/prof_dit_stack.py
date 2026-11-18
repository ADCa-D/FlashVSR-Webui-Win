# -*- coding: utf-8 -*-
"""带调用栈的 profile：把 copy_/clone/contiguous/to/mul 归因到具体 Python 行。

mxfp8 下 GEMM 只占 40%，剩下 48% 是数据搬运。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

import engine as E  # noqa: E402


def main():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None,
                      quant=os.environ.get("BENCH_QUANT", "mxfp8"))
    pipe = eng.pipe
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    wan = sys.modules["flashvsr_node.src.models.wan_video_dit"]
    pl = sys.modules["flashvsr_node.src.pipelines.flashvsr_tiny_long"]

    G = int(os.environ.get("BENCH_G", "16"))
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(1, 16, 6, 2 * G, 2 * G, generator=g).to(dtype=torch.bfloat16, device="cuda")
    ts = torch.tensor([500.0], dtype=torch.bfloat16, device="cuda")
    dit.reinit_cross_kv(pipe.prompt_emb_posi["context"])
    with torch.no_grad():
        t = dit.time_embedding(wan.sinusoidal_embedding_1d(dit.freq_dim, ts))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    nb = len(dit.blocks)
    tk = float(os.environ.get("BENCH_TOPK", "22.5"))
    kv = float(os.environ.get("BENCH_KV", "2.0"))

    def one():
        with torch.no_grad():
            return pl.model_fn_wan_video(
                dit, x, ts, None, LQ_latents=None, is_stream=True,
                pre_cache_k=[None] * nb, pre_cache_v=[None] * nb,
                topk_ratio=tk, kv_ratio=kv, cur_process_idx=0,
                t_mod=t_mod, t=t, local_range=9)

    for _ in range(3):
        one()
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                 with_stack=True) as prof:
        for _ in range(2):
            one()
        torch.cuda.synchronize()

    import sys as _sys
    _sys.setrecursionlimit(200000)
    ka = prof.key_averages(group_by_stack_n=3)
    print("=== 按 CUDA 时间，带栈 ===")
    print(ka.table(sort_by="cuda_time_total", row_limit=90,
                   max_name_column_width=34, max_src_column_width=70))


if __name__ == "__main__":
    main()
