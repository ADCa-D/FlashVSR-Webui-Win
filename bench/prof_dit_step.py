# -*- coding: utf-8 -*-
"""DiT 单步算子级 profile：看 CUDA 时间里谁最贵（含 rope / mask 构造 / rearrange）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

import engine as E  # noqa: E402


def main():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant=os.environ.get("BENCH_QUANT", "none"))
    pipe = eng.pipe
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    wan = sys.modules["flashvsr_node.src.models.wan_video_dit"]
    pl = sys.modules["flashvsr_node.src.pipelines.flashvsr_tiny_long"]

    g = torch.Generator().manual_seed(1234)
    G = int(os.environ.get("BENCH_G", "8"))
    x = torch.randn(1, 16, 6, 2 * G, 2 * G, generator=g).to(dtype=torch.bfloat16, device="cuda")
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
                topk_ratio=float(os.environ.get("BENCH_TOPK", "2.0")), kv_ratio=float(os.environ.get("BENCH_KV", "3.0")), cur_process_idx=0,
                t_mod=t_mod, t=t, local_range=9)

    for _ in range(3):
        one()
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(3):
            one()
        torch.cuda.synchronize()

    print("=== 按 CUDA 总时间 top 18 ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))
    print("=== 按 CPU(自身) top 12 ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))


if __name__ == "__main__":
    main()
