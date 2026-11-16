# -*- coding: utf-8 -*-
"""Python 侧同一个 DiT forward 的显存峰值，用来和 cpp/dit.exe 对标。

两边都是"加载 bf16 权重 + 跑一次 30 层前向"，所以 allocated peak 之差就是激活开销。
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

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
    t = dit.time_embedding(wan.sinusoidal_embedding_1d(dit.freq_dim, ts))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    with torch.no_grad():
        for _ in range(2):   # 预热一次
            pl.model_fn_wan_video(
                dit, x, ts, None, LQ_latents=None, is_stream=True,
                pre_cache_k=[None] * len(dit.blocks), pre_cache_v=[None] * len(dit.blocks),
                topk_ratio=2.0, kv_ratio=3.0, cur_process_idx=0, t_mod=t_mod, t=t)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pl.model_fn_wan_video(
            dit, x, ts, None, LQ_latents=None, is_stream=True,
            pre_cache_k=[None] * len(dit.blocks), pre_cache_v=[None] * len(dit.blocks),
            topk_ratio=2.0, kv_ratio=3.0, cur_process_idx=0, t_mod=t_mod, t=t)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    print(f"forward {dt*1e3:.0f} ms")

    free_b, total_b = torch.cuda.mem_get_info()
    print(f"allocated peak {torch.cuda.max_memory_allocated()/2**20:.0f} MiB / "
          f"reserved peak {torch.cuda.max_memory_reserved()/2**20:.0f} MiB")
    print(f"device peak {(total_b-free_b)/2**20:.0f} MiB (total {total_b/2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
