# -*- coding: utf-8 -*-
"""同一进程里连跑 5 次完全相同的单步，看结果是否稳定。

不稳 → 模型本身有非确定性（cudnn.benchmark 选算法 / kernel 里有 atomic）。
"""
import os
import sys

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
    print("cudnn.benchmark =", torch.backends.cudnn.benchmark,
          "| deterministic =", torch.backends.cudnn.deterministic,
          "| use_deterministic_algorithms =", torch.are_deterministic_algorithms_enabled())

    g = torch.Generator().manual_seed(1234)
    x = torch.randn(1, 16, 6, 16, 16, generator=g).to(dtype=torch.bfloat16, device="cuda")
    ts = torch.tensor([500.0], dtype=torch.bfloat16, device="cuda")
    dit.reinit_cross_kv(pipe.prompt_emb_posi["context"])
    with torch.no_grad():
        t = dit.time_embedding(wan.sinusoidal_embedding_1d(dit.freq_dim, ts))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    nb = len(dit.blocks)

    ys = []
    for i in range(5):
        with torch.no_grad():
            y = pl.model_fn_wan_video(
                dit, x, ts, None, LQ_latents=None, is_stream=True,
                pre_cache_k=[None] * nb, pre_cache_v=[None] * nb,
                topk_ratio=2.0, kv_ratio=3.0, cur_process_idx=0,
                t_mod=t_mod, t=t, local_range=9)[0].float().clone()
        torch.cuda.synchronize()
        ys.append(y)
        print("run%d  y.mean %.6f  absmean %.6f  y[0,0,0,0,0] %.5f"
              % (i, y.mean(), y.abs().mean(), y.flatten()[0]))

    base = ys[0]
    for i, y in enumerate(ys[1:], 1):
        d = (y - base).abs()
        print("run%d vs run0: maxdiff %.5f  rel %.2e" % (i, d.max(), (y - base).norm() / base.norm()))

    # 关掉 cudnn benchmark 再跑两次
    torch.backends.cudnn.benchmark = False
    print("--- cudnn.benchmark=False ---")
    for i in range(2):
        with torch.no_grad():
            y = pl.model_fn_wan_video(
                dit, x, ts, None, LQ_latents=None, is_stream=True,
                pre_cache_k=[None] * nb, pre_cache_v=[None] * nb,
                topk_ratio=2.0, kv_ratio=3.0, cur_process_idx=0,
                t_mod=t_mod, t=t, local_range=9)[0].float().clone()
        print("run%d  y.mean %.6f  rel vs run0 %.2e"
              % (i, y.mean(), (y - base).norm() / base.norm()))


if __name__ == "__main__":
    main()
