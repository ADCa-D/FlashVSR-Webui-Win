# -*- coding: utf-8 -*-
"""能不能把整步 DiT 变成**可复现**的？这是所有 A/B 的前提。

逐个试：
  1. CUBLAS_WORKSPACE_CONFIG=:4096:8 + use_deterministic_algorithms(True, warn_only=False)
  2. 再把 self-attn 换成 SDPA（排除 Triton kernel）
每组连跑 3 次，看相对差是否真的为 0。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import engine as E  # noqa: E402


def main():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant=os.environ.get("BENCH_QUANT", "mxfp8"))
    pipe = eng.pipe
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    wan = sys.modules["flashvsr_node.src.models.wan_video_dit"]
    pl = sys.modules["flashvsr_node.src.pipelines.flashvsr_tiny_long"]

    G = 16
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(1, 16, 6, 2 * G, 2 * G, generator=g).to(dtype=torch.bfloat16, device="cuda")
    ts = torch.tensor([500.0], dtype=torch.bfloat16, device="cuda")
    dit.reinit_cross_kv(pipe.prompt_emb_posi["context"])
    with torch.no_grad():
        t = dit.time_embedding(wan.sinusoidal_embedding_1d(dit.freq_dim, ts))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    nb = len(dit.blocks)

    def run():
        with torch.no_grad():
            y = pl.model_fn_wan_video(
                dit, x, ts, None, LQ_latents=None, is_stream=True,
                pre_cache_k=[None] * nb, pre_cache_v=[None] * nb,
                topk_ratio=22.5, kv_ratio=2.0, cur_process_idx=0,
                t_mod=t_mod, t=t, local_range=9)[0].float().clone()
        torch.cuda.synchronize()
        return y

    def spread(tag, n=3):
        ys = [run() for _ in range(n)]
        base = ys[0]
        rels = [float((y - base).norm() / base.norm()) for y in ys[1:]]
        flag = "**可复现**" if max(rels) == 0 else "不可复现"
        print("%-46s %s  rel=%s" % (tag, flag, ["%.1e" % r for r in rels]))

    spread("基线（默认设置）")

    # 1) 严格确定性 + cuBLAS workspace
    torch.use_deterministic_algorithms(True)
    spread("+use_deterministic_algorithms(True)")

    # 2) 换掉 Triton 注意力
    orig = wan.sparse_sageattn
    wan.sparse_sageattn = lambda q, k, v, **kw: F.scaled_dot_product_attention(q, k, v)
    pl.model_fn_wan_video.__globals__["sparse_sageattn"] = wan.sparse_sageattn
    spread("+attn 换 SDPA")
    wan.sparse_sageattn = orig
    pl.model_fn_wan_video.__globals__["sparse_sageattn"] = orig


if __name__ == "__main__":
    print("CUBLAS_WORKSPACE_CONFIG =", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    main()
