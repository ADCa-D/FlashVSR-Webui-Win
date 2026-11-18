# -*- coding: utf-8 -*-
"""定位非确定性的算子：把 self-attn 的 Triton kernel 换成 SDPA，看是否变确定。

如果换掉就确定 → 源头是 sparse_sageattn（自定义 Triton kernel）
如果还非确定 → 源头在 GEMM（cuBLAS split-K）或别处
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

    def run():
        with torch.no_grad():
            y = pl.model_fn_wan_video(
                dit, x, ts, None, LQ_latents=None, is_stream=True,
                pre_cache_k=[None] * nb, pre_cache_v=[None] * nb,
                topk_ratio=2.0, kv_ratio=3.0, cur_process_idx=0,
                t_mod=t_mod, t=t, local_range=9)[0].float().clone()
        torch.cuda.synchronize()
        return y

    def spread(tag, n=3):
        ys = [run() for _ in range(n)]
        base = ys[0]
        rels = [float((y - base).norm() / base.norm()) for y in ys[1:]]
        print("%-26s 相对差 %s  → %s" % (tag, ["%.1e" % r for r in rels],
                                        "确定" if max(rels) == 0 else "非确定"))

    spread("现状 sparse_sageattn")

    # 换成 SDPA（无 mask 语义：把 mask_id 忽略）
    orig = wan.sparse_sageattn
    wan.sparse_sageattn = lambda q, k, v, **kw: F.scaled_dot_product_attention(q, k, v)
    pl.model_fn_wan_video.__globals__["sparse_sageattn"] = wan.sparse_sageattn
    spread("self-attn 换 SDPA")
    wan.sparse_sageattn = orig
    pl.model_fn_wan_video.__globals__["sparse_sageattn"] = orig

    # 只蒙掉 attention（输出置零），剩下纯 GEMM/elementwise
    wan.sparse_sageattn = lambda q, k, v, **kw: torch.zeros_like(q)
    pl.model_fn_wan_video.__globals__["sparse_sageattn"] = wan.sparse_sageattn
    spread("self-attn 输出置零")
    wan.sparse_sageattn = orig
    pl.model_fn_wan_video.__globals__["sparse_sageattn"] = orig


if __name__ == "__main__":
    main()
