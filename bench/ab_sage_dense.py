# -*- coding: utf-8 -*-
"""隔离实验：把 self-attn 从「block mask + sparse_sage」换成「无 mask + sageattn」。

做法：只把 generate_draft_block_mask_sage 换成返回 None，
flash_attention 会因为 attention_mask=None 落到已有的 SAGE_ATTN_AVAILABLE 分支。
其余一切（权重、输入、topk/kv 参数）保持不变。

输出：单步耗时、与现状的相对差、以及 mask 相关 kernel 是否消失。
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

import engine as E  # noqa: E402


def bench_step(fn, n=15, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(n):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / n


def main():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="none")
    pipe = eng.pipe
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    wan = sys.modules["flashvsr_node.src.models.wan_video_dit"]
    pl = sys.modules["flashvsr_node.src.pipelines.flashvsr_tiny_long"]

    try:
        import sageattention  # noqa: F401
        wan.SAGE_ATTN_AVAILABLE = True
        wan.sageattn = sageattention.sageattn
        print("sageattn 已就绪")
    except Exception as e:
        print("sageattn 不可用:", e)
        return

    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 16, 6, 32, 32, generator=g).to(dtype=torch.bfloat16, device="cuda")
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

    y_sparse = one()[0].float().clone()
    ms_sparse = bench_step(one)

    # 切到「无 mask + sageattn」
    orig_mask = wan.generate_draft_block_mask_sage
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask_sage"] = lambda *a, **k: None
    y_sage = one()[0].float().clone()
    ms_sage = bench_step(one)
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask_sage"] = orig_mask

    rel = (y_sparse - y_sage).norm() / y_sparse.norm().clamp_min(1e-6)
    print("\n单步（%d block）: sparse %6.1f ms → sage %6.1f ms   %+.1f%%（%.2fx）"
          % (nb, ms_sparse, ms_sage, 100 * (ms_sage / ms_sparse - 1), ms_sparse / ms_sage))
    print("输出差异: 相对 %.2e  maxdiff %.4f  |y|均值 %.3f" %
          (rel, (y_sparse - y_sage).abs().max(), y_sparse.abs().mean()))


if __name__ == "__main__":
    main()
