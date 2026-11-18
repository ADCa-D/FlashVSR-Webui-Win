# -*- coding: utf-8 -*-
"""sprase/topk 在什么条件下才真正生效？

公式（engine.py:967 + pipeline）：
    topk_ratio = sparse * 768*1280 / (tile*scale)^2
    topk       = int(square_num * topk_ratio) - 1 ,  square_num = (2*h*w/128)^2
而 mask 每行只能有 `min(列数-1, topk)` 个 True —— 列数随 tile/kv 增长，预算随 1/tile² 下降。

这里直接测（cols 从返回的 mask 反推，不靠推导）：
  g        = DiT 网格边长（tile=128 → g=16），x 空间尺寸 = 2g
  cols     = mask 的最后一维
  预算      = 传进 mask 的 topk
  生效 k    = min(cols-1, 预算)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

import engine as E  # noqa: E402

REF = 768 * 1280
SPARSE = float(os.environ.get("BENCH_SPARSE", "1.5"))
SCALE = 2


def main():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="none")
    pipe = eng.pipe
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    wan = sys.modules["flashvsr_node.src.models.wan_video_dit"]

    rec = {}
    orig = wan.generate_draft_block_mask_sage

    def probe(batch_size, nheads, seqlen, q_w, k_w, topk=10, local_attn_mask=None, sig=None):
        m = orig(batch_size, nheads, seqlen, q_w, k_w, topk=topk,
                 local_attn_mask=local_attn_mask, sig=sig)
        rec["cols"] = int(m.shape[-1])
        rec["rows"] = int(m.shape[-2])
        rec["budget"] = int(topk)
        rec["k"] = int(min(m.shape[-1] - 1, topk))
        return m

    wan.generate_draft_block_mask_sage = probe
    pl = sys.modules["flashvsr_node.src.pipelines.flashvsr_tiny_long"]
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask_sage"] = probe

    ts = torch.tensor([500.0], dtype=torch.bfloat16, device="cuda")
    dit.reinit_cross_kv(pipe.prompt_emb_posi["context"])
    with torch.no_grad():
        t = dit.time_embedding(wan.sinusoidal_embedding_1d(dit.freq_dim, ts))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    nb = len(dit.blocks)

    print("sparse=%.2f scale=%d   （DiT 网格 g；tile = g*8 像素）" % (SPARSE, SCALE))
    print("%4s %5s %6s %8s %7s %7s %6s %7s %s" %
          ("g", "tile", "kv", "预算topk", "列数", "生效k", "覆盖率", "饱和?", "说明"))
    for g in (8, 16, 24, 32):   # 窗口是 (2,8,8)，g 必须是 8 的倍数
        tile = g * 8
        for kv in (2.0, 3.0, 4.0):
            rec.clear()
            x = torch.zeros(1, 16, 6, 2 * g, 2 * g, dtype=torch.bfloat16, device="cuda")
            tr = SPARSE * REF / (tile * SCALE) ** 2
            with torch.no_grad():
                try:
                    pl.model_fn_wan_video(
                        dit, x, ts, None, LQ_latents=None, is_stream=True,
                        pre_cache_k=[None] * nb, pre_cache_v=[None] * nb,
                        topk_ratio=tr, kv_ratio=kv, cur_process_idx=0,
                        t_mod=t_mod, t=t, local_range=9)
                except Exception as e:
                    print("%4d %5d %6.1f %s" % (g, tile, kv, type(e).__name__ + ": " + str(e)[:44]))
                    continue
            cols, bud, k = rec["cols"], rec["budget"], rec["k"]
            print("%4d %5d %6.1f %8d %7d %7d %5.1f%% %7s %s"
                  % (g, tile, kv, bud, cols, k, 100.0 * k / cols,
                     "是" if bud >= cols - 1 else "否",
                     "sparse 退化" if bud >= cols - 1 else "sparse 生效"))
    wan.generate_draft_block_mask_sage = orig
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask_sage"] = orig


if __name__ == "__main__":
    main()
