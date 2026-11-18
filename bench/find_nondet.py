# -*- coding: utf-8 -*-
"""定位非确定性：同一输入跑两遍，逐层比较中间张量，看第一个 diverge 的算子。"""
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

    g = torch.Generator().manual_seed(1234)
    x = torch.randn(1, 16, 6, 16, 16, generator=g).to(dtype=torch.bfloat16, device="cuda")
    ts = torch.tensor([500.0], dtype=torch.bfloat16, device="cuda")
    dit.reinit_cross_kv(pipe.prompt_emb_posi["context"])
    with torch.no_grad():
        t = dit.time_embedding(wan.sinusoidal_embedding_1d(dit.freq_dim, ts))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    nb = len(dit.blocks)

    # 抓 block0 的中间量
    caps = {}
    orig_rope = wan.rope_apply
    orig_mask = wan.generate_draft_block_mask_sage
    first = {"rope": 0, "mask": 0}

    def rope_spy(v, freqs, n):
        out = orig_rope(v, freqs, n)
        if first["rope"] < 2:
            caps.setdefault("rope", []).append(out.detach().float().clone())
            first["rope"] += 1
        return out

    def mask_spy(batch_size, nheads, seqlen, q_w, k_w, topk=10, local_attn_mask=None, sig=None):
        m = orig_mask(batch_size, nheads, seqlen, q_w, k_w, topk=topk,
                      local_attn_mask=local_attn_mask, sig=sig)
        if first["mask"] < 2:
            caps.setdefault("mask", []).append(m.detach().clone())
            caps.setdefault("q_w", []).append(q_w.detach().float().clone())
            caps.setdefault("k_w", []).append(k_w.detach().float().clone())
            first["mask"] += 1
        return m

    wan.rope_apply = rope_spy
    wan.generate_draft_block_mask_sage = mask_spy
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask_sage"] = mask_spy

    outs = []
    for i in range(2):
        first["rope"] = 0
        first["mask"] = 0
        with torch.no_grad():
            y = pl.model_fn_wan_video(
                dit, x, ts, None, LQ_latents=None, is_stream=True,
                pre_cache_k=[None] * nb, pre_cache_v=[None] * nb,
                topk_ratio=2.0, kv_ratio=3.0, cur_process_idx=0,
                t_mod=t_mod, t=t, local_range=9)[0].float().clone()
        torch.cuda.synchronize()
        outs.append(y)

    wan.rope_apply = orig_rope
    wan.generate_draft_block_mask_sage = orig_mask
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask_sage"] = orig_mask

    def cmp(tag, a, b):
        same = torch.equal(a, b)
        d = (a - b).abs().max().item()
        print("%-18s %s  maxdiff %.6f  nelem_diff %d/%d"
              % (tag, "一致" if same else "**不同**", d if not same else 0.0,
                 int((a != b).sum()) if not same else 0, a.numel()))

    print("== block0 输入侧 ==")
    for key in ("q_w", "k_w", "rope", "mask"):
        if key in caps and len(caps[key]) >= 2:
            # rope 每步调 q/k 两次，这里比对同一 index
            cmp(key, caps[key][0], caps[key][1])
    print("== 输出 ==")
    cmp("y", outs[0], outs[1])


if __name__ == "__main__":
    main()
