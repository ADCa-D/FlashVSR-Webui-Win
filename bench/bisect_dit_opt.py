# -*- coding: utf-8 -*-
"""逐项回退优化，定位输出变化的来源。"""
import os
import sys

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
    with torch.no_grad():
        t = dit.time_embedding(wan.sinusoidal_embedding_1d(dit.freq_dim, ts))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    nb = len(dit.blocks)

    def run():
        with torch.no_grad():
            return pl.model_fn_wan_video(
                dit, x, ts, None, LQ_latents=None, is_stream=True,
                pre_cache_k=[None] * nb, pre_cache_v=[None] * nb,
                topk_ratio=2.0, kv_ratio=3.0, cur_process_idx=0,
                t_mod=t_mod, t=t, local_range=9)[0].float()

    def cmp(tag, a, b):
        print("%-28s maxdiff %.5f  mean|d| %.6f  rel %.3e" % (
            tag, (a - b).abs().max(), (a - b).abs().mean(),
            (a - b).norm() / a.norm()))

    y_new = run()
    print("new y stats: mean %.4f absmean %.4f" % (y_new.mean(), y_new.abs().mean()))

    # 1) 回退 rope
    def rope_old(x, freqs, num_heads):
        x = x.view(x.shape[0], x.shape[1], num_heads, -1)
        o = torch.view_as_complex(x.to(torch.float64).reshape(
            x.shape[0], x.shape[1], x.shape[2], -1, 2))
        return torch.view_as_real(o * freqs).flatten(2).to(x.dtype)
    rope_new = wan.rope_apply
    wan.rope_apply = rope_old
    y = run()
    cmp("回退 rope", y_new, y)
    print("   old-rope y stats: mean %.4f absmean %.4f" % (y.mean(), y.abs().mean()))
    wan.rope_apply = rope_new

    # 2) 回退 flash_attention 的常量张量（本质无影响，验证用）
    # 3) 回退 mask 构建
    def mask_old(batch_size, nheads, seqlen, q_w, k_w, topk=10, local_attn_mask=None):
        avgpool_q = torch.mean(q_w, dim=1)
        avgpool_q = wan.rearrange(avgpool_q, 's (h d) -> s h d', h=nheads)
        q_heads = avgpool_q.permute(1, 0, 2)
        D = avgpool_q.shape[-1]
        k_w_split = k_w.view(k_w.shape[0], 2, 64, k_w.shape[2])
        avgpool_k_split = torch.mean(k_w_split, dim=2)
        avgpool_k_refined = wan.rearrange(avgpool_k_split, 's two d -> (s two) d', two=2)
        avgpool_k_refined = wan.rearrange(avgpool_k_refined, 's (h d) -> s h d', h=nheads)
        k_heads_doubled = avgpool_k_refined.permute(1, 0, 2)
        k1, k2 = torch.chunk(k_heads_doubled, 2, dim=1)
        s1 = torch.einsum("hld,hmd->hlm", q_heads, k1) / wan.math.sqrt(D)
        s2 = torch.einsum("hld,hmd->hlm", q_heads, k2) / wan.math.sqrt(D)
        scores = torch.cat([s1, s2], dim=-1)
        repeat_head = scores.shape[0]
        repeat_len = scores.shape[1] // local_attn_mask.shape[0]
        repeat_num = (scores.shape[2] // 2) // local_attn_mask.shape[1]
        lm = local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
        lm = wan.rearrange(lm, 'x a y b -> (x a) (y b)')
        lm = lm.repeat_interleave(2, dim=1)
        lm = lm.unsqueeze(0).repeat(repeat_head, 1, 1)
        lm = lm.to(torch.float32)
        lm = lm.masked_fill(lm == False, -float('inf'))
        lm = lm.masked_fill(lm == True, 0)
        scores = scores + lm
        attn_map = torch.softmax(scores, dim=-1)
        attn_map = wan.rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen)
        loop_num, s1_, s2_ = attn_map.shape
        flat = attn_map.reshape(loop_num, -1)
        apply_topk = min(flat.shape[1] - 1, topk)
        th = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1].unsqueeze(1)
        mask_new = (flat > th).reshape(loop_num, s1_, s2_)
        mask_new = wan.rearrange(mask_new, '(h it) s1 s2 -> h (it s1) s2', it=seqlen)
        return mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    mask_new_fn = wan.generate_draft_block_mask_sage
    wan.generate_draft_block_mask_sage = mask_old
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask_sage"] = mask_old
    y = run()
    cmp("回退 mask 构建", y_new, y)
    print("   old-mask y stats: mean %.4f absmean %.4f" % (y.mean(), y.abs().mean()))
    wan.generate_draft_block_mask_sage = mask_new_fn
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask_sage"] = mask_new_fn


if __name__ == "__main__":
    main()
