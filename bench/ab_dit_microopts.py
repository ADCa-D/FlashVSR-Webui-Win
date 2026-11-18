# -*- coding: utf-8 -*-
"""真实配置（topk_ratio=22.5 / kv=2.0）下，逐项回退优化，看每项值多少 ms。

回退项（用等价实现覆盖 wan 模块里的函数）：
  1. mask 展开（去掉缓存）
  2. flash_attention 里无条件创建 cu_seqlens/head_mask_type
  3. build_step_freqs（每步 CPU cat + H2D）
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

import engine as E  # noqa: E402

TOPKS = [float(x) for x in (os.environ.get("BENCH_TOPKS", "2.0,22.5").split(","))]
KV = float(os.environ.get("BENCH_KV", "2.0"))


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

    # ---- 备份现实现
    fast_mask = wan.generate_draft_block_mask_sage
    fast_freqs = pl.build_step_freqs
    fast_attn = wan.flash_attention

    # ---- 回退版 mask（原样照抄旧代码）
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
        rh = scores.shape[0]
        rl = scores.shape[1] // local_attn_mask.shape[0]
        rn = (scores.shape[2] // 2) // local_attn_mask.shape[1]
        lm = local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(rl, 1, rn, 1)
        lm = wan.rearrange(lm, 'x a y b -> (x a) (y b)')
        lm = lm.repeat_interleave(2, dim=1).unsqueeze(0).repeat(rh, 1, 1)
        lm = lm.to(torch.float32)
        lm = lm.masked_fill(lm == False, -float('inf')).masked_fill(lm == True, 0)
        attn_map = torch.softmax(scores + lm, dim=-1)
        attn_map = wan.rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen)
        ln, s1_, s2_ = attn_map.shape
        flat = attn_map.reshape(ln, -1)
        k = min(flat.shape[1] - 1, topk)
        th = torch.topk(flat, k=k + 1, dim=1, largest=True).values[:, -1].unsqueeze(1)
        mask_new = (flat > th).reshape(ln, s1_, s2_)
        mask_new = wan.rearrange(mask_new, '(h it) s1 s2 -> h (it s1) s2', it=seqlen)
        return mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # ---- 回退版 freqs（CPU 上 cat + to(device)）
    def freqs_old(dit_, f_, h_, w_, idx=0, device=None):
        if idx == 0:
            return torch.cat([
                dit_.freqs[0][:f_].view(f_, 1, 1, -1).expand(f_, h_, w_, -1),
                dit_.freqs[1][:h_].view(1, h_, 1, -1).expand(f_, h_, w_, -1),
                dit_.freqs[2][:w_].view(1, 1, w_, -1).expand(f_, h_, w_, -1),
            ], dim=-1).reshape(f_ * h_ * w_, 1, -1).to(device)

        return torch.cat([
            dit_.freqs[0][4 + idx * 2:4 + idx * 2 + f_].view(f_, 1, 1, -1).expand(f_, h_, w_, -1),
            dit_.freqs[1][:h_].view(1, h_, 1, -1).expand(f_, h_, w_, -1),
            dit_.freqs[2][:w_].view(1, 1, w_, -1).expand(f_, h_, w_, -1),
        ], dim=-1).reshape(f_ * h_ * w_, 1, -1).to(device)

    # ---- 回退版 flash_attention（只在 sage 路径前无条件造 3 个常量张量）
    def attn_old(q, k, v, num_heads, compatibility_mode=False, attention_mask=None, return_KV=False):
        if attention_mask is not None:
            q2 = wan.rearrange(q, "b s (n d) -> b n s d", n=num_heads)
            k2 = wan.rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v2 = wan.rearrange(v, "b s (n d) -> b n s d", n=num_heads)
            cu_q = torch.tensor([0, q.shape[1]], device=q.device, dtype=torch.int32)
            cu_k = torch.tensor([0, k.shape[1]], device=k.device, dtype=torch.int32)
            hm = torch.tensor([1] * num_heads, device=q.device, dtype=torch.int32)
            x = wan.sparse_sageattn(q2, k2, v2, mask_id=attention_mask.to(torch.int8),
                                    is_causal=False, tensor_layout="HND")
            return wan.rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        return fast_attn(q, k, v, num_heads, compatibility_mode, attention_mask, return_KV)

    def bench(tag, n=12):
        with torch.no_grad():
            for _ in range(3):
                one()
            torch.cuda.synchronize()
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            ev0.record()
            for _ in range(n):
                one()
            ev1.record()
            torch.cuda.synchronize()
        print("  %-28s %6.1f ms/步" % (tag, ev0.elapsed_time(ev1) / n), flush=True)

    for tk in TOPKS:
        print(f"topk_ratio={tk} kv={KV}", flush=True)

        def one():
            with torch.no_grad():
                return pl.model_fn_wan_video(
                    dit, x, ts, None, LQ_latents=None, is_stream=True,
                    pre_cache_k=[None] * nb, pre_cache_v=[None] * nb,
                    topk_ratio=tk, kv_ratio=KV, cur_process_idx=0,
                    t_mod=t_mod, t=t, local_range=9)

        bench("全部优化")
        wan.generate_draft_block_mask_sage = mask_old
        pl.model_fn_wan_video.__globals__["generate_draft_block_mask_sage"] = mask_old
        bench("回退 mask 缓存")
        wan.generate_draft_block_mask_sage = fast_mask
        pl.model_fn_wan_video.__globals__["generate_draft_block_mask_sage"] = fast_mask
        pl.build_step_freqs = freqs_old
        bench("回退 freqs 缓存")
        pl.build_step_freqs = fast_freqs
        wan.flash_attention = attn_old
        bench("回退 flash_attention")
        wan.flash_attention = fast_attn


if __name__ == "__main__":
    main()
