# -*- coding: utf-8 -*-
"""在真实模型输入下对比 rope 新旧实现（block0 的 q/k）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

import engine as E  # noqa: E402


def rope_old(x, freqs, num_heads):
    x = x.view(x.shape[0], x.shape[1], num_heads, -1)
    o = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    return torch.view_as_real(o * freqs).flatten(2).to(x.dtype)


def main():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="none")
    pipe = eng.pipe
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    wan = sys.modules["flashvsr_node.src.models.wan_video_dit"]

    g = torch.Generator().manual_seed(1234)
    x = torch.randn(1, 16, 6, 16, 16, generator=g).to(dtype=torch.bfloat16, device="cuda")
    ts = torch.tensor([500.0], dtype=torch.bfloat16, device="cuda")
    dit.reinit_cross_kv(pipe.prompt_emb_posi["context"])
    with torch.no_grad():
        t = dit.time_embedding(wan.sinusoidal_embedding_1d(dit.freq_dim, ts))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        x_patch, grid = dit.patchify(x)
        blk = dit.blocks[0]
        shift_msa, scale_msa = (blk.modulation.to(dtype=t_mod.dtype) + t_mod).chunk(6, dim=1)[:2]
        inp = wan.modulate(blk.norm1(x_patch), shift_msa, scale_msa)
        q_pre = blk.self_attn.norm_q(blk.self_attn.q(inp))
        k_pre = blk.self_attn.norm_k(blk.self_attn.k(inp))
        f, h, w = grid
        freqs = torch.cat([
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        print("grid", grid, "freqs", tuple(freqs.shape), freqs.dtype)
        print("q_pre", tuple(q_pre.shape), q_pre.dtype, "contig", q_pre.is_contiguous())
        for tag, v in (("q", q_pre), ("k", k_pre)):
            a = rope_old(v, freqs, blk.self_attn.num_heads)
            b = wan.rope_apply(v, freqs, blk.self_attn.num_heads)
            print("%s maxdiff %.3e  nelem %d  exact %s" % (
                tag, (a.float() - b.float()).abs().max(), int((a != b).sum()), torch.equal(a, b)))
        cos, sin = wan._rope_cos_sin(freqs)
        print("cos", tuple(cos.shape), cos.dtype, "sin", tuple(sin.shape))
        print("cos[0,:4]", cos[0, :4].tolist())
        print("freqs.real[0,0,:4]", freqs.real[0, 0, :4].tolist())


if __name__ == "__main__":
    main()
