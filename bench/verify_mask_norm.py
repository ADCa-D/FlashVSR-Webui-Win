# -*- coding: utf-8 -*-
"""两项改动的单元级校验：
  1. 饱和时 ones mask 的 shape 必须与真 mask 一致（否则 attn kernel 崩）
  2. F.rms_norm vs 旧 RMSNorm：同一输入下的 maxdiff / 相对差
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
    wan = sys.modules["flashvsr_node.src.models.wan_video_dit"]
    pl = sys.modules["flashvsr_node.src.pipelines.flashvsr_tiny_long"]

    # ---- 1) mask shape
    cap = {}
    orig = wan.generate_draft_block_mask_sage

    def spy(batch_size, nheads, seqlen, q_w, k_w, topk=10, local_attn_mask=None, sig=None):
        m = orig(batch_size, nheads, seqlen, q_w, k_w, topk=topk,
                 local_attn_mask=local_attn_mask, sig=sig)
        cap.setdefault("fast", m.shape)
        cap.setdefault("dtype", m.dtype)
        wan._SKIP_TRIVIAL_MASK = False
        m2 = orig(batch_size, nheads, seqlen, q_w, k_w, topk=topk,
                  local_attn_mask=local_attn_mask, sig=sig)
        wan._SKIP_TRIVIAL_MASK = True
        cap.setdefault("slow", m2.shape)
        cap.setdefault("diff", int((m != m2).sum()))
        return m

    wan.generate_draft_block_mask_sage = spy
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask_sage"] = spy
    dit = pipe.dit
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 16, 6, 32, 32, generator=g).to(dtype=torch.bfloat16, device="cuda")
    ts = torch.tensor([500.0], dtype=torch.bfloat16, device="cuda")
    dit.reinit_cross_kv(pipe.prompt_emb_posi["context"])
    with torch.no_grad():
        t = dit.time_embedding(wan.sinusoidal_embedding_1d(dit.freq_dim, ts))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        pl.model_fn_wan_video(dit, x, ts, None, LQ_latents=None, is_stream=True,
                              pre_cache_k=[None] * len(dit.blocks),
                              pre_cache_v=[None] * len(dit.blocks),
                              topk_ratio=22.5, kv_ratio=2.0, cur_process_idx=0,
                              t_mod=t_mod, t=t, local_range=9)
    wan.generate_draft_block_mask_sage = orig
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask_sage"] = orig
    print("1) mask：快路径 %s / 慢路径 %s  dtype=%s  不同 cell=%d"
          % (tuple(cap["fast"]), tuple(cap["slow"]), cap["dtype"], cap["diff"]))

    # ---- 2) RMSNorm
    W = 1536
    rn = wan.RMSNorm(W).cuda().float()
    torch.nn.init.normal_(rn.weight, 1.0, 0.02)
    for dt in (torch.bfloat16, torch.float32):
        v = (torch.randn(1, 1024, W, device="cuda") * 3).to(dt)
        with torch.no_grad():
            new = torch.nn.functional.rms_norm(v, (W,), rn.weight.to(dt), rn.eps)
            old = rn.norm(v.float()).to(dt) * rn.weight.to(dt)
        d = (new.float() - old.float()).abs()
        print("2) RMSNorm %s: equal=%s  maxdiff %.5f  相对 %.2e"
              % (str(dt).split(".")[-1], torch.equal(new, old), d.max().item(),
                 (d.norm() / old.float().norm().clamp_min(1e-6)).item()))


if __name__ == "__main__":
    main()
