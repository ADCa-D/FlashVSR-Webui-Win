# -*- coding: utf-8 -*-
"""非确定性来源判定：把 SelfAttention/DiTBlock/mask/rope/flash_attention/freqs
全部换回 .orig 的原始实现，再连跑 3 次。

  · 仍然每次不同 → 与我改动无关，是上游/内核层面固有的（种子也救不了）
  · 变回一致     → 是我的改动引入的
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

import engine as E  # noqa: E402

MODEL = r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast\src\models"


def load_orig():
    import importlib
    import types
    if "fvnode" not in sys.modules:
        pkg = types.ModuleType("fvnode")
        pkg.__path__ = [r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast"]
        sys.modules["fvnode"] = pkg
    tmp = os.path.join(MODEL, "_orig_ref.py")
    with open(os.path.join(MODEL, "wan_video_dit.py.orig"), "r", encoding="utf-8") as f:
        src = f.read()
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(src)
    return importlib.import_module("fvnode.src.models._orig_ref")


def main():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="none")
    pipe = eng.pipe
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    wan = sys.modules["flashvsr_node.src.models.wan_video_dit"]
    pl = sys.modules["flashvsr_node.src.pipelines.flashvsr_tiny_long"]
    orig = load_orig()

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

    def spread(tag, ys):
        base = ys[0]
        rels = [float((y - base).norm() / base.norm()) for y in ys[1:]]
        print("%-14s y.mean 首轮 %.6f ｜ 相对差 %s ｜ %s"
              % (tag, base.mean(), ["%.1e" % r for r in rels],
                 "确定" if max(rels) == 0 else "**非确定**"))

    spread("当前代码", [run() for _ in range(3)])

    # ---- 全部退回 .orig ----
    wan.rope_apply = orig.rope_apply
    wan.flash_attention = orig.flash_attention
    wan.SelfAttention.forward = orig.SelfAttention.forward
    wan.DiTBlock.forward = orig.DiTBlock.forward
    _o_dbm, _o_dbs = orig.generate_draft_block_mask, orig.generate_draft_block_mask_sage
    wan.generate_draft_block_mask = lambda *a, **k: _o_dbm(*a, **{kk: vv for kk, vv in k.items() if kk != "sig"})
    wan.generate_draft_block_mask_sage = lambda *a, **k: _o_dbs(*a, **{kk: vv for kk, vv in k.items() if kk != "sig"})
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask"] = wan.generate_draft_block_mask
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask_sage"] = wan.generate_draft_block_mask_sage

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

    pl.build_step_freqs = freqs_old
    spread("全部退回 .orig", [run() for _ in range(3)])

    # ---- 再开确定性算法试试 ----
    torch.use_deterministic_algorithms(True, warn_only=True)
    spread("+deterministic", [run() for _ in range(3)])


if __name__ == "__main__":
    main()
