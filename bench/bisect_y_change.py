# -*- coding: utf-8 -*-
"""用 .orig 里的原始实现逐个替换回来，定位输出变化的来源。"""
import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

import engine as E  # noqa: E402

MODEL = r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast\src\models"


def load_orig():
    sys.path.insert(0, r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes")
    pkg = sys.modules.get("fvnode")
    if pkg is None:
        pkg = types.ModuleType("fvnode")
        pkg.__path__ = [r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast"]
        sys.modules["fvnode"] = pkg
    # 相对 import 需要包内名字，所以把 .orig 复制成临时 .py 再导入
    tmp = os.path.join(MODEL, "_orig_ref.py")
    with open(os.path.join(MODEL, "wan_video_dit.py.orig"), "r", encoding="utf-8") as f:
        src = f.read()
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(src)
    try:
        import importlib
        return importlib.import_module("fvnode.src.models._orig_ref")
    finally:
        pass


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
                t_mod=t_mod, t=t, local_range=9)[0].float()
        return y

    y0 = run()
    print("当前代码            y.mean %.5f  absmean %.5f  y[0,0,0,0]=%.5f"
          % (y0.mean(), y0.abs().mean(), y0.flatten()[0]))

    # 1) 原始 mask 函数（丢弃 sig 参数）
    m_new = (wan.generate_draft_block_mask, wan.generate_draft_block_mask_sage)
    _old_dbm = orig.generate_draft_block_mask
    _old_dbs = orig.generate_draft_block_mask_sage
    wan.generate_draft_block_mask = lambda *a, **k: _old_dbm(*a, **{kk: vv for kk, vv in k.items() if kk != "sig"})
    wan.generate_draft_block_mask_sage = lambda *a, **k: _old_dbs(*a, **{kk: vv for kk, vv in k.items() if kk != "sig"})
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask"] = wan.generate_draft_block_mask
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask_sage"] = wan.generate_draft_block_mask_sage
    y = run()
    print("回退 mask 函数      与当前相对差 %.3e   y.mean %.5f" %
          ((y - y0).norm() / y0.norm(), y.mean()))
    wan.generate_draft_block_mask, wan.generate_draft_block_mask_sage = m_new
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask"] = m_new[0]
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask_sage"] = m_new[1]

    # 2) 原始 freqs（每步 CPU cat + H2D）
    f_new = pl.build_step_freqs

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
    y = run()
    print("回退 freqs 缓存     与当前相对差 %.3e   y.mean %.5f" %
          ((y - y0).norm() / y0.norm(), y.mean()))
    pl.build_step_freqs = f_new

    # 3) 原始 sum: 直接比 freqs 张量
    with torch.no_grad():
        a = f_new(dit, 6, 8, 8, 0, torch.device("cuda"))
        b = freqs_old(dit, 6, 8, 8, 0, torch.device("cuda"))
        print("freqs 张量: equal=%s maxdiff=%.3e" %
              (torch.equal(a, b), (a - b).abs().max() if not torch.equal(a, b) else 0.0))

    # 4) 原始 rope / flash_attention / SelfAttention.forward
    for name in ("rope_apply", "flash_attention"):
        setattr(wan, name, getattr(orig, name))
    y = run()
    print("回退 rope+flash_attn 与当前相对差 %.3e   y.mean %.5f" %
          ((y - y0).norm() / y0.norm(), y.mean()))


if __name__ == "__main__":
    main()
