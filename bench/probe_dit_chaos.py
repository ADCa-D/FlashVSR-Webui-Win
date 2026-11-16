# -*- coding: utf-8 -*-
"""DiT 前向对初值的敏感度：只加 1 ULP 的扰动，看最终输出差多少。

如果 y 的 maxdiff 也是几个单位，那 "C++ vs Python 的 y 要 maxdiff<1e-2" 这个验收线
本身就不成立（bf16 30 层前向是混沌系统），必须改用成片画质指标（PSNR/VMAF）。
"""
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
    t = dit.time_embedding(wan.sinusoidal_embedding_1d(dit.freq_dim, ts))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    grabs = {}

    def hook(name):
        def fn(mod, inp, out):
            if name not in grabs:
                v = out[0] if isinstance(out, (tuple, list)) else out
                grabs[name] = v.detach().float().cpu()
        return fn

    hs = [dit.blocks[i].register_forward_hook(hook(f"b{i:02d}")) for i in (0, 15, 29)]

    def run(xx):
        grabs.clear()
        with torch.no_grad():
            y = pl.model_fn_wan_video(
                dit, xx, ts, None, LQ_latents=None, is_stream=True,
                pre_cache_k=[None] * len(dit.blocks), pre_cache_v=[None] * len(dit.blocks),
                topk_ratio=2.0, kv_ratio=3.0, cur_process_idx=0, t_mod=t_mod, t=t)[0]
        return y, dict(grabs)

    y0, g0 = run(x)
    y0 = y0.float().cpu()

    # 加 1 ULP（bf16 的最小可表示变化）到 x
    xb = x.view(torch.int16)
    xp = (xb + 1).view(torch.bfloat16)
    nchanged = (xp != x).float().mean().item()
    print(f"扰动：{nchanged*100:.1f}% 的元素变了 1 ULP，最大相对变化 "
          f"{((xp.float()-x.float()).abs()/x.float().abs().clamp_min(1e-6)).max().item():.2e}")

    y1, g1 = run(xp)
    y1 = y1.float().cpu()

    for h in hs:
        h.remove()

    d = (y0 - y1).abs()
    print(f"最终 y: maxdiff {d.max().item():.4f}  meandiff {d.mean().item():.5f}")
    print(f"        参考量级 |y| mean {y0.abs().mean().item():.4f}  max {y0.abs().max().item():.4f}")
    for k in ("b00", "b15", "b29"):
        if k in g0 and k in g1:
            lb = (g0[k] - g1[k]).abs()
            print(f"  {k}: maxdiff {lb.max().item():.5f}  |out| mean {g0[k].abs().mean().item():.4f}")


if __name__ == "__main__":
    main()
