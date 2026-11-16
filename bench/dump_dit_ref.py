# -*- coding: utf-8 -*-
"""转储 DiT 单次 forward 的参考张量，作为 C++ 端逐层对齐的基准。

固定的形状对应一个真实 tile：latent 16×16（=128 px / VAE 8×），帧 6（流式首窗）。
导出内容：
  x / timestep / context          —— 输入
  t / t_mod                       —— 时间嵌入
  x_patch                         —— patchify 之后
  block00 / block15 / block29     —— 各段输出
  y                               —— 最终输出（含 head + unpatchify）
  mask0 / local0 / q_w0 / k_w0    —— block0 的稀疏块掩码与生成它的池化张量

用法：python bench/dump_dit_ref.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import engine as E  # noqa: E402

OUT = os.path.join(ROOT, "work", "dit_ref")
SEED = 1234


def main():
    os.makedirs(OUT, exist_ok=True)

    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="none")
    pipe = eng.pipe
    # ensure_loaded 会把模型留在 CPU 上（省显存），参考跑起来前先搬上卡
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    wan = sys.modules["flashvsr_node.src.models.wan_video_dit"]

    # ---- 抓住 block 掩码的生成过程（生产路径是稀疏块注意力） ----------------
    cap = {}
    orig_sage = wan.generate_draft_block_mask_sage
    orig_blk = wan.generate_draft_block_mask
    cur = {}

    def patched(kind, orig):
        def fn(batch_size, nheads, seqlen, q_w, k_w, topk=10, local_attn_mask=None):
            m = orig(batch_size, nheads, seqlen, q_w, k_w, topk=topk,
                     local_attn_mask=local_attn_mask)
            if not cap:
                cap["mask"] = m.detach().to(torch.int8).cpu()
                cap["local"] = local_attn_mask.detach().cpu()
                cap["q_w"] = q_w.detach().float().cpu()
                cap["k_w"] = k_w.detach().float().cpu()
                cap["topk"] = topk
                cap["seqlen"] = seqlen
                cap["kind"] = kind
            return m
        return fn

    wan.generate_draft_block_mask_sage = patched("sage", orig_sage)
    wan.generate_draft_block_mask = patched("block_attn", orig_blk)
    # 生产路径从 pipeline 模块里查这两个名字，得两边都换
    pl = sys.modules["flashvsr_node.src.pipelines.flashvsr_tiny_long"]
    frozen = sys.modules["flashvsr_node.src.models.wan_video_dit"]
    for name in ("generate_draft_block_mask_sage", "generate_draft_block_mask", "USE_BLOCK_ATTN"):
        if hasattr(pl, name):
            pass
    fn_mod = pl.model_fn_wan_video.__globals__
    fn_mod["generate_draft_block_mask_sage"] = wan.generate_draft_block_mask_sage
    fn_mod["generate_draft_block_mask"] = wan.generate_draft_block_mask

    # ---- 固定输入 ---------------------------------------------------------
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(1, 16, 6, 16, 16, generator=g).to(dtype=torch.bfloat16, device="cuda")
    ts = torch.tensor([500.0], dtype=torch.bfloat16, device="cuda")
    ctx = pipe.prompt_emb_posi["context"]
    dit.reinit_cross_kv(ctx)

    grabs = {}

    def hook(name):
        def fn(mod, inp, out):
            if name not in grabs:
                v = out[0] if isinstance(out, (tuple, list)) else out
                grabs[name] = v.detach().float().cpu() if torch.is_tensor(v) else v
        return fn

    hs = [
        dit.blocks[0].register_forward_hook(hook("block00")),
        dit.blocks[15].register_forward_hook(hook("block15")),
        dit.blocks[-1].register_forward_hook(hook("block29")),
    ]

    with torch.no_grad():
        # 顺带把 time 嵌入也抓出来（pipeline 里 t/t_mod 是预先算好的）
        t = dit.time_embedding(
            wan.sinusoidal_embedding_1d(dit.freq_dim, ts))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        x_patch, grid = dit.patchify(x)
        # 生产路径是 model_fn_wan_video（block 固定按 3 元组解包 → 必须 is_stream=True）
        y, cache_k, cache_v = pl.model_fn_wan_video(
            dit, x, ts, None, LQ_latents=None, is_stream=True,
            pre_cache_k=[None] * len(dit.blocks), pre_cache_v=[None] * len(dit.blocks),
            topk_ratio=2.0, kv_ratio=3.0, cur_process_idx=0, t_mod=t_mod, t=t,
            local_range=9)

    for h in hs:
        h.remove()
    wan.generate_draft_block_mask_sage = orig_sage

    ref = {
        "seed": SEED,
        "config": {
            "dim": dit.dim, "ffn_dim": dit.blocks[0].ffn_dim,
            "num_heads": dit.blocks[0].num_heads, "num_layers": len(dit.blocks),
            "freq_dim": dit.freq_dim, "patch_size": list(dit.patch_size),
            "eps": 1e-6, "in_dim": x.shape[1], "out_dim": y.shape[1],
            "grid": list(grid), "seq_len": int(x_patch.shape[1]),
            "ctx_len": int(ctx.shape[1]),
        },
        "x": x.detach().float().cpu(),
        "timestep": ts.cpu(),
        "context": ctx.detach().float().cpu(),
        "t": t.detach().float().cpu(),
        "t_mod": t_mod.detach().float().cpu(),
        "x_patch": x_patch.detach().float().cpu(),
        "y": y.detach().float().cpu(),
        **grabs, **cap,
    }
    torch.save(ref, os.path.join(OUT, "ref.pt"))

    cfg = ref["config"]
    print(f"config: {cfg}")
    print(f"x {tuple(ref['x'].shape)}  x_patch {tuple(ref['x_patch'].shape)}  "
          f"y {tuple(ref['y'].shape)}")
    for k in ("t", "t_mod", "x_patch", "block00", "block15", "block29", "y",
              "local", "q_w", "k_w"):
        if k in ref and torch.is_tensor(ref[k]):
            v = ref[k]
            print(f"  {k:9s} {tuple(v.shape)}  mean={v.float().mean():+.5f}")
    m = ref.get("mask")
    if torch.is_tensor(m):
        print(f"  mask      {tuple(m.shape)}  密度={m.float().mean():.4f}  "
              f"topk={ref.get('topk')}  来源={ref.get('kind')}")
    print(f"  cache_k   {tuple(cache_k[0].shape)} (per block, kv_len=3)")
    print(f"saved -> {os.path.join(OUT, 'ref.pt')}")


if __name__ == "__main__":
    main()
