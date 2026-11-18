# -*- coding: utf-8 -*-
"""DiT 单步（流式，40 block）耗时 + 输出指纹。

用途：
  1. 改代码前后对比 ms/步（同输入、同 topk/kv，关掉随机性）
  2. 输出 y 的 maxdiff / PSNR，确认优化没有改变数值

用法：
  python bench/bench_dit_step.py            # 默认 10 次
  python bench/bench_dit_step.py 20
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import engine as E  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def main():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant=os.environ.get("BENCH_QUANT", "none"))
    pipe = eng.pipe
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    wan = sys.modules["flashvsr_node.src.models.wan_video_dit"]
    pl = sys.modules["flashvsr_node.src.pipelines.flashvsr_tiny_long"]

    g = torch.Generator().manual_seed(1234)
    G = int(os.environ.get("BENCH_G", "8"))
    x = torch.randn(1, 16, 6, 2 * G, 2 * G, generator=g).to(dtype=torch.bfloat16, device="cuda")
    ts = torch.tensor([500.0], dtype=torch.bfloat16, device="cuda")
    dit.reinit_cross_kv(pipe.prompt_emb_posi["context"])

    with torch.no_grad():
        t = dit.time_embedding(wan.sinusoidal_embedding_1d(dit.freq_dim, ts))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    nb = len(dit.blocks)
    tk = float(os.environ.get("BENCH_TOPK", "2.0"))
    kv = float(os.environ.get("BENCH_KV", "3.0"))

    def one():
        with torch.no_grad():
            return pl.model_fn_wan_video(
                dit, x, ts, None, LQ_latents=None, is_stream=True,
                pre_cache_k=[None] * nb, pre_cache_v=[None] * nb,
                topk_ratio=tk, kv_ratio=kv, cur_process_idx=0,
                t_mod=t_mod, t=t, local_range=9)

    # 预热
    y0 = one()[0]
    torch.cuda.synchronize()

    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(N):
        y = one()[0]
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / N

    d = (y.float() - y0.float()).abs().max().item()
    print(f"blocks={nb}  {ms:.1f} ms/步  ({1000/ms:.2f} 步/s)  y mean {y.float().mean():.4f} "
          f"max {y.float().abs().max():.4f}")
    print(f"fingerprint: y[0,0,0,0,0]={y.flatten()[0].item():.6f}  y.abs().mean={y.float().abs().mean():.6f}")
    torch.save({"y": y.float().cpu(), "x": x.cpu(), "ts": ts.cpu()}, os.path.join(ROOT, "work", "dit_step_out.pt"))
    print("saved work/dit_step_out.pt")


if __name__ == "__main__":
    main()
