# -*- coding: utf-8 -*-
"""流式多步（f=6 → f=2 → f=2 …）走 CUDA Graph 与 eager 的等价性 + 耗时。

覆盖第二种图形状：第一帧 6 帧、后续 2 帧，且带 KV cache。
用法：python bench/bench_dit_stream.py [steps]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

import engine as E  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def main():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="none")
    pipe = eng.pipe
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    wan = sys.modules["flashvsr_node.src.models.wan_video_dit"]
    pl = sys.modules["flashvsr_node.src.pipelines.flashvsr_tiny_long"]

    g = torch.Generator().manual_seed(7)
    ts = torch.tensor([500.0], dtype=torch.bfloat16, device="cuda")
    dit.reinit_cross_kv(pipe.prompt_emb_posi["context"])
    with torch.no_grad():
        t = dit.time_embedding(wan.sinusoidal_embedding_1d(dit.freq_dim, ts))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    nb = len(dit.blocks)
    lat = [torch.randn(1, 16, 6, 16, 16, generator=g).to(dtype=torch.bfloat16, device="cuda")]
    for _ in range(STEPS - 1):
        lat.append(torch.randn(1, 16, 2, 16, 16, generator=g).to(dtype=torch.bfloat16, device="cuda"))

    def run_stream():
        ks = [None] * nb
        vs = [None] * nb
        outs, times = [], []
        for i, x in enumerate(lat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                y, ks, vs = pl.model_fn_wan_video(
                    dit, x, ts, None, LQ_latents=None, is_stream=True,
                    pre_cache_k=ks, pre_cache_v=vs, topk_ratio=2.0, kv_ratio=3.0,
                    cur_process_idx=i, t_mod=t_mod, t=t, local_range=9)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            outs.append(y.float().clone())
        return outs, times

    mode = "graph"
    y_g, t_g = run_stream()
    y_g2, t_g2 = run_stream()          # 第二次：全部走 replay
    print("graph  每步 ms:", " ".join("%.1f" % v for v in t_g2))
    print("graph  合计 %.0f ms/步" % (sum(t_g2) / len(t_g2)))

    print("--- 切到 eager 对比 ---")
    if getattr(eng, "dit_graph", None) is not None:
        eng.dit_graph.enabled = False
    y_e, t_e = run_stream()
    print("eager  每步 ms:", " ".join("%.1f" % v for v in t_e))
    print("eager  合计 %.0f ms/步" % (sum(t_e) / len(t_e)))

    for i, (a, b) in enumerate(zip(y_g2, y_e)):
        d = (a - b).abs()
        rel = d.norm() / a.norm()
        print("step%d maxdiff %.5f rel %.2e" % (i, d.max(), rel))


if __name__ == "__main__":
    main()
