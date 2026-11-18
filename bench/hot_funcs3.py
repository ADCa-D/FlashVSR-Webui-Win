# -*- coding: utf-8 -*-
"""热点函数：用 NVTX range 标记函数，再用 torch.profiler（**不开 with_stack**）读区间时间。

为什么不用 with_stack：本机 torch 的 profiler 会递归遍历 cpu_parent 链，
栈一深就指数级变慢（表现为"占显存不动"）。
为什么不用逐调用 CUDA event：嵌套区间 + 队列反压会让和超过总时长（实测 139 ms vs 101.9）。

NVTX range 会被 profiler 记成独立事件，key_averages 直接给出每个区间的
self CUDA / self CPU，而且嵌套关系由 NVTX 自己管，不需要栈。
"""
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import engine as E  # noqa: E402

LABELS = []


def nvtx_range(label, fn):
    def inner(*a, **k):
        torch.cuda.nvtx.range_push(label)
        try:
            return fn(*a, **k)
        finally:
            torch.cuda.nvtx.range_pop()
    return inner


def patch_func(mod, name, label):
    setattr(mod, name, nvtx_range(label, getattr(mod, name)))
    LABELS.append(label)


def patch_method(cls, name, label):
    setattr(cls, name, nvtx_range(label, getattr(cls, name)))
    LABELS.append(label)


def patch_static(cls, name, label):
    setattr(cls, name, staticmethod(nvtx_range(label, getattr(cls, name))))
    LABELS.append(label)


def main():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None,
                      quant=os.environ.get("BENCH_QUANT", "mxfp8"))
    pipe = eng.pipe
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    wan = sys.modules["flashvsr_node.src.models.wan_video_dit"]
    pl = sys.modules["flashvsr_node.src.pipelines.flashvsr_tiny_long"]
    import comfy_kitchen as ck

    G = int(os.environ.get("BENCH_G", "16"))
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(1, 16, 6, 2 * G, 2 * G, generator=g).to(dtype=torch.bfloat16, device="cuda")
    ts = torch.tensor([500.0], dtype=torch.bfloat16, device="cuda")
    dit.reinit_cross_kv(pipe.prompt_emb_posi["context"])
    with torch.no_grad():
        t = dit.time_embedding(wan.sinusoidal_embedding_1d(dit.freq_dim, ts))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    nb = len(dit.blocks)

    def one():
        with torch.no_grad():
            return pl.model_fn_wan_video(
                dit, x, ts, None, LQ_latents=None, is_stream=True,
                pre_cache_k=[None] * nb, pre_cache_v=[None] * nb,
                topk_ratio=22.5, kv_ratio=2.0, cur_process_idx=0,
                t_mod=t_mod, t=t, local_range=9)

    for _ in range(3):
        one()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(5):
        one()
    ev1.record()
    torch.cuda.synchronize()
    base_ms = ev0.elapsed_time(ev1) / 5

    patch_func(wan, "rope_apply", "rope_apply")
    patch_static(wan.WindowPartition3D, "partition", "partition")
    patch_static(wan.WindowPartition3D, "reverse", "reverse")
    patch_method(wan.RMSNorm, "forward", "rmsnorm")
    patch_func(wan, "modulate", "modulate")
    patch_func(wan, "generate_draft_block_mask_sage", "mask")
    patch_func(wan, "sparse_sageattn", "attn_self")
    patch_func(wan, "flash_attention", "attn_all")     # 含 attn_self，用差值拿 cross
    patch_func(ck, "quantize_mxfp8", "quantize")
    patch_func(ck, "scaled_mm_mxfp8", "gemm")
    patch_method(wan.DiTBlock, "forward", "block_total")
    for name in ("generate_draft_block_mask_sage", "sparse_sageattn", "flash_attention"):
        pl.model_fn_wan_video.__globals__[name] = getattr(wan, name)

    refs = {"generate_draft_block_mask_sage": 0, "sparse_sageattn": 0, "flash_attention": 0}
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(3):
            one()
        torch.cuda.synchronize()

    stat = {}
    for ev in prof.key_averages():
        name = ev.key
        if name in LABELS:
            stat[name] = (ev.self_cuda_time_total / 1000.0 / 3,
                          ev.self_cpu_time_total / 1000.0 / 3,
                          ev.count / 3)

    cross = stat.get("attn_all", (0, 0, 0))
    self_a = stat.get("attn_self", (0, 0, 0))
    rows = []
    for k, (gms, cms, cnt) in stat.items():
        if k == "block_total":
            continue
        rows.append((k, gms, cms, cnt))
    # attn_all - attn_self = cross-attn
    rows = [r for r in rows if r[0] != "attn_all"]
    rows.append(("attn_cross(SDPA)", max(0.0, cross[0] - self_a[0]),
                 max(0.0, cross[1] - self_a[1]), max(0.0, cross[2] - self_a[2])))
    rows.sort(key=lambda r: -r[1])

    tot_g = sum(r[1] for r in rows)
    tot_c = sum(r[2] for r in rows)
    bt = stat.get("block_total", (0, 0, 0))

    lines = ["# 热点函数（NVTX 区间 + 算子级计时，无栈、无递归）", "",
             "环境：RTX 5060 / `%s` / DiT 网格 %dx%d（%d token）/ %d block。"
             % (os.environ.get("BENCH_QUANT", "mxfp8"), G, G, 6 * G * G, nb),
             "",
             "基线（未插桩）**%.1f ms/步**；`block_total` 区间实测 %.1f ms GPU / %.1f ms CPU。"
             % (base_ms, bt[0], bt[1]), "",
             "| GPU ms/步 | CPU ms/步 | 次数/步 | 函数 |", "|---|---|---|---|"]
    for k, gms, cms, cnt in rows:
        if gms < 0.01 and cms < 0.05:
            continue
        lines.append("| %.2f | %.2f | %d | `%s` |" % (gms, cms, int(cnt), k))
    lines += ["", "区间合计 GPU %.1f ms / CPU %.1f ms（含嵌套，故小于 block_total）。" % (tot_g, tot_c), ""]

    out = os.path.join(ROOT, "review", "HOT_FUNCS.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
