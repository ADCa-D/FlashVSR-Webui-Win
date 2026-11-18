# -*- coding: utf-8 -*-
"""热点函数计时（不用 torch.profiler，避免它递归建树的坑）。

做法：把**叶子函数**包一层，每次调用记一对 CUDA event + 墙钟；跑完 3 步再统一读。
叶子 = 不与其它被计时函数重叠的调用点，避免重复计算：

  rope_apply / WindowPartition3D.partition+reverse / RMSNorm.forward /
  modulate / generate_draft_block_mask_sage / sparse_sageattn / flash_attention /
  quantize_mxfp8 / scaled_mm_mxfp8 / build_step_freqs

输出：review/HOT_FUNCS.md（每步 ms、调用次数、占比）
"""
import collections
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import engine as E  # noqa: E402

ACC = {}
PAIRS = []
STEPS = 3


class Leaf:
    def __init__(self):
        self.cpu = 0.0
        self.n = 0


def rec(label):
    return ACC.setdefault(label, Leaf())


def timed(label, fn):
    def inner(*a, **k):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        e0.record()
        r = fn(*a, **k)
        e1.record()
        rec(label).cpu += (time.perf_counter() - t0) * 1000.0
        rec(label).n += 1
        PAIRS.append((label, e0, e1))
        return r
    return inner


def patch_func(mod, name, label):
    orig = getattr(mod, name)
    setattr(mod, name, timed(label, orig))
    return orig


def patch_method(cls, name, label):
    orig = getattr(cls, name)
    setattr(cls, name, timed(label, orig))
    return orig


def patch_static(cls, name, label):
    orig = getattr(cls, name)
    setattr(cls, name, staticmethod(timed(label, orig)))
    return orig


def main():
    print("先跑一遍不计时，作为基线")
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
    print("基线 %.1f ms/步（无计时开销）" % base_ms)

    # ---- 装计时
    patch_func(wan, "rope_apply", "rope_apply")
    patch_static(wan.WindowPartition3D, "partition", "WindowPartition.partition")
    patch_static(wan.WindowPartition3D, "reverse", "WindowPartition.reverse")
    patch_method(wan.RMSNorm, "forward", "RMSNorm.forward")
    patch_func(wan, "modulate", "modulate")
    patch_func(wan, "build_step_freqs", "build_step_freqs")
    orig_sage = patch_func(wan, "generate_draft_block_mask_sage", "mask(topk/softmax)")
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask_sage"] = wan.generate_draft_block_mask_sage
    patch_func(wan, "sparse_sageattn", "sparse_sageattn(self-attn)")
    pl.model_fn_wan_video.__globals__["sparse_sageattn"] = wan.sparse_sageattn
    patch_func(wan, "flash_attention", "flash_attention(self+cross)")
    pl.model_fn_wan_video.__globals__["flash_attention"] = wan.flash_attention
    patch_func(ck, "quantize_mxfp8", "quantize_mxfp8")
    patch_func(ck, "scaled_mm_mxfp8", "scaled_mm_mxfp8")
    patch_method(wan.DiTBlock, "forward", "DiTBlock.forward(含全部子调用)")

    with torch.no_grad():
        for _ in range(STEPS):
            one()
        torch.cuda.synchronize()

    # ---- 汇总
    gpu = collections.defaultdict(float)
    for label, a, b in PAIRS:
        gpu[label] += a.elapsed_time(b)

    rows = []
    for label, lf in ACC.items():
        rows.append((label, gpu[label] / STEPS, lf.cpu / STEPS, lf.n / STEPS))
    rows.sort(key=lambda r: -r[1])
    tot_gpu = sum(r[1] for r in rows if "含全部子" not in r[0])
    tot_cpu = sum(r[2] for r in rows if "含全部子" not in r[0])

    lines = ["# 热点函数（叶子级，CUDA event 计时）", "",
             "环境：RTX 5060 / `%s` / DiT 网格 %dx%d（%d token）/ %d block / 单步。"
             % (os.environ.get("BENCH_QUANT", "mxfp8"), G, G, 6 * G * G, nb), "",
             "基线（不计时）**%.1f ms/步**；下表为叶子函数自身的 GPU/CPU 时间。" % base_ms, "",
             "| GPU ms/步 | CPU ms/步 | 次数/步 | 函数 |", "|---|---|---|---|"]
    for label, gms, cms, n in rows:
        if gms < 0.02 and cms < 0.05:
            continue
        lines.append("| %.2f | %.2f | %d | %s |" % (gms, cms, int(n), label))
    lines += ["",
              "叶子合计：GPU %.1f ms，CPU %.1f ms；基线 %.1f ms（差值为未归因：GEMM 之外的"
              "elementwise/copy/cat 与 launch 间隙）。" % (tot_gpu, tot_cpu, base_ms), ""]

    out = os.path.join(ROOT, "review", "HOT_FUNCS.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
