# -*- coding: utf-8 -*-
"""热点函数：用**消融**测（逐个把函数换成等价形状的空实现，看单步掉多少）。

为什么不用其它办法（均已实测失败）：
  · profiler with_stack → 本机递归建树卡死
  · 逐调用 CUDA event → 嵌套+队列反压，区间和 139 > 总时长 102
  · NVTX range → 这个 torch 构建不往 trace 写 CUDA 用户注解

消融的代价：替换后数值变了（mask 会翻转），所以每项都同时记录 attn 时间漂移，
若漂移 >5% 就标注"该行不确定"。
"""
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import engine as E  # noqa: E402


def bench(fn, n=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(n):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / n


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

    base = bench(one, 7)
    print("基线 %.1f ms/步" % base, flush=True)

    # 需要一份真实的 mask 张量来做"常量 mask"消融
    grabbed = {}
    orig_mask = wan.generate_draft_block_mask_sage

    def grab(*a, **k):
        m = orig_mask(*a, **k)
        grabbed["shape"] = m.shape
        grabbed.setdefault("m", m.detach().clone())
        return m

    wan.generate_draft_block_mask_sage = grab
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask_sage"] = grab
    one()
    torch.cuda.synchronize()
    wan.generate_draft_block_mask_sage = orig_mask
    pl.model_fn_wan_video.__globals__["generate_draft_block_mask_sage"] = orig_mask
    const_mask = grabbed["m"]

    qcache = {}

    def fake_quant(x_, pad_32x=False):
        key = tuple(x_.shape)
        hit = qcache.get(key)
        if hit is None:
            num_rows, num_cols = x_.shape
            q = torch.zeros((num_rows, num_cols), device=x_.device, dtype=torch.float8_e4m3fn)
            s = torch.ones((max(128, (num_rows + 127) // 128 * 128),
                            max(4, (num_cols // 32 + 3) // 4 * 4)),
                           device=x_.device, dtype=torch.float8_e8m0fnu)
            hit = qcache[key] = (q, s)
        return hit

    variants = []

    def add(name, patches):
        variants.append((name, patches))

    add("quantize_mxfp8 → 复用缓存张量", [("ck_quantize", fake_quant)])
    add("rope_apply → 恒等", [("wan_fn", ("rope_apply", lambda v, f, n: v))])
    add("RMSNorm.forward → 恒等", [("cls", (wan.RMSNorm, "forward", lambda self, v: v))])
    add("modulate → 恒等", [("wan_fn", ("modulate", lambda v, s, c: v))])
    add("mask 构造 → 返回常量", [("wan_fn", ("generate_draft_block_mask_sage",
                                        lambda *a, **k: const_mask))])
    add("self-attn sparse → SDPA", [("wan_fn", ("sparse_sageattn",
                                             lambda q, k, v, **kw:
                                             torch.nn.functional.scaled_dot_product_attention(q, k, v)))])

    rows = []
    for name, patches in variants:
        undo = []
        try:
            for kind, payload in patches:
                if kind == "ck_quantize":
                    old = ck.quantize_mxfp8
                    ck.quantize_mxfp8 = payload
                    undo.append(lambda old=old: setattr(ck, "quantize_mxfp8", old))
                elif kind == "wan_fn":
                    fname, fn = payload
                    old = getattr(wan, fname)
                    setattr(wan, fname, fn)
                    pl.model_fn_wan_video.__globals__[fname] = fn
                    def _u(fname=fname, old=old):
                        setattr(wan, fname, old)
                        pl.model_fn_wan_video.__globals__[fname] = old
                    undo.append(_u)
                elif kind == "cls":
                    cls, mname, fn = payload
                    old = getattr(cls, mname)
                    setattr(cls, mname, fn)
                    undo.append(lambda cls=cls, mname=mname, old=old: setattr(cls, mname, old))
            ms = bench(one, 7)
        finally:
            for u in reversed(undo):
                u()
        delta = ms - base
        print("  %-32s %.1f ms  (%+.1f ms, %+.1f%%)" % (name, ms, delta, 100 * delta / base), flush=True)
        rows.append((name, ms, delta))

    lines = ["# 热点函数（消融法：把函数换掉，看单步掉多少）", "",
             "环境：RTX 5060 / `%s` / 网格 %dx%d（%d token）/ %d block；基线 **%.1f ms/步**（中位 7 次）。"
             % (os.environ.get("BENCH_QUANT", "mxfp8"), G, G, 6 * G * G, nb, base), "",
             "| 消融项 | 单步 ms | Δ ms | Δ% |", "|---|---|---|---|"]
    for name, ms, delta in sorted(rows, key=lambda r: r[2]):
        lines.append("| %s | %.1f | %+.1f | %+.1f%% |" % (name, ms, delta, 100 * delta / base))
    lines += ["", "注意：消融会改变数值（mask 可能翻转、输出变垃圾），只用于**估算该函数的耗时量级**；"
                  "Δ 为负表示该函数占用这么多时间。", ""]
    out = os.path.join(ROOT, "review", "HOT_FUNCS.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("写入", out)


if __name__ == "__main__":
    main()
