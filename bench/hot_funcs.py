# -*- coding: utf-8 -*-
"""抓热点函数：按 **Python 函数/行号** 聚合 GPU 与 CPU 时间（不是按 torch 算子名）。

做法：with_stack=True 抓一次 3 步，然后逐事件取栈里最内层的"项目文件"帧
（wan_video_dit.py / engine.py / pipelines/），累加 self_cuda / self_cpu。

输出：review/HOT_FUNCS.md
"""
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import engine as E  # noqa: E402

TARGET_KEYS = ("wan_video_dit.py", "wan_video_vae.py", "flashvsr_tiny_long.py", "engine.py")
STEPS = 3


def pick_frame(ev):
    """取栈里最内层、且属于目标文件的帧。"""
    stack = getattr(ev, "stack", None) or []
    for fr in reversed(stack):
        fn = getattr(fr, "filename", "") or ""
        if any(k in fn for k in TARGET_KEYS):
            return fn, getattr(fr, "lineno", 0), getattr(fr, "name", "?")
    return None


def short(fn):
    for k in TARGET_KEYS:
        if fn.endswith(k):
            return k
    return os.path.basename(fn)


def main():
    sys.setrecursionlimit(200000)   # torch 的 profiler 会用递归遍历 cpu_parent 链
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None,
                      quant=os.environ.get("BENCH_QUANT", "mxfp8"))
    pipe = eng.pipe
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    wan = sys.modules["flashvsr_node.src.models.wan_video_dit"]
    pl = sys.modules["flashvsr_node.src.pipelines.flashvsr_tiny_long"]

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

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                 with_stack=True) as prof:
        for _ in range(STEPS):
            one()
        torch.cuda.synchronize()

    gpu = collections.defaultdict(float)
    cpu = collections.defaultdict(float)
    ops = collections.defaultdict(lambda: collections.Counter())
    for ev in prof.events():
        g_ = getattr(ev, "self_cuda_time_total", 0.0) or 0.0
        c_ = getattr(ev, "self_cpu_time_total", 0.0) or 0.0
        if g_ <= 0 and c_ <= 0:
            continue
        hit = pick_frame(ev)
        key = ("<未归因>", 0, "?") if hit is None else (short(hit[0]), hit[1], hit[2])
        gpu[key] += g_ / 1000.0 / STEPS      # ms/step
        cpu[key] += c_ / 1000.0 / STEPS
        ops[key][ev.key] += 1

    total_cuda = sum(gpu.values())
    total_cpu = sum(cpu.values())
    print("归因到项目文件的 GPU %.1f ms/步，CPU %.1f ms/步" % (total_cuda, total_cpu))

    lines = ["# 热点函数（按 Python 函数/行号聚合）", "",
             "环境：RTX 5060 / mxfp8 / DiT 网格 16×16（1536 token）/ 30 block；"
             "数值为 **单步** 的 self CUDA 与 self CPU（3 步平均）。", "",
             "GPU 合计 %.1f ms/步，CPU 合计 %.1f ms/步。" % (total_cuda, total_cpu), ""]

    for tag, data, total in (("GPU 时间", gpu, total_cuda), ("CPU 时间", cpu, total_cpu)):
        lines += ["## 按 %s 排序（top 20）" % tag, "",
                  "| ms/步 | 占比 | 位置 | 主要算子 |", "|---|---|---|---|"]
        for key, ms in sorted(data.items(), key=lambda kv: -kv[1])[:20]:
            if ms < 0.05:
                continue
            fn, ln, name = key
            top = ", ".join("%s×%d" % (k.split("::")[-1][:28], v)
                            for k, v in ops[key].most_common(3))
            lines.append("| %.2f | %.1f%% | `%s:%s %s` | %s |"
                         % (ms, 100.0 * ms / max(total, 1e-9), fn, ln, name, top))
        lines.append("")

    out = os.path.join(ROOT, "review", "HOT_FUNCS.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("写入", out)
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
