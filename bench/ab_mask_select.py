# -*- coding: utf-8 -*-
"""隔离实验：只改「每行保留哪些块」的选法，看差异是统计噪声还是系统性偏移。

对照：
  旧  mask = flat > 第(k+1)大值   ← 行内 True 的个数不确定，并列值时可能全 False
  新  mask = topk(k).indices + scatter_   ← 严格 k 个，行内一定有 True

两件事分开量：
  A. 掩码层：在**真实潜变量**（跑一遍 e2e）上统计两种选法的差异
     行数 / 每行 True 个数 / 空行数 / 差异 cell 占比
  B. 输出层：同一段片子跑两遍（只切这个开关）比逐帧 PSNR、NaN、锐度

用法：
  python bench/ab_mask_select.py [frames]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import engine as E  # noqa: E402

SRC = os.environ.get("FLASHVSR_AB_SRC", r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_long_544.mp4")
WORK = os.path.join(ROOT, "work")
MAXF = int(sys.argv[1]) if len(sys.argv) > 1 else 33

STATS = {"rows": 0, "cells": 0, "diff_cells": 0, "empty_old": 0,
         "count_old_hist": {}, "count_new_neq_k": 0, "k_hist": {}}


def install_probe(wan, mode):
    """把选块函数换成一个同时算两种、并按 mode 返回的版本，顺便累计统计。"""
    orig = wan._select_topk_mask

    def probe(flat, apply_topk, loop_num, s1, s2):
        if apply_topk <= 0:
            return orig(flat, apply_topk, loop_num, s1, s2)
        # 旧：> 第 k+1 大
        th = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1].unsqueeze(1)
        old = flat > th
        # 新：严格 k 个
        idx = torch.topk(flat, k=apply_topk, dim=1, largest=True, sorted=False).indices
        new = torch.zeros_like(flat, dtype=torch.bool)
        new.scatter_(1, idx, True)
        with torch.no_grad():
            cnt_old = old.sum(1)
            STATS["rows"] += int(old.shape[0])
            STATS["cells"] += int(old.numel())
            STATS["diff_cells"] += int((old != new).sum())
            STATS["empty_old"] += int((cnt_old == 0).sum())
            STATS["count_new_neq_k"] += int((new.sum(1) != apply_topk).sum())
            for v, c in zip(*torch.unique(cnt_old, return_counts=True)):
                STATS["count_old_hist"][int(v)] = STATS["count_old_hist"].get(int(v), 0) + int(c)
            STATS["k_hist"][int(apply_topk)] = STATS["k_hist"].get(int(apply_topk), 0) + 1
        chosen = old if mode == "old" else new
        return chosen.reshape(loop_num, s1, s2)

    wan._select_topk_mask = probe
    # 两个 mask 函数都通过模块全局查这个名字，但 pipeline 里调用的是导入后的函数对象，
    # 它们内部查的是 wan 模块的全局 → 直接改模块属性即可
    return orig


def reset_stats():
    STATS.update({"rows": 0, "cells": 0, "diff_cells": 0, "empty_old": 0,
                  "count_old_hist": {}, "count_new_neq_k": 0, "k_hist": {}})


def psnr(a, b):
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    return 99.0 if mse <= 0 else 10 * np.log10((255.0 ** 2) / mse)


def sharpness(frames):
    """平均梯度能量（越大越锐）——用来判断差异是"变软"还是"加噪声"。"""
    g = frames.astype(np.float32)
    dx = np.abs(np.diff(g, axis=2)).mean()
    dy = np.abs(np.diff(g, axis=1)).mean()
    return float(dx + dy)


def run(eng, dst):
    params = dict(scale=2, tile=128, kv=2.0, chunk=0, overlap=1, mode="tiny-long",
                  attention="sparse", quant="none", pre_scale=0.5, max_frames=MAXF,
                  encoder="cpu", quality=15, bitdepth=8, tile_overlap=12,
                  color_fix=True, seam_fix="smooth", restart=True)
    return eng.upscale_video(SRC, dst, params, lambda *a, **k: None)


def main():
    os.makedirs(WORK, exist_ok=True)
    out_old = os.path.join(WORK, "mask_old.mp4")
    out_old2 = os.path.join(WORK, "mask_old2.mp4")
    out_new = os.path.join(WORK, "mask_new.mp4")

    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a, **k: None, quant="none")
    wan = sys.modules["flashvsr_node.src.models.wan_video_dit"]

    # ---- A. 掩码统计（只跑一遍，两种选法在同一次前向里同时算）
    reset_stats()
    orig = install_probe(wan, "old")
    run(eng, out_old)                      # 顺带产出旧版视频
    wan._select_topk_mask = orig

    rows, cells = STATS["rows"], STATS["cells"]
    print("=== A. 掩码层（真实潜变量，%d 行 × %d 列）===" % (rows, cells // max(rows, 1)))
    print("  apply_topk 取值分布:", STATS["k_hist"])
    print("  旧选法每行 True 个数分布:", dict(sorted(STATS["count_old_hist"].items())[:8]),
          "…" if len(STATS["count_old_hist"]) > 8 else "")
    print("  旧选法空行（全 False，会让 sparse attn 无输入）: %d / %d 行" % (STATS["empty_old"], rows))
    print("  新选法不等于 k 的行: %d / %d" % (STATS["count_new_neq_k"], rows))
    print("  两种选法不同的 cell: %d / %d = %.3f%%"
          % (STATS["diff_cells"], cells, 100.0 * STATS["diff_cells"] / max(cells, 1)))

    # ---- A2. 噪声底：同配置再跑一遍旧选法
    run(eng, out_old2)

    # ---- B. 输出层
    install_probe(wan, "new")
    info = run(eng, out_new)
    wan._select_topk_mask = orig
    print("\n=== B. 输出层 ===")
    print("  frames=%s in_wh=%s out_wh=%s" % (info["frames"], info["in_wh"], info["out_wh"]))

    fa = E.decode_frames_range(out_old, 0, MAXF, pre_scale=1.0)
    fa2 = E.decode_frames_range(out_old2, 0, MAXF, pre_scale=1.0)
    fb = E.decode_frames_range(out_new, 0, MAXF, pre_scale=1.0)
    n = min(len(fa), len(fa2), len(fb))

    def report(tag, x, y):
        ps = [psnr(x[i], y[i]) for i in range(n)]
        df = [float(np.abs(x[i].astype(np.float32) - y[i].astype(np.float32)).mean()) for i in range(n)]
        print("  %-22s PSNR 最小 %.2f / 中位 %.2f / 均值 %.2f dB   每帧|diff| %.3f 灰度级"
              % (tag, min(ps), float(np.median(ps)), float(np.mean(ps)), float(np.mean(df))))
        return ps

    ps_noise = report("噪声底 old vs old2", fa, fa2)
    ps_sig = report("信号    old vs new", fa, fb)
    # 逐帧配对比较：信号是否显著高于噪声底
    import statistics
    delta = [a - b for a, b in zip(ps_sig, ps_noise)]
    print("  逐帧 PSNR 差值(new-old2 相对)：均值 %.2f dB，标准差 %.2f，最大下降 %.2f dB"
          % (statistics.mean(delta), statistics.pstdev(delta), min(delta)))
    print("  全黑帧: noise=%d new=%d"
          % (sum(1 for f in fa2 if float(np.asarray(f).mean()) < 1.0),
             sum(1 for f in fb if float(np.asarray(f).mean()) < 1.0)))
    print("  锐度(梯度能量): old=%.3f old2=%.3f new=%.3f  new 相对 old %+.2f%%"
          % (sharpness(fa[:n]), sharpness(fa2[:n]), sharpness(fb[:n]),
             100.0 * (sharpness(fb[:n]) / max(sharpness(fa[:n]), 1e-9) - 1)))


if __name__ == "__main__":
    main()
