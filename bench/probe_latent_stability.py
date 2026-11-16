# -*- coding: utf-8 -*-
"""决定性测量：静止 tile 的 **latent** 是否真的跨窗稳定。

hook TCDecoder.decode_video，抓每步的 latent 输入，然后：
  1. 从源像素算静止 mask（按 latent 块对应的源尺寸）
  2. 对每步的"最新 latent 帧"按块做差，比较静止块 vs 运动块
  3. 和 bf16 混沌基线（y 加 1 ULP → maxdiff 2.34 / mean 0.30）对照

判据：静止块的跨窗 latent 差落在混沌基线内 → 复用 RGB 安全；否则要迟滞兜底。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import engine as E  # noqa: E402

SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_long_544.mp4"
SCALE = 2
FRAMES = 42          # → 3 步（4 个窗口）
CS = 256             # 256×256 裁剪：整帧跑会 OOM，生产是 128px 分块


def main():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
    pipe = eng.pipe
    mod = eng.mod

    steps = []
    rgbs = []
    orig = pipe.TCDecoder.decode_video

    def hooked(x, parallel=True, show_progress_bar=False, cond=None):
        out = orig(x, parallel=parallel, show_progress_bar=show_progress_bar, cond=cond)
        steps.append(x.detach().float().cpu())
        rgbs.append(out.detach().float().cpu())
        return out

    pipe.TCDecoder.decode_video = hooked

    u8 = E.decode_frames_range(SRC, 0, FRAMES, pre_scale=1.0, fps=25.0)
    n0 = u8.shape[0]
    # 自动挑一个"最静止"的 2×2 块（256×256）作为裁剪窗口——否则整块全在运动，
    # 就没法回答"静止块能不能复用"这个问题
    g_all = u8.astype(np.float32).mean(axis=-1)
    bh_a, bw_a = g_all.shape[1] // 128, g_all.shape[2] // 128
    acc = np.zeros((bh_a, bw_a))
    for t in range(2, len(g_all)):
        dd = np.abs(g_all[t] - g_all[t - 2])[: bh_a * 128, : bw_a * 128]
        acc += dd.reshape(bh_a, 128, bw_a, 128).mean(axis=(1, 3))
    acc /= max(1, len(g_all) - 2)
    best, bi, bj = None, 0, 0
    for i in range(max(1, bh_a - 1)):
        for j in range(max(1, bw_a - 1)):
            s = acc[i:i + 2, j:j + 2].sum()
            if best is None or s < best:
                best, bi, bj = s, i, j
    y0, x0, CS = bi * 128, bj * 128, 256
    print(f"裁剪窗口: 块({bi},{bj}) = 源 ({x0},{y0}) {CS}×{CS}，"
          f"该处平均像素差 {acc[bi:bi+2, bj:bj+2].mean():.2f}/255")
    u8 = u8[:, y0:y0 + CS, x0:x0 + CS, :]
    frames = torch.from_numpy(np.ascontiguousarray(u8)).float().div_(255.0)

    add_n = mod.next_8n5(n0) - n0
    if add_n > 0:
        frames = torch.cat([frames, frames[-1:, :, :, :].repeat([add_n, 1, 1, 1])], dim=0)

    LQ, th, tw, F = mod.prepare_input_tensor(
        frames, pipe.device, scale=SCALE, dtype=pipe.torch_dtype)

    with torch.no_grad():
        pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
             tiled=True, LQ_video=LQ, num_frames=F, height=th, width=tw,
             is_full_block=False, if_buffer=True,
             topk_ratio=2.0 * 768 * 1280 / (th * tw), kv_ratio=3.0, local_range=9,
             color_fix=False, unload_dit=False, force_offload=False)
    pipe.TCDecoder.decode_video = orig

    print(f"captured {len(steps)} steps; latent shapes {[tuple(s.shape) for s in steps]}")

    # 每步取 **最新的一帧** latent（形状随窗口变化：首窗 6 帧，之后 2 帧）
    latest = [s[0, -1] for s in steps]                 # [16, h8, w8]
    h8, w8 = latest[0].shape[1], latest[0].shape[2]
    LB = 16                                            # latent 块 = 16 px
    nb_h, nb_w = h8 // LB, w8 // LB
    psz = CS // nb_h                                   # 对应源像素块边长

    def lat_blocks(l):
        v = l.mean(dim=0)                              # [h8, w8]（信道取平均）
        v = v[: nb_h * LB, : nb_w * LB].reshape(nb_h, LB, nb_w, LB).mean(axis=(1, 3))
        return v.numpy()

    def lat_diff_blocks(a, b):
        """块内 |差的平均|（先取绝对值再块平均，否则块内运动会互相抵消）"""
        d = (a - b).abs().mean(dim=0)                  # [h8, w8]
        return d[: nb_h * LB, : nb_w * LB].reshape(nb_h, LB, nb_w, LB).mean(axis=(1, 3)).numpy()

    # 源像素静止 mask（同一块网格）
    g = u8.astype(np.float32).mean(axis=-1)            # [N,H,W] 0..255
    def pix_blocks(t_prev, t_cur):
        d = np.abs(g[t_cur] - g[t_prev])[: nb_h * psz, : nb_w * psz]
        return d.reshape(nb_h, psz, nb_w, psz).mean(axis=(1, 3))

    print(f"latent grid {nb_h}x{nb_w} blocks of {LB} latent px  (={psz} source px)")
    print(f"混沌基线参照: y maxdiff 2.34 / mean 0.30；|latent| mean "
          f"{latest[-1].abs().mean():.4f}")

    # 相邻步间隔 2 个 latent 帧；源像素取同样间隔（8 像素帧）
    for i in range(1, len(latest)):
        d = lat_diff_blocks(latest[i], latest[i - 1])
        tpx = min(8, n0 - 1)
        pm = pix_blocks(max(0, n0 - 1 - tpx), n0 - 1)   # 末尾几帧的像素差做 mask
        print(f"\nstep {i-1}→{i}:  块内 |Δlatent| mean {d.mean():.4f}  "
              f"median {np.median(d):.4f}  p90 {np.percentile(d, 90):.4f}  max {d.max():.4f}")
        for tag, m in (("静止块", pm < 1.0), ("运动块", pm >= 1.0)):
            v = d[m]
            if v.size:
                print(f"   {tag}: n={v.size:3d}  mean {v.mean():.4f}  "
                      f"median {np.median(v):.4f}  max {v.max():.4f}")
        print(f"   像素 mask: 静止 {int(np.sum(pm < 1.0))}/{pm.size} 块")
        for thr in (0.02, 0.05, 0.1, 0.22):
            print(f"   块内 |Δlatent| < {thr:4.2f} 占比: {(d < thr).mean()*100:5.1f}%")

    # ---- 解码后的 RGB 跨窗稳定性（这才是"能不能复用上一窗 RGB"的判据）----
    print("\n解码 RGB 跨窗差（同位置块，块内 |Δ| 平均，单位 0-255）:")
    print(f"  (rgb 张量形状 {tuple(rgbs[0].shape)} dtype {rgbs[0].dtype})")

    def last_frame(t):
        t = t.float()[0]
        if t.dim() == 3 and t.shape[0] == 3:        # [3,T,H,W]
            return t[:, -1]
        if t.dim() == 3 and t.shape[-1] == 3:       # [T,H,W,3]
            return t[-1].permute(2, 0, 1)
        if t.dim() == 4 and t.shape[1] == 3:        # [T,3,H,W]
            return t[-1]
        return t.reshape(-1, *t.shape[-2:])[-1]

    for i in range(1, len(rgbs)):
        d = (last_frame(rgbs[i]) - last_frame(rgbs[i - 1])).abs().mean(dim=0) * 255.0
        d = d.detach().cpu().numpy()
        H, W = d.shape
        bs = 128 if min(H, W) >= 256 else 64
        nb_h, nb_w = max(1, H // bs), max(1, W // bs)
        blk = d[: nb_h * bs, : nb_w * bs].reshape(nb_h, bs, nb_w, bs).mean(axis=(1, 3))
        print(f"  step {i-1}→{i}: mean {blk.mean():.2f}  median {np.median(blk):.2f}  "
              f"max {blk.max():.2f}   | 全块 < 1/255 占比 {(blk < 1.0).mean()*100:5.1f}%")


if __name__ == "__main__":
    sys.exit(main())
