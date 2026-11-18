# -*- coding: utf-8 -*-
"""静止 tile 跳过实验：跳过一次 decode 会不会污染同窗后面的 tile。

背景（读源码）：TCDecoder.decode_video 会返回并保存 self.mem
（src/models/TCDecoder.py:263 x, self.mem = apply_model_with_memblocks(...)），
而 flashvsr_tiny_long.py:472 只在**窗口末尾** clean_mem()。也就是说同一窗口内
tile 之间共享 mem 状态。那么"跳过某个 tile 的解码"= 让 mem 少推进一次 →
后面所有 tile 都在错的 mem 上解码。这个脚本量它有多大。

四个问题：
  A. 不隔离时跳过 tile2，后面 tile3..5 的输出变了多少？
  B. 每 tile 前 clean_mem（隔离）后，跳过是否只影响 tile2 自己？
  C. 隔离本身有没有代价（输出 vs 不隔离，耗时 vs 不隔离）？
  D. 单次 decode 的实际耗时 → 40% 静止率能省多少 ms。

跑法：
  & D:\ComfyUI_windows_portable\python_embeded\python.exe D:\flashvsr-web\bench\probe_static_skip.py 256 6 6
"""
import os
import sys
import time

ROOT = r"D:\flashvsr-web"
NODE = r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast"
CKPT = r"D:\ComfyUI_windows_portable\ComfyUI\models\FlashVSR-v1.1\TCDecoder.ckpt"
sys.path.insert(0, NODE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from src.models.TCDecoder import build_tcdecoder  # noqa: E402

torch.backends.cudnn.benchmark = True


def psnr255(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse <= 0 else 10 * np.log10(255.0 ** 2 / mse)


def to255(t):
    return (t * 127.5 + 127.5).clamp(0, 255).cpu().numpy()


def build():
    dec = build_tcdecoder(new_channels=[512, 256, 128, 128], device="cuda",
                          dtype=torch.bfloat16, new_latent_channels=16 + 768)
    dec.load_state_dict(torch.load(CKPT, map_location="cuda"), strict=False)
    dec.clean_mem()
    dec.eval()
    return dec


def make_tiles(lat, T, n, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    tiles = []
    for _ in range(n):
        x = torch.randn(1, 16, T, lat, lat, generator=g).to("cuda", torch.bfloat16)
        cond = torch.randn(1, 3, 4 * T, lat * 8, lat * 8, generator=g).to("cuda", torch.bfloat16)
        tiles.append((x.transpose(1, 2).contiguous(), cond))
    return tiles


def run_seq(dec, tiles, isolate=False, skip=(), reuse=None):
    """顺序解码一个窗口内的所有 tile。skip 里的下标直接返回 reuse[i]（模拟复用）。"""
    dec.clean_mem()
    outs, times = [], []
    with torch.no_grad():
        for i, (xt, cond) in enumerate(tiles):
            if isolate:
                dec.clean_mem()
            if i in skip:
                outs.append(reuse[i])
                times.append(0.0)
                continue
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            o = dec.decode_video(xt, parallel=False, show_progress_bar=False, cond=cond)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            outs.append(o.float())
    return outs, times


def cmp(tag, a, b, idx):
    for i in idx:
        da = to255(a[i])
        db = to255(b[i])
        n = min(da.shape[1], db.shape[1])
        d = np.abs(da[:, :n] - db[:, :n])
        print(f"  {tag} tile{i}: PSNR {psnr255(da[:, :n], db[:, :n]):6.2f} dB  "
              f"max|d| {d.max():5.1f}  mean|d| {d.mean():.3f}", flush=True)


def main():
    tile = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    lat = tile // 8
    dec = build()
    tiles = make_tiles(lat, T, n)
    skip = {2}
    print(f"tile={tile} latent={lat} T={T} tiles={n}  单 tile 输出 {tile}x{tile}x{4*T}帧",
          flush=True)

    run_seq(dec, tiles[:1])

    ref_a, t_a = run_seq(dec, tiles, isolate=False)
    hit_a, _ = run_seq(dec, tiles, isolate=False, skip=skip, reuse=ref_a)
    print("A. 不隔离（生产现状：mem 跨 tile 共享）", flush=True)
    print(f"  单 tile decode {1000 * np.mean(t_a):.0f} ms（{1000 * sum(t_a):.0f} ms / {n} tile）",
          flush=True)
    cmp("跳过 tile2 后", ref_a, hit_a, [2, 3, 4, 5])

    ref_b, t_b = run_seq(dec, tiles, isolate=True)
    hit_b, _ = run_seq(dec, tiles, isolate=True, skip=skip, reuse=ref_b)
    print("B. 每 tile clean_mem（隔离）", flush=True)
    print(f"  单 tile decode {1000 * np.mean(t_b):.0f} ms（{1000 * sum(t_b):.0f} ms / {n} tile）",
          flush=True)
    cmp("跳过 tile2 后", ref_b, hit_b, [2, 3, 4, 5])

    print("C. 隔离本身的影响（ref_b vs ref_a，同一批输入）", flush=True)
    cmp("clean_mem 每 tile 后", ref_b, ref_a, [0, 1, 2, 3, 4, 5])
    print(f"  耗时：不隔离 {1000 * sum(t_a):.0f} ms  vs  隔离 {1000 * sum(t_b):.0f} ms "
          f"（{(sum(t_b) / max(sum(t_a), 1e-9) - 1) * 100:+.1f}%）", flush=True)

    per = float(np.mean(t_a))
    for frac in (0.4, 0.3, 0.2):
        print(f"D. 静止率 {frac:.0%} → 省 {1000 * frac * per * n:.0f} ms / 窗（{n} tile）",
              flush=True)


if __name__ == "__main__":
    main()
