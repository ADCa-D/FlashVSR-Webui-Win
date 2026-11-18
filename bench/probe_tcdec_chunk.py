# -*- coding: utf-8 -*-
"""TCDecoder 分块解码实验（只加载 180 MB 解码器，不建 DiT）。

三个问题：
  1. 时间轴分块 + parallel=True，比现在的 parallel=False 逐帧快多少？
  2. 空间条带分块（带 halo）能不能省时间？（halo 开销 vs 缓存收益）
  3. 两者对输出的影响有多大（PSNR vs 基线，0..255 尺度）。

形状依据 flashvsr_tiny_long.py:443 —— x=latent(B,T,16,h,w)，cond=LQ(B,3,4T,h,w)，
decode_video 里 pixel_shuffle(cond) 把 cond 拉到 (B,768,T,h*8,w*8) 再与 x 拼通道。
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
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    return 99.0 if mse <= 0 else 10 * np.log10(255.0 ** 2 / mse)


def build():
    dec = build_tcdecoder(new_channels=[512, 256, 128, 128], device="cuda",
                          dtype=torch.bfloat16, new_latent_channels=16 + 768)
    dec.load_state_dict(torch.load(CKPT, map_location="cuda"), strict=False)
    dec.clean_mem()
    dec.eval()
    return dec


def make_inputs(lat_hw, chunks, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, 16, chunks, lat_hw, lat_hw, device="cpu", generator=g).to("cuda", torch.bfloat16)
    # x 是 (B,C,T,h,w)，decode_video 要 (B,T,C,h,w)
    xt = x.transpose(1, 2).contiguous()
    # pixel_shuffle 把空间 /8、通道 ×256：cond 是输出 RGB 尺寸（latent×8）的 LQ 帧
    cond = torch.randn(1, 3, 4 * x.size(2), lat_hw * 8, lat_hw * 8, device="cpu", generator=g)
    cond = cond.to("cuda", torch.bfloat16)
    return xt, cond


def run_full(dec, xt, cond, parallel, cl=False):
    dec.clean_mem()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = dec.decode_video(xt, parallel=parallel, show_progress_bar=False, cond=cond)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    return dt, out.float().clone(), peak


def run_time_chunk(dec, xt, cond, k, parallel=False):
    """时间轴按 k 帧一段，mem 跨段保留（真正的流式）。"""
    dec.clean_mem()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    T = xt.size(1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = []
    with torch.no_grad():
        for s in range(0, T, k):
            e = min(s + k, T)
            xs = xt[:, s:e].contiguous()
            cs = cond[:, :, 4 * s:4 * e].contiguous()
            o = dec.decode_video(xs, parallel=parallel,
                                 show_progress_bar=False, cond=cs)
            if o.numel():
                outs.append(o)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    return dt, torch.cat(outs, 1).float().clone(), peak


def run_spatial_strips(dec, xt, cond, strips, halo):
    """空间按 strips 条切开（latent 维度），每条带 halo 后再裁掉。"""
    dec.clean_mem()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    h = xt.size(3)
    step = (h + strips - 1) // strips
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    parts = []
    with torch.no_grad():
        for i in range(strips):
            lo = i * step
            hi = min(h, lo + step)
            if lo >= hi:
                break
            a = max(0, lo - halo)
            b = min(h, hi + halo)
            # 每个条带独立 decode：mem 必须清空（条带间时间轴状态不共享）
            dec.clean_mem()
            o = dec.decode_video(xt[:, :, :, a:b].contiguous(), parallel=True,
                                 show_progress_bar=False,
                                 cond=cond[:, :, :, a * 8:b * 8].contiguous())
            # 输出空间是 latent 的 8 倍
            o = o[:, :, :, (lo - a) * 8:(lo - a) * 8 + (hi - lo) * 8, :]
            parts.append(o)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    return dt, torch.cat(parts, 3).float().clone(), peak


def report(tag, dt, out, peak, ref):
    if ref is None:
        print(f"{tag:26s} {dt * 1000:7.0f} ms  peak {peak:6.0f} MiB  out {tuple(out.shape)}", flush=True)
        return
    a = (out * 127.5 + 127.5).clamp(0, 255).cpu().numpy()
    b = (ref * 127.5 + 127.5).clamp(0, 255).cpu().numpy()
    n = min(a.shape[2], b.shape[2])
    a, b = a[:, :, :n], b[:, :, :n]
    d = np.abs(a - b)
    print(f"{tag:26s} {dt * 1000:7.0f} ms  peak {peak:6.0f} MiB  "
          f"PSNR {psnr255(a, b):6.2f} dB  max|d| {d.max():5.1f}  mean|d| {d.mean():.4f}", flush=True)


def main():
    tile = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    lat = tile // 8
    dec = build()
    xt, cond = make_inputs(lat, frames)
    print(f"tile={tile} latent={lat} T={frames} x{tuple(xt.shape)} cond{tuple(cond.shape)}", flush=True)

    # 预热（cudnn autotune）
    run_full(dec, xt, cond, False)

    dt0, ref, pk0 = run_full(dec, xt, cond, False)
    report("baseline par=False", dt0, ref, pk0, None)

    dt1, o1, pk1 = run_full(dec, xt, cond, True)
    report("full par=True", dt1, o1, pk1, ref)

    for k in (1, 2, 3, 4):
        if k > frames:
            continue
        dt, o, pk = run_time_chunk(dec, xt, cond, k)
        report(f"time-chunk k={k}", dt, o, pk, ref)

    for strips, halo in ((2, 0), (2, 2), (2, 4), (4, 4)):
        dt, o, pk = run_spatial_strips(dec, xt, cond, strips, halo)
        report(f"strips={strips} halo={halo}", dt, o, pk, ref)


if __name__ == "__main__":
    main()
