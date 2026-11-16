# -*- coding: utf-8 -*-
"""关键一问：把"激活量化"塞进内核（寄存器内 cast），能否救回 fp8 卷积的收益？

上一版失败原因：每层额外 4 个 kernel（float→div→clamp→cast）+ 布局拷贝 ≈2.6 ms/层。
本版：内核直接读 **bf16 激活**，在寄存器里乘 1/scale 并 cast 到 fp8 —— 零额外内存往返；
scale 用标定值（每层一个常数），不做每帧 amax。

用法：python bench/triton_conv_fp8_inline.py
"""
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv3x3_bf16in_fp8w(
    x_ptr, w_ptr, out_ptr,
    M, COUT, H, W, inv_sa, sw,
    CIN: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < COUT

    pix = offs_m
    n_idx = pix // (H * W)
    s = pix % (H * W)
    h = s // W
    w_ = s % W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for tap in tl.static_range(9):
        dh = tap // 3 - 1
        dw = tap % 3 - 1
        hh = h + dh
        ww = w_ + dw
        ok = mask_m & (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)
        base = (n_idx * H * W + hh * W + ww) * CIN
        for k0 in tl.static_range(0, CIN, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(x_ptr + base[:, None] + ks[None, :], mask=ok[:, None], other=0.0)
            a = (a * inv_sa).to(tl.float8e4nv)          # ← 寄存器内量化，无额外内存
            b = tl.load(w_ptr + (tap * COUT + offs_n)[None, :] * CIN + ks[:, None],
                        mask=mask_n[None, :], other=0.0)
            acc += tl.dot(a, b, out_dtype=tl.float32)

    out = acc * (sw / inv_sa)
    tl.store(out_ptr + offs_m[:, None] * COUT + offs_n[None, :],
             out.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def bench(fn, warm=3, iters=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def prep_w(wt):
    amax = wt.abs().amax().clamp_min(1e-8)
    sw = (amax / 448.0).float()
    wq = (wt.float() / sw).clamp(-448, 448).to(torch.float8_e4m3fn)
    # (COUT,CIN,3,3) → (3,3,COUT,CIN)
    return wq.permute(2, 3, 0, 1).contiguous(), sw


def conv_inline(x_nhwc_bf16, wq, sw, sa):
    n, h, w, c = x_nhwc_bf16.shape
    cout = wq.shape[2]
    m = n * h * w
    out = torch.empty((m, cout), device=x_nhwc_bf16.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(m, 64), triton.cdiv(cout, 64))
    _conv3x3_bf16in_fp8w[grid](
        x_nhwc_bf16, wq, out, m, cout, h, w, 1.0 / sa.item(), sw.item(),
        CIN=c, BLOCK_M=64, BLOCK_N=64, BLOCK_K=min(64, c), num_warps=4, num_stages=3)
    return out.reshape(n, h, w, cout)


SHAPES = [("8x128x256x256", (8, 128, 256, 256), 128),
          ("8x256x256x256", (8, 256, 256, 256), 256),
          ("8x512x128x128", (8, 512, 128, 128), 512)]

print(f"{'形状':16s} {'cudnn bf16':>11s} {'inline fp8':>11s} {'加速':>6s} {'相对误差%':>9s}")
for tag, (n, c, h, w), cout in SHAPES:
    x = torch.randn(n, h, w, c, device="cuda", dtype=torch.bfloat16) * 0.5
    wt = torch.randn(cout, c, 3, 3, device="cuda", dtype=torch.bfloat16) * 0.05
    wq, sw = prep_w(wt)
    sa = (x.abs().amax() / 448.0).float()          # 标定值（离线一次）

    ref = F.conv2d(x.permute(0, 3, 1, 2).contiguous(), wt, padding=1)

    def run():
        y = conv_inline(x, wq, sw, sa)              # 输入直接 bf16，无需预量化
        return y.permute(0, 3, 1, 2)

    got = run()
    rel = ((got.float() - ref.float()).abs().mean() / ref.float().abs().mean() * 100).item()
    t_tri = bench(run)
    t_bf = bench(lambda: F.conv2d(x.permute(0, 3, 1, 2).contiguous(), wt, padding=1))
    print(f"{tag:16s} {t_bf*1e3:9.2f}ms {t_tri*1e3:9.2f}ms {t_bf/t_tri:5.2f}x {rel:8.2f}%")
    del x, wt, wq, ref, got
    torch.cuda.empty_cache()
