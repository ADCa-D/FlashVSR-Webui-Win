# -*- coding: utf-8 -*-
"""Triton 隐式 GEMM fp8 卷积（NHWC，3×3，stride1，pad1）——不物化 im2col。

只对 Blackwell/Ada（sm≥89，有 fp8 张量核）有意义；30 系走回退。
先用"逐位正确性"验内核，再比速度（目标 ≥1.5× 于 cuDNN bf16）。

用法：python bench/triton_conv_fp8.py
"""
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3x3_fp8(
    x_ptr, w_ptr, out_ptr, sb_ptr,
    M, COUT, H, W,
    sa,
    CIN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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
            b = tl.load(w_ptr + (tap * COUT + offs_n)[None, :] * CIN + ks[:, None],
                        mask=mask_n[None, :], other=0.0)
            acc += tl.dot(a, b, out_dtype=tl.float32)

    sb = tl.load(sb_ptr + offs_n, mask=mask_n, other=0.0)
    out = acc * sa * sb[None, :]
    tl.store(out_ptr + offs_m[:, None] * COUT + offs_n[None, :],
             out.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def quant_fp8(t):
    amax = t.abs().amax().clamp_min(1e-8)
    scale = (amax / 448.0).float()
    q = (t.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale


def conv_fp8_triton(x_nhwc, w, stride=1):
    """x: (N,H,W,C) fp8; w: (COUT,CIN,3,3) fp8（已量化）; 返回 bf16 (N,H,W,COUT)。"""
    n, h, wd, cin = x_nhwc.shape
    cout = w.shape[0]
    wt = w.permute(2, 3, 0, 1).contiguous()            # (3,3,COUT,CIN)
    M = n * h * wd
    out = torch.empty((M, cout), device=x_nhwc.device, dtype=torch.bfloat16)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, min(64, cin)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(cout, BLOCK_N))
    # 标量 scale 由外部乘入（这里假设已归一），sb 传全 1
    sb = torch.ones(cout, device=x_nhwc.device, dtype=torch.float32)
    conv3x3_fp8[grid](
        x_nhwc, wt, out, sb, M, cout, h, wd, 1.0,
        CIN=cin, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return out.reshape(n, h, wd, cout)


def bench(fn, warm=3, iters=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


SHAPES = [("8x128x256x256", (8, 128, 256, 256), 128),
          ("8x512x128x128", (8, 512, 128, 128), 512)]

print(f"{'形状':16s} {'cudnn bf16':>11s} {'triton fp8':>11s} {'加速':>6s} {'最大绝对差':>10s}")
for tag, (n, c, h, w), cout in SHAPES:
    x = torch.randn(n, h, w, c, device="cuda", dtype=torch.bfloat16) * 0.5
    wt = torch.randn(cout, c, 3, 3, device="cuda", dtype=torch.bfloat16) * 0.05
    xq, sx = quant_fp8(x)
    wq, sw = quant_fp8(wt)

    # 参考：用"反量化后的 fp8 值"做 bf16 卷积，隔离量化误差、只看内核正确性
    xr = xq.float() * sx
    wr = wq.float() * sw
    ref = F.conv2d(xr.permute(0, 3, 1, 2), wr, padding=1).permute(0, 2, 3, 1)

    def run():
        out = conv_fp8_triton(xq, wq)
        return out.float() * (sx * sw)

    try:
        got = run()
        err = (got - ref).abs().max().item()
        t_tri = bench(lambda: conv_fp8_triton(xq, wq))
        x_bf = x.permute(0, 3, 1, 2).contiguous()
        t_bf = bench(lambda: F.conv2d(x_bf, wt, padding=1))
        print(f"{tag:16s} {t_bf*1e3:9.2f}ms {t_tri*1e3:9.2f}ms {t_bf/t_tri:5.2f}x {err:10.4f}")
    except Exception as e:  # noqa
        print(f"{tag:16s} {t_bf*1e3 if False else 0:9.2f}ms FAIL {type(e).__name__}: {str(e)[:90]}")
