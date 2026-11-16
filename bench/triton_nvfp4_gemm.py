# -*- coding: utf-8 -*-
"""NVFP4 GEMM via Triton `tl.dot_scaled`：数值 vs comfy_kitchen(cutlass)、耗时对比。

布局（来自 ck.quantize_nvfp4 实测）：
  values  uint8 (M, K/2)     —— 每字节 2 个 e2m1
  scales  fp8_e4m3 (M, K/16) —— 每 16 个元素一个 block scale
  per-tensor scale 在外部乘（engine 的 _ts() = amax/(6*448)）
"""
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import comfy_kitchen as ck


def _ts(t):
    return (t.abs().amax() / (6.0 * 448.0)).clamp(min=1e-8).float().reshape(1)


@triton.jit
def _gemm_nvfp4(a_ptr, asf_ptr, b_ptr, bsf_ptr, c_ptr, M, N, KP, SK, sa, sb,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, KP, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        # 每 BLOCK_K 个 packed（= BLOCK_K*2 个元素）对应 BLOCK_K//8 个 scale block
        rs = k0 // 8 + tl.arange(0, BLOCK_K // 8)
        a = tl.load(a_ptr + rm[:, None] * KP + rk[None, :], mask=rm[:, None] < M)
        asf = tl.load(asf_ptr + rm[:, None] * SK + rs[None, :], mask=rm[:, None] < M)
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :], mask=rn[None, :] < N)
        bsf = tl.load(bsf_ptr + rs[:, None] * N + rn[None, :], mask=rn[None, :] < N)
        acc = tl.dot_scaled(a, asf, "e2m1", b, bsf, "e2m1", acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], (acc * sa * sb).to(tl.bfloat16),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def bench(fn, warm=3, iters=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    M, N, K = 2304, 1536, 8960
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02

    ta, tw = _ts(a), _ts(w)
    aq, asf = ck.quantize_nvfp4(a, ta, pad_16x=True)      # (M,K/2) u8, (M,K/16) fp8
    wq, wsf = ck.quantize_nvfp4(w, tw, pad_16x=True)      # (N,K/2), (N,K/16)

    # ---- 参考：comfy_kitchen（cutlass）
    ref = ck.scaled_mm_nvfp4(aq, wq, ta, tw, asf, wsf, out_dtype=torch.bfloat16)
    t_ck = bench(lambda: ck.scaled_mm_nvfp4(aq, wq, ta, tw, asf, wsf, out_dtype=torch.bfloat16))

    # ---- Triton：B 需要 (K/2, N) 布局 + scale (K/16, N)
    b_t = wq.t().contiguous()
    bsf_t = wsf.t().contiguous()
    out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    BM, BN, BK = 64, 64, 64
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    def tri():
        _gemm_nvfp4[grid](aq, asf, b_t, bsf_t, out, M, N, K // 2, K // 16,
                          float(ta.item()), float(tw.item()),
                          BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=4, num_stages=3)

    try:
        tri()
        torch.cuda.synchronize()
        err = (out.float() - ref.float()).abs()
        rel = (err.mean() / ref.float().abs().mean()).item() * 100
        t_tri = bench(tri)
        print(f"NVFP4 GEMM  M={M} N={N} K={K}")
        print(f"  ck(cutlass) {t_ck*1e3:6.2f} ms")
        print(f"  triton      {t_tri*1e3:6.2f} ms   ({t_ck/t_tri:.2f}×)")
        print(f"  数值：平均绝对差 {err.mean().item():.5f}  相对 {rel:.2f}%  "
              f"最大 {err.max().item():.4f}")
    except Exception as e:  # noqa
        import traceback
        tb = traceback.format_exc()
        with open("work/nvfp4_tb.txt", "w", encoding="utf-8") as f:
            f.write(tb)
        print("triton FAIL:", type(e).__name__, str(e)[:400])


if __name__ == "__main__":
    main()
