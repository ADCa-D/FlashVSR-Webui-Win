# -*- coding: utf-8 -*-
"""smoke test：Triton 3.9 的 block-scaled MMA（tl.dot_scaled）在本机 sm120 上能不能用。

A. bf16 + 无 scale       —— 只验证原语能否 lower（MMA 路径是否存在）
B. mxfp8 (e4m3 + e8m0)   —— 与 comfy_kitchen 的 scaled_mm_mxfp8 比数值

通过了才值得写 fp4（e2m1）版本。
"""
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_bf16(a_ptr, b_ptr, c_ptr, M, N, K,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M))
        b = tl.load(b_ptr + rn[None, :] * K + rk[:, None], mask=(rn[None, :] < N))
        # dot_scaled 的 bf16 路径：无 scale
        acc = tl.dot_scaled(a, None, "bf16", b, None, "bf16", acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16),
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
    print("triton", triton.__version__, "| device", torch.cuda.get_device_name(0),
          torch.cuda.get_device_capability(0))

    M, N, K = 2048, 1536, 2048
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    # ---------- A：bf16 + 无 scale
    try:
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        _gemm_bf16[grid](a, b, c, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
                         num_warps=4, num_stages=3)
        torch.cuda.synchronize()
        ref = (a.float() @ b.float().t()).to(torch.bfloat16)
        err = (c.float() - ref.float()).abs().max().item()
        t = bench(lambda: _gemm_bf16[grid](a, b, c, M, N, K, BLOCK_M=64, BLOCK_N=64,
                                           BLOCK_K=64, num_warps=4, num_stages=3))
        tf = 2 * M * N * K / t / 1e12
        print(f"A bf16 dot_scaled: OK  max_err={err:.4f}  {t*1e3:.2f} ms  {tf:.1f} TFLOPS")
    except Exception as e:
        print(f"A bf16 dot_scaled: FAIL {type(e).__name__}: {str(e)[:200]}")

    # ---------- B：mxfp8（comfy_kitchen 产出 values + e8m0 scales）
    try:
        import comfy_kitchen as ck
        aq, asc = ck.quantize_mxfp8(a)
        bq, bsc = ck.quantize_mxfp8(b)
        got = ck.scaled_mm_mxfp8(aq, bq, asc, bsc, out_dtype=torch.bfloat16)
        ref8 = got.float()
        print(f"B comfy_kitchen mxfp8 基准: {got.shape}  mean={ref8.mean().item():.4f}")
        print(f"   （Triton 侧需要与它相同的 values/scales 布局，下一步再对齐）")
    except Exception as e:
        print(f"B mxfp8: FAIL {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
