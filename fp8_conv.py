# -*- coding: utf-8 -*-
"""fp8 卷积（v2）：激活在**内核寄存器内**量化，避免每层 4 个 kernel + 3 遍内存往返。

实测（含布局 permute，不含外层 NCHW→NHWC 拷贝）：
  8×128×256²  7.79 → 3.10 ms（2.51×）   相对误差 3.7%
  8×256×256² 25.01 → 12.12 ms（2.06×）
  8×512×128² 20.57 → 12.19 ms（1.69×）

scale 用"首次 forward 标定一次"的常数（不做每帧 amax）；权重 per-tensor 量化。
"""
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # noqa
    _HAS_TRITON = False


def available():
    if not _HAS_TRITON:
        return False
    try:
        major, minor = torch.cuda.get_device_capability(0)
        return (major, minor) >= (8, 9)
    except Exception:  # noqa
        return False


if _HAS_TRITON:

    @triton.jit
    def _conv3x3_bf16in_fp8w(
        x_ptr, w_ptr, out_ptr, M, COUT, H, W, inv_sa, sw,
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
                a = (a * inv_sa).to(tl.float8e4nv)      # 寄存器内量化
                b = tl.load(w_ptr + (tap * COUT + offs_n)[None, :] * CIN + ks[:, None],
                            mask=mask_n[None, :], other=0.0)
                acc += tl.dot(a, b, out_dtype=tl.float32)

        out = acc * (sw / inv_sa)
        tl.store(out_ptr + offs_m[:, None] * COUT + offs_n[None, :],
                 out.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


class Conv2dFP8(nn.Module):
    """k=3/s1/p1 Conv2d → fp8 隐式 GEMM（激活 bf16 进、寄存器内转 fp8）。"""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        assert conv.kernel_size == (3, 3) and conv.stride == (1, 1)
        assert conv.padding == (1, 1) and conv.groups == 1 and conv.dilation == (1, 1)
        self.conv = conv
        self._wq = None
        self._sw = None
        self._inv_sa = None

    def _prep_weights(self):
        if self._wq is not None:
            return
        w = self.conv.weight.detach()
        amax = w.abs().amax().clamp_min(1e-8)
        sw = (amax / 448.0).float()
        self._wq = (w.float() / sw).clamp(-448, 448).to(torch.float8_e4m3fn)
        self._wq = self._wq.permute(2, 3, 0, 1).contiguous()      # (3,3,COUT,CIN)
        self._sw = sw

    def forward(self, x):
        self._prep_weights()
        x = x.to(torch.bfloat16)
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        n, c, h, w = x.shape
        x_nhwc = x.permute(0, 2, 3, 1)                            # channels_last 下是 view
        if self._inv_sa is None:
            sa = (x_nhwc.abs().amax() / 448.0).clamp_min(1e-6).float()
            self._inv_sa = (1.0 / sa)
        cout = self.conv.out_channels
        m = n * h * w
        out = torch.empty((m, cout), device=x.device, dtype=torch.bfloat16)
        grid = (triton.cdiv(m, 64), triton.cdiv(cout, 64))
        _conv3x3_bf16in_fp8w[grid](
            x_nhwc, self._wq, out, m, cout, h, w, float(self._inv_sa), float(self._sw),
            CIN=c, BLOCK_M=64, BLOCK_N=64, BLOCK_K=min(64, c), num_warps=4, num_stages=3)
        y = out.reshape(n, h, w, cout).permute(0, 3, 1, 2)
        y = y.to(x.dtype)
        if self.conv.bias is not None:
            y = y + self.conv.bias.view(1, -1, 1, 1)
        return y


def patch_module(mod, verbose=False):
    """替换模块内所有 k3/s1/p1 Conv2d；返回替换数量（0 表示平台不支持）。"""
    if not available():
        return 0
    n = 0
    for name, child in list(mod.named_children()):
        if isinstance(child, Conv2dFP8):
            continue
        if isinstance(child, nn.Conv2d):
            k = child.kernel_size if isinstance(child.kernel_size, tuple) else (child.kernel_size,) * 2
            s = child.stride if isinstance(child.stride, tuple) else (child.stride,) * 2
            p = child.padding if isinstance(child.padding, tuple) else (child.padding,) * 2
            if k == (3, 3) and s == (1, 1) and p == (1, 1) and child.groups == 1:
                setattr(mod, name, Conv2dFP8(child))
                n += 1
                continue
        n += patch_module(child, verbose)
    if verbose and n:
        print(f"[fp8conv] 替换 {n} 个 Conv2d", flush=True)
    return n
