# -*- coding: utf-8 -*-
"""对比 rope_apply 新旧实现（新=实数 fp32，旧=complex128），以及 mask 展开是否等价。"""
import sys

sys.path.insert(0, r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast\src\models")

import torch  # noqa: E402
import wan_video_dit as W  # noqa: E402


def rope_old(x, freqs, num_heads):
    x = x.view(x.shape[0], x.shape[1], num_heads, -1)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def main():
    torch.manual_seed(0)
    S, n, hd = 384, 12, 128
    freqs = W.precompute_freqs_cis_3d(hd, 1024)
    fr = torch.cat([
        freqs[0][:6].view(6, 1, 1, -1).expand(6, 8, 16, -1),
        freqs[1][:8].view(1, 8, 1, -1).expand(6, 8, 16, -1),
        freqs[2][:16].view(1, 1, 16, -1).expand(6, 8, 16, -1),
    ], dim=-1).reshape(S, 1, -1)
    x = torch.randn(1, S, n * hd, device="cuda", dtype=torch.bfloat16)

    a = rope_old(x, fr, n)
    b = W.rope_apply(x, fr, n)
    print("freqs shape", tuple(fr.shape), "cos/sin", tuple(W._rope_cos_sin(fr)[0].shape))
    print("maxdiff(bf16) %.3e   equal %s" % ((a.float() - b.float()).abs().max().item(),
                                             torch.equal(a, b)))
    # 同一对象第二次调用（走缓存）
    b2 = W.rope_apply(x, fr, n)
    print("cache hit equal:", torch.equal(b, b2))


if __name__ == "__main__":
    main()
