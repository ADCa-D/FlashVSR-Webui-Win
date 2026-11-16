# -*- coding: utf-8 -*-
"""复刻 wan.rope_apply，确认参考实现到底做了什么。"""
import math

import torch
from safetensors import safe_open

with safe_open("work/dit_ref/ref.safetensors", framework="pt", device="cpu") as f:
    q_pre = f.get_tensor("b0_q_pre")          # [1,384,1536] (f32 存的是 bf16 值)
    q_rope = f.get_tensor("b0_q_rope")
    q_w = f.get_tensor("q_w")                 # [3,128,1536] window 分区后的 q

theta = 10000.0


def pre(dim):
    return 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].double() / dim))


F, H, W = 6, 8, 8
n = 12
f_cis = torch.polar(torch.ones(1024, 22, dtype=torch.float64),
                    torch.outer(torch.arange(1024, dtype=torch.float64), pre(44)))
h_cis = torch.polar(torch.ones(1024, 21, dtype=torch.float64),
                    torch.outer(torch.arange(1024, dtype=torch.float64), pre(42)))
w_cis = torch.polar(torch.ones(1024, 21, dtype=torch.float64),
                    torch.outer(torch.arange(1024, dtype=torch.float64), pre(42)))
freqs = torch.cat([
    f_cis[:F].view(F, 1, 1, -1).expand(F, H, W, -1),
    h_cis[:H].view(1, H, 1, -1).expand(F, H, W, -1),
    w_cis[:W].view(1, 1, W, -1).expand(F, H, W, -1),
], dim=-1).reshape(F * H * W, 1, -1)
print("freqs", tuple(freqs.shape), freqs.dtype)


def rope_like(x, freqs, n):
    """wan.rope_apply 的逐行复刻"""
    b, s, _ = x.shape
    xr = x.view(b, s, n, -1)
    xo = torch.view_as_complex(xr.double().reshape(b, s, n, -1, 2))
    o = torch.view_as_real(xo * freqs).flatten(2)
    return o.to(x.dtype)


qp = q_pre.to(torch.bfloat16)
mine = rope_like(qp, freqs, n)
print("复刻 vs ref  maxdiff:", float((mine.float() - q_rope).abs().max()))

# 手工旋转（double）对照
for s, pair in ((0, 5), (180, 5), (90, 5)):
    ch = 10 * 128 + pair * 2
    re, im = float(qp[0, s, ch]), float(qp[0, s, ch + 1])
    ang = s * float(freqs[s, 0, pair].angle())
    c, sn = math.cos(ang), math.sin(ang)
    print(f"s={s:3d} pair={pair} pre=({re:.4f},{im:.4f}) "
          f"ref=({float(q_rope[0, s, ch]):.4f},{float(q_rope[0, s, ch + 1]):.4f}) "
          f"mine=({float(mine[0, s, ch]):.4f},{float(mine[0, s, ch + 1]):.4f}) "
          f"manual=({re * c - im * sn:.4f},{re * sn + im * c:.4f})")

# window 分区后的 q_w 是否 = 分区的 mine / ref
def partition(t):
    x = t.view(1, F, H, W, 1536)
    x = x.view(1, F // 2, 2, H // 8, 8, W // 8, 8, 1536)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, 128, 1536)


print("q_w vs partition(ref_q_rope) maxdiff:",
      float((q_w - partition(q_rope)).abs().max()))
print("q_w vs partition(mine)       maxdiff:",
      float((q_w - partition(mine)).abs().max()))
