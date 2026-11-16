# -*- coding: utf-8 -*-
"""定位 rope 差异：把参考里 token 180 / pair 5 的中间量算出来对照。"""
import math

import torch
from safetensors import safe_open

with safe_open("work/dit_ref/ref.safetensors", framework="pt", device="cpu") as f:
    q_pre = f.get_tensor("b0_q_pre").reshape(-1, 1536)
    q_rope = f.get_tensor("b0_q_rope").reshape(-1, 1536)
    k_pre = f.get_tensor("b0_k_pre").reshape(-1, 1536)
    k_rope = f.get_tensor("b0_k_rope").reshape(-1, 1536)

theta = 10000.0


def pre(dim):
    return 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].double() / dim))


fr = torch.cat([pre(44), pre(42), pre(42)])
print("len(freqs) =", fr.numel())
print("freq[0:6]  =", [round(float(v), 8) for v in fr[:6]])
print("freq[5]    =", float(fr[5]))

s = 180
ang = s * float(fr[5])
print(f"angle(s=180,j=5) = {ang:.8f}  cos={math.cos(ang):.8f} sin={math.sin(ang):.8f}")

print("q_pre  [180, 1290:1292] =", [round(float(v), 6) for v in q_pre[180, 1290:1292]])
print("q_rope [180, 1290:1292] =", [round(float(v), 6) for v in q_rope[180, 1290:1292]])
print("q_pre  [0, 1290:1292]   =", [round(float(v), 6) for v in q_pre[0, 1290:1292]])
print("q_rope [0, 1290:1292]   =", [round(float(v), 6) for v in q_rope[0, 1290:1292]])

# 复算（float64，和 torch.view_as_complex 等价）
re = q_pre[180, 1290].double()
im = q_pre[180, 1291].double()
c, sn = math.cos(ang), math.sin(ang)
print(f"manual: out0={re*c - im*sn:.6f} out1={re*sn + im*c:.6f}")

# 看看差异是不是"整块 token 偏移"：把 ref 的 token 换成别的 s 试
best = None
for s2 in range(0, 384):
    ang2 = s2 * float(fr[5])
    o0 = float(q_pre[180, 1290].double() * math.cos(ang2) - q_pre[180, 1291].double() * math.sin(ang2))
    err = abs(o0 - float(q_rope[180, 1290]))
    if best is None or err < best[0]:
        best = (err, s2)
print(f"若用 s={best[1]} 复算 out0，误差 {best[0]:.6f}（用 s=180 时下面比对）")

# 逐 token 检查：ref 的 rope 是否等于 s=token 的结果
errs = []
for s2 in (0, 1, 2, 90, 179, 180, 181, 383):
    a = s2 * float(fr[5])
    o0 = q_pre[s2, 1290].double() * math.cos(a) - q_pre[s2, 1291].double() * math.sin(a)
    o1 = q_pre[s2, 1290].double() * math.sin(a) + q_pre[s2, 1291].double() * math.cos(a)
    errs.append((s2, float(q_rope[s2, 1290]), float(o0), float(q_rope[s2, 1291]), float(o1)))
for e in errs:
    print(f"  s={e[0]:3d} ref0={e[1]:9.4f} calc0={e[2]:9.4f} | ref1={e[3]:9.4f} calc1={e[4]:9.4f}")
