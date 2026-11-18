# -*- coding: utf-8 -*-
"""证伪/证实：_local_mask_additive 里的 permute(0,2,1,3) 是否真的错位。

关键：必须用**非全 True**的局部窗，否则任何 permute 都测不出来。
选 bh=8,bw=8, local_range=3 → 掩码是稀疏的。
"""
import sys

sys.path.insert(0, r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes")
import importlib  # noqa: E402
import types  # noqa: E402

import torch  # noqa: E402
from einops import rearrange  # noqa: E402

pkg = types.ModuleType("fvnode")
pkg.__path__ = [r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast"]
sys.modules["fvnode"] = pkg
W = importlib.import_module("fvnode.src.models.wan_video_dit")


def original(lm, rl, rn, kv_double):
    """改动前（上游）的做法"""
    m = lm.unsqueeze(1).unsqueeze(0).repeat(rl, 1, rn, 1)
    m = rearrange(m, 'x a y b -> (x a) (y b)')
    if kv_double:
        m = m.repeat_interleave(2, dim=1)
    return m


def with_permute(lm, rl, rn, kv_double):
    """我写的版本（带 permute）"""
    bh, bw = lm.shape
    m = lm.unsqueeze(1).unsqueeze(0).repeat(rl, 1, rn, 1)
    m = m.permute(0, 2, 1, 3).reshape(rl * bh, rn * bw)
    if kv_double:
        m = m.repeat_interleave(2, dim=1)
    return m


def plain_reshape(lm, rl, rn, kv_double):
    """应该是正确写法：直接 reshape"""
    bh, bw = lm.shape
    m = lm.unsqueeze(1).unsqueeze(0).repeat(rl, 1, rn, 1)
    m = m.reshape(rl * bh, rn * bw)
    if kv_double:
        m = m.repeat_interleave(2, dim=1)
    return m


def main():
    dev = torch.device("cuda")
    for (bh, bw, rl, rn) in [(8, 8, 1, 1), (8, 8, 2, 1), (8, 8, 2, 2), (8, 8, 3, 2), (2, 3, 3, 4)]:
        lm = W.build_local_block_mask_shifted_vec_normal_slide(bh, bw, 3, 3, include_self=True, device=dev)
        density = float(lm.float().mean())
        for kv in (False, True):
            a = original(lm, rl, rn, kv)
            b = with_permute(lm, rl, rn, kv)
            c = plain_reshape(lm, rl, rn, kv)
            print("bh=%d bw=%d rl=%d rn=%d kv=%d  密度 %.2f  permute版 %s  reshape版 %s"
                  % (bh, bw, rl, rn, kv, density,
                     "一致" if torch.equal(a, b) else "**错位 %d/%d**" % (int((a != b).sum()), a.numel()),
                     "一致" if torch.equal(a, c) else "**错位**"))


if __name__ == "__main__":
    main()
