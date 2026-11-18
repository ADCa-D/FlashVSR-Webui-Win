# -*- coding: utf-8 -*-
"""验证生产版 `_local_mask_additive` 与上游原实现逐位一致（用**非全 True**的掩码）。

之前那次 torch.equal 全 True，是因为小 tile 下 local_range=9 > 网格，掩码退化成全 True，
任何 permute 都测不出来。这里固定 local_range=3，让掩码真的稀疏。
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


def upstream(lm, rl, rn, rh, kv_double):
    m = lm.unsqueeze(1).unsqueeze(0).repeat(rl, 1, rn, 1)
    m = rearrange(m, 'x a y b -> (x a) (y b)')
    if kv_double:
        m = m.repeat_interleave(2, dim=1)
    m = m.unsqueeze(0).repeat(rh, 1, 1).to(torch.float32)
    return m.masked_fill(m == False, -float('inf')).masked_fill(m == True, 0)


def main():
    dev = torch.device("cuda")
    bad = 0
    for (bh, bw, rl, rn, rh, lr) in [
        (8, 8, 1, 1, 12, 3), (8, 8, 2, 1, 12, 3), (8, 8, 2, 2, 12, 3),
        (8, 8, 3, 2, 12, 3), (2, 3, 3, 4, 12, 3), (5, 5, 6, 3, 12, 3),
        (8, 8, 2, 2, 12, 9),   # 大范围 → 接近全 True，回归测试
    ]:
        lm = W.build_local_block_mask_shifted_vec_normal_slide(
            bh, bw, lr, lr, include_self=True, device=dev)
        den = float(lm.float().mean())
        for kv in (False, True):
            shape = (rh, rl * bh, (rn * bw) * (2 if kv else 1))
            new = W._local_mask_additive(lm, ("sig", bh, bw, lr, True), rl, rn, rh,
                                         kv, shape, torch.float32, dev)
            old = upstream(lm, rl, rn, rh, kv)
            ok = torch.equal(new.expand_as(old), old)
            bad += 0 if ok else 1
            print("bh=%d bw=%d rl=%d rn=%d range=%d kv=%d 密度%.2f  %s"
                  % (bh, bw, rl, rn, lr, kv, den, "一致" if ok else "**不一致**"))
        W._LOCAL_MASK_CACHE.clear()
    print("\n不一致项: %d" % bad)


if __name__ == "__main__":
    main()
