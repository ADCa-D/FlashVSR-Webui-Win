# -*- coding: utf-8 -*-
"""VAE 改动 A 批的单元级验证（同输入、同进程、同 kernel，这时"逐位"才有意义）。

  1. CausalConv3d：空间 padding 交回 nn.Conv3d —— 与 .orig 的实现比
  2. Upsample：去掉 .float() —— nearest-exact 在 bf16 下应逐位相同
  3. encode/decode 的 cat 重构 —— 累加方式等价（用同一批输出张量直接对拍）
"""
import importlib.util
import os
import sys
import types

MODEL = r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast\src\models"

sys.path.insert(0, r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes")
pkg = types.ModuleType("fvnode")
pkg.__path__ = [r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast"]
sys.modules["fvnode"] = pkg

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

V = importlib.import_module("fvnode.src.models.wan_video_vae")


def load_orig():
    tmp = os.path.join(MODEL, "_vae_orig.py")
    with open(os.path.join(MODEL, "wan_video_vae.py.orig"), "r", encoding="utf-8") as f:
        s = f.read()
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(s)
    return importlib.import_module("fvnode.src.models._vae_orig")


def t1(orig):
    print("1) CausalConv3d：显式 F.pad（旧） vs 交给 Conv3d padding（新）")
    torch.manual_seed(0)
    for (k, pad, cache_t, shape) in [
        (3, 1, None, (1, 8, 3, 9, 11)),
        (3, 1, 2, (1, 8, 2, 9, 11)),
        (3, 1, 2, (1, 8, 5, 16, 16)),
        (1, 0, None, (1, 8, 4, 9, 11)),
    ]:
        a = V.CausalConv3d(8, 8, k, padding=pad).cuda().eval()
        # 同一个模块搬到旧实现里：直接复制参数，保证权重相同
        b = orig.CausalConv3d(8, 8, k, padding=pad).cuda().eval()
        b.load_state_dict(a.state_dict())
        x = torch.randn(*shape, device="cuda")
        cx = None if cache_t is None else torch.randn(1, 8, cache_t, shape[3], shape[4], device="cuda")
        with torch.no_grad():
            ya = a(x, cx)
            yb = b(x, cx)
        d = (ya - yb).abs().max().item()
        print("   k=%d pad=%d cache=%s %s → maxdiff %.3e  %s"
              % (k, pad, cache_t, tuple(shape), d, "相同" if torch.equal(ya, yb) else "不同"))

    # 负 pad（cache 比需要补的多）分支
    a = V.CausalConv3d(8, 8, 3, padding=1).cuda().eval()
    b = orig.CausalConv3d(8, 8, 3, padding=1).cuda().eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 8, 2, 8, 8, device="cuda")
    cx = torch.randn(1, 8, 3, 8, 8, device="cuda")     # 3 帧 > 2*pad_t=2 → t_pad 变负
    with torch.no_grad():
        ya, yb = a(x, cx), b(x, cx)
    print("   负 pad 分支（cache 3 帧）→ maxdiff %.3e  %s"
          % ((ya - yb).abs().max(), "相同" if torch.equal(ya, yb) else "不同"))


def t2():
    print("2) Upsample：x.float()（旧） vs 直接 bf16（新）")
    for mode, sf in (("nearest-exact", (2.0, 2.0)), ("nearest", (2.0, 2.0))):
        up = V.Upsample(scale_factor=sf, mode=mode).cuda()
        x = torch.randn(1, 16, 34, 30, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            new = up(x)
            old = nn.Upsample(scale_factor=sf, mode=mode).cuda()(x.float()).to(x.dtype)
        print("   %-14s 输出 %s  maxdiff %.3e  %s"
              % (mode, tuple(new.shape), (new.float() - old.float()).abs().max().item(),
                 "相同" if torch.equal(new, old) else "不同"))


def t3():
    print("3) 循环 cat vs list+一次 cat（同一批分块输出）")
    torch.manual_seed(0)
    parts = [torch.randn(1, 4, n, 6, 6, device="cuda") for n in (1, 4, 4, 4)]
    acc = parts[0]
    for p in parts[1:]:
        acc = torch.cat([acc, p], 2)
    once = torch.cat(parts, 2)
    print("   逐次 cat %s vs 一次 cat %s → %s"
          % (tuple(acc.shape), tuple(once.shape),
             "相同" if torch.equal(acc, once) else "不同"))


if __name__ == "__main__":
    orig = load_orig()
    t1(orig)
    t2()
    t3()
    os.remove(os.path.join(MODEL, "_vae_orig.py"))
