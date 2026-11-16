# -*- coding: utf-8 -*-
"""显存账：ensure_loaded 之后，权重到底在 CPU 还是 GPU？

  quant=none   → 可能整模型还留在 CPU（真正上卡发生在推理时）
  quant=mxfp8  → 量化那步多半把权重留在 GPU

输出：每个子模块的参数总量（按设备分）、torch 记账、设备实际占用。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import engine as E  # noqa: E402

which = sys.argv[1] if len(sys.argv) > 1 else "mxfp8"


def devices_of(mod):
    """统计每个设备上的参数字节数。"""
    acc = {}
    for p in mod.parameters():
        acc[str(p.device)] = acc.get(str(p.device), 0) + p.numel() * p.element_size()
    # 被量化替换掉的 forward 里挂着的 fp8 张量不体现在 parameters()，单独抓
    extra = 0
    for m in mod.modules():
        f = getattr(m, "forward", None)
        for cell in getattr(f, "__closure__", None) or ():
            try:
                v = cell.cell_contents
            except ValueError:
                continue
            if torch.is_tensor(v):
                extra += v.numel() * v.element_size()
    return {k: v / 2 ** 20 for k, v in acc.items()}, extra / 2 ** 20


eng = E.Engine()
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant=which)
pipe = eng.pipe

print(f"=== quant={which} ===")
for name in ("dit", "TCDecoder"):
    mod = getattr(pipe, name, None)
    if mod is None:
        continue
    acc, extra = devices_of(mod)
    total = sum(acc.values())
    print(f"  {name:12s} parameters {total:7.0f} MiB  按设备 {acc}  "
          f"闭包里的张量 {extra:.0f} MiB")

free_b, total_b = torch.cuda.mem_get_info()
print(f"  torch allocated {torch.cuda.memory_allocated()/2**20:.0f} MiB  "
      f"reserved {torch.cuda.memory_reserved()/2**20:.0f} MiB")
print(f"  设备实际占用 {(total_b-free_b)/2**20:.0f} MiB")
