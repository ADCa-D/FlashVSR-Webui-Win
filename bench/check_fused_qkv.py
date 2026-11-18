# -*- coding: utf-8 -*-
"""校验 Fused QKV（分两步，避免被模型自身的 0.2 抖动干扰）：

  1. 结构正确性（bf16、未量化）：cat([q;k;v]) 的权重做一次 F.linear 再 chunk，
     必须 ≈ 三次独立投影 —— 这一步只验「拼接顺序 + chunk 切分」
  2. 量化误差（mxfp8 fused vs bf16 fused）：只应差在 fp8 量化上（~1e-2 相对）
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import engine as E  # noqa: E402


def step1():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="none")
    pipe = eng.pipe
    pipe.load_models_to_device(["dit"])
    sa = pipe.dit.blocks[0].self_attn
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 1024, 1536, generator=g).to(dtype=torch.bfloat16, device="cuda")

    with torch.no_grad():
        W = torch.cat([sa.q.weight, sa.k.weight, sa.v.weight], 0)
        b = None
        if sa.q.bias is not None:
            b = torch.cat([sa.q.bias, sa.k.bias, sa.v.bias], 0)
        yq, yk, yv = F.linear(x, W, b).chunk(3, dim=-1)
        rq, rk, rv = sa.q(x), sa.k(x), sa.v(x)
        for name, a, ref in (("q", yq, rq), ("k", yk, rk), ("v", yv, rv)):
            rel = float((a.float() - ref.float()).norm() / ref.float().norm())
            print("  [结构/bf16] %s: 相对差 %.2e  maxdiff %.4f  (应只在 bf16 舍入量级)"
                  % (name, rel, (a.float() - ref.float()).abs().max()))
        return x.clone(), (yq.clone(), yk.clone(), yv.clone())


def step2(x, ref_bf16):
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
    pipe = eng.pipe
    pipe.load_models_to_device(["dit"])
    sa = pipe.dit.blocks[0].self_attn
    fused = getattr(sa, "_qkv_fused", None)
    print("  [mxfp8] fused 已安装:", fused is not None)
    if fused is None:
        return
    with torch.no_grad():
        y = fused(x).chunk(3, dim=-1)
        for name, a, ref in zip(("q", "k", "v"), y, ref_bf16):
            rel = float((a.float() - ref.float()).norm() / ref.float().norm())
            print("  [fp8 误差] %s: 相对差 %.2e（fp8 量化误差，与未 fused 时同量级即可）" % (name, rel))


if __name__ == "__main__":
    print("1) 结构正确性（bf16，cat[q;k;v] vs 三次独立投影）")
    x, ref = step1()
    print("2) fp8 量化误差（mxfp8 fused vs bf16 fused）")
    step2(x, ref)
