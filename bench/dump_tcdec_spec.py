# -*- coding: utf-8 -*-
"""把 TCDecoder 的真实层序、参数、以及一次 decode 的张量形状全部扒出来。

输出：
  work/tcdec/spec.json     —— 每层的类型/参数/输入输出形状（C++ 移植的机械依据）
  work/tcdec/ref.pt        —— 固定输入下的参考输出（latent / cond / rgb / 每层输出）
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import engine as E  # noqa: E402

OUT = os.path.join(ROOT, "work", "tcdec")
TCD = r"D:\ComfyUI_windows_portable\ComfyUI\models\FlashVSR-v1.1\TCDecoder.ckpt"


def describe(mod, idx):
    t = type(mod).__name__
    d = {"idx": idx, "type": t}
    if hasattr(mod, "in_channels"):
        d["in"] = mod.in_channels
        d["out"] = mod.out_channels
        d["k"] = list(mod.kernel_size) if hasattr(mod.kernel_size, "__len__") else mod.kernel_size
        d["stride"] = list(mod.stride) if hasattr(mod.stride, "__len__") else mod.stride
        d["padding"] = list(mod.padding) if hasattr(mod.padding, "__len__") else mod.padding
        d["bias"] = mod.bias is not None
    if t == "MemBlock":
        d["n_in"] = mod.conv[0].in_channels // 2
        d["n_out"] = mod.conv[-1].out_channels
        d["skip"] = type(mod.skip).__name__
    if t in ("TPool", "TGrow"):
        d["stride_n"] = mod.stride
        d["conv_in"] = mod.conv.in_channels
        d["conv_out"] = mod.conv.out_channels
    if t in ("Upsample",):
        d["scale"] = mod.scale_factor
    if t == "PixelShuffle3d":
        d["ff"], d["hh"], d["ww"] = mod.ff, mod.hh, mod.ww
    if t == "IdentityConv2d":
        d["in"] = mod.in_channels
        d["out"] = mod.out_channels
        d["k"] = mod.kernel_size[0]
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
    pipe = eng.pipe
    dec = pipe.TCDecoder
    dec.to("cuda")

    layers = [describe(m, i) for i, m in enumerate(dec.decoder)]
    print(f"decoder 层数 {len(layers)}")
    for L in layers:
        print(f"  {L['idx']:2d} {L['type']:16s} " +
              (f"in={L.get('in')} out={L.get('out')} k={L.get('k')} s={L.get('stride')} "
               f"pad={L.get('padding')} bias={L.get('bias')}"
               if "in" in L else
               f"stride={L.get('stride_n') or L.get('scale')} "
               f"conv={L.get('conv_in')}->{L.get('conv_out')}"
               if "conv_in" in L or "scale" in L else
               f"n_in={L.get('n_in')} n_out={L.get('n_out')} skip={L.get('skip')}"
               if L["type"] == "MemBlock" else ""))
    print(f"pixel_shuffle: ff={dec.pixel_shuffle.ff} hh={dec.pixel_shuffle.hh} "
          f"ww={dec.pixel_shuffle.ww}")
    print(f"frames_to_trim={dec.frames_to_trim}")

    # ---- 固定输入跑一次，抓每层输出形状 ----------------------------------
    torch.manual_seed(0)
    T, C, H, W = 2, 16, 16, 16
    x = torch.randn(1, T, C, H, W, device="cuda", dtype=torch.bfloat16)
    cond = torch.randn(1, T, 3, H * 8, W * 8, device="cuda", dtype=torch.bfloat16) * 0.5

    shapes, grabs = [], {}
    orig = dec.decode_video

    def probe_call(xx, parallel=True, show_progress_bar=False, cond=None):
        out = orig(xx, parallel=parallel, show_progress_bar=show_progress_bar, cond=cond)
        return out

    # 逐层 hook
    hooks = []
    for i, m in enumerate(dec.decoder):
        def mk(i):
            def fn(mod, inp, out):
                a = inp[0] if isinstance(inp, (tuple, list)) else inp
                if torch.is_tensor(a):
                    shapes.append({"idx": i, "type": type(mod).__name__,
                                   "in": list(a.shape),
                                   "out": list(out.shape) if torch.is_tensor(out) else None})
            return fn
        hooks.append(m.register_forward_hook(mk(i)))

    dec.clean_mem()
    with torch.no_grad():
        y = probe_call(x, parallel=False, cond=cond)
    for h in hooks:
        h.remove()

    print(f"\nlatent {tuple(x.shape)} + cond {tuple(cond.shape)} -> rgb {tuple(y.shape)}")
    for s in shapes[:6]:
        print(f"  layer {s['idx']:2d} {s['type']:16s} in {s['in']} -> out {s['out']}")
    print("  ...")

    torch.save({"x": x.float().cpu(), "cond": cond.float().cpu(), "y": y.float().cpu(),
                "shapes": shapes, "layers": layers,
                "pixel_shuffle": [dec.pixel_shuffle.ff, dec.pixel_shuffle.hh, dec.pixel_shuffle.ww],
                "frames_to_trim": dec.frames_to_trim},
               os.path.join(OUT, "ref.pt"))
    with open(os.path.join(OUT, "spec.json"), "w", encoding="utf-8") as f:
        json.dump({"layers": layers, "shapes": shapes,
                   "pixel_shuffle": [dec.pixel_shuffle.ff, dec.pixel_shuffle.hh,
                                     dec.pixel_shuffle.ww],
                   "frames_to_trim": dec.frames_to_trim}, f, ensure_ascii=False, indent=1)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
