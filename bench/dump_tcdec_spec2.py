# -*- coding: utf-8 -*-
"""从**真实流水线**里扒 TCDecoder 的输入/输出形状与逐层形状（合成输入对不上形状）。

存：work/tcdec/ref.pt（首次 decode 的 x / cond / rgb / 逐层形状）
    work/tcdec/spec.json（层类型与参数）
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import engine as E  # noqa: E402

SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_long_544.mp4"
OUT = os.path.join(ROOT, "work", "tcdec")
FRAMES = 25
CS = 256


def describe(mod, idx):
    t = type(mod).__name__
    d = {"idx": idx, "type": t}
    if hasattr(mod, "in_channels"):
        d.update({"in": mod.in_channels, "out": mod.out_channels,
                  "k": list(mod.kernel_size) if hasattr(mod.kernel_size, "__len__") else mod.kernel_size,
                  "stride": list(mod.stride) if hasattr(mod.stride, "__len__") else mod.stride,
                  "padding": list(mod.padding) if hasattr(mod.padding, "__len__") else mod.padding,
                  "bias": mod.bias is not None})
    if t == "MemBlock":
        d.update({"n_in": mod.conv[0].in_channels // 2, "n_out": mod.conv[-1].out_channels,
                  "skip": type(mod.skip).__name__})
    if t in ("TPool", "TGrow"):
        d.update({"stride_n": mod.stride, "conv_in": mod.conv.in_channels,
                  "conv_out": mod.conv.out_channels})
    if t == "Upsample":
        d["scale"] = mod.scale_factor
    if t == "PixelShuffle3d":
        d.update({"ff": mod.ff, "hh": mod.hh, "ww": mod.ww})
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
    pipe = eng.pipe
    mod = eng.mod
    dec = pipe.TCDecoder

    layers = [describe(m, i) for i, m in enumerate(dec.decoder)]
    print(f"decoder 层数 {len(layers)}  frames_to_trim={dec.frames_to_trim}")
    for L in layers:
        extra = ""
        if "in" in L:
            extra = f"in={L['in']} out={L['out']} k={L['k']} s={L['stride']} pad={L['padding']} bias={L['bias']}"
        elif L["type"] == "MemBlock":
            extra = f"n_in={L['n_in']} n_out={L['n_out']} skip={L['skip']}"
        elif "conv_in" in L:
            extra = f"stride={L['stride_n']} conv {L['conv_in']}->{L['conv_out']}"
        elif L["type"] == "Upsample":
            extra = f"scale={L['scale']}"
        print(f"  {L['idx']:2d} {L['type']:16s} {extra}")

    cap = {}

    def mk(i):
        def fn(m, inp, out):
            if cap.get("done"):
                return
            a = inp[0] if isinstance(inp, (tuple, list)) else inp
            if torch.is_tensor(a):
                cap.setdefault("shapes", []).append(
                    {"idx": i, "type": type(m).__name__, "in": list(a.shape),
                     "out": list(out.shape) if torch.is_tensor(out) else None})
        return fn

    hooks = [m.register_forward_hook(mk(i)) for i, m in enumerate(dec.decoder)]

    orig = dec.decode_video

    def hooked(x, parallel=True, show_progress_bar=False, cond=None):
        out = orig(x, parallel=parallel, show_progress_bar=show_progress_bar, cond=cond)
        if not cap.get("done"):
            cap["done"] = True
            cap["x"] = x.detach().float().cpu()
            cap["cond"] = cond.detach().float().cpu() if cond is not None else None
            cap["y"] = out.detach().float().cpu()
            cap["parallel"] = parallel
        return out

    dec.decode_video = hooked

    u8 = E.decode_frames_range(SRC, 0, FRAMES, pre_scale=1.0, fps=25.0)
    n0 = u8.shape[0]
    y0, x0 = (u8.shape[1] - CS) // 2, (u8.shape[2] - CS) // 2
    u8 = u8[:, y0:y0 + CS, x0:x0 + CS, :]
    frames = torch.from_numpy(np.ascontiguousarray(u8)).float().div_(255.0)
    add_n = mod.next_8n5(n0) - n0
    if add_n > 0:
        frames = torch.cat([frames, frames[-1:, :, :, :].repeat([add_n, 1, 1, 1])], dim=0)
    LQ, th, tw, F = mod.prepare_input_tensor(frames, pipe.device, scale=2,
                                             dtype=pipe.torch_dtype)
    with torch.no_grad():
        pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
             tiled=True, LQ_video=LQ, num_frames=F, height=th, width=tw,
             is_full_block=False, if_buffer=True,
             topk_ratio=2.0 * 768 * 1280 / (th * tw), kv_ratio=3.0, local_range=9,
             color_fix=False, unload_dit=False, force_offload=False)
    dec.decode_video = orig
    for h in hooks:
        h.remove()

    print(f"\nx {tuple(cap['x'].shape)}  cond "
          f"{tuple(cap['cond'].shape) if cap['cond'] is not None else None}"
          f"  -> y {tuple(cap['y'].shape)}  parallel={cap.get('parallel')}")
    sh = cap.get("shapes", [])
    for s in sh[:8]:
        print(f"  layer {s['idx']:2d} {s['type']:16s} in {s['in']} -> out {s['out']}")
    if len(sh) > 8:
        print("  ...")
        for s in sh[-4:]:
            print(f"  layer {s['idx']:2d} {s['type']:16s} in {s['in']} -> out {s['out']}")

    torch.save({"x": cap["x"], "cond": cap["cond"], "y": cap["y"], "shapes": sh,
                "layers": layers, "frames_to_trim": dec.frames_to_trim},
               os.path.join(OUT, "ref.pt"))
    with open(os.path.join(OUT, "spec.json"), "w", encoding="utf-8") as f:
        json.dump({"layers": layers, "shapes": sh,
                   "frames_to_trim": dec.frames_to_trim}, f, ensure_ascii=False, indent=1)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
