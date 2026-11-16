# -*- coding: utf-8 -*-
"""不加载模型，直接用真实视频帧验证 GPU 版 prepare 与节点版是否逐位一致。

差异会经 fp8 量化放大成输出差异，所以必须定位在 prepare 这一层。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import engine as E  # noqa: E402

SRC = os.path.join(r"D:\ComfyUI_windows_portable\ComfyUI\input", "flashvsr_long_544.mp4")
frames_u8 = E.decode_frames_range(SRC, 0, 9, pre_scale=0.5, fps=25.0)
mod = E.load_node_module()
print("node module:", getattr(mod, "__file__", "?"))


class FakeMod(object):
    compute_scaled_and_target_dims = staticmethod(mod.compute_scaled_and_target_dims)
    largest_8n1_leq = staticmethod(mod.largest_8n1_leq)
    prepare_input_tensor = staticmethod(mod.prepare_input_tensor)
    clean_vram = staticmethod(lambda: None)


fm = FakeMod()

tile, scale, ov = 128, 2, 32
H, W = frames_u8.shape[1], frames_u8.shape[2]
coords = mod.calculate_tile_coords(H, W, tile, ov)
print("frames", frames_u8.shape, "tile", tile, "ov", ov, "coords", coords[:4])

worst = 0.0
for k, (x1, y1, x2, y2) in enumerate(coords[:4]):
    tile_u8 = torch.from_numpy(np.ascontiguousarray(frames_u8[:, y1:y2, x1:x2, :]))
    ref_in = tile_u8.float().div_(255.0)
    ref, tH, tW, F = mod.prepare_input_tensor(ref_in, "cuda", scale=scale, dtype=torch.bfloat16)
    mine, tH2, tW2, F2 = E._prepare_input_gpu(fm, tile_u8, "cuda", scale=scale, dtype=torch.bfloat16)
    d = (ref.to("cuda").float() - mine.to("cuda").float()).abs()
    worst = max(worst, float(d.max()))
    print(f"tile{k} {x1},{y1}: {tuple(ref.shape)} vs {tuple(mine.shape)} "
          f"max_d={float(d.max()):.3e} mean={float(d.mean()):.3e} "
          f"tHW/F={tH},{tW},{F} vs {tH2},{tW2},{F2}", flush=True)

u8_ref, *_ = mod.prepare_input_tensor(
    torch.from_numpy(np.ascontiguousarray(frames_u8[:8, 0:tile, 0:tile, :])).float().div_(255.0),
    "cuda", scale=scale, dtype=torch.bfloat16)
u8_mine, *_ = E._prepare_input_gpu(
    fm, torch.from_numpy(np.ascontiguousarray(frames_u8[:8, 0:tile, 0:tile, :])), "cuda",
    scale=scale, dtype=torch.bfloat16)
du = (u8_ref.to("cuda").float() - u8_mine.to("cuda").float()).abs()
print(f"uint8 path: max_d={float(du.max()):.3e} mean={float(du.mean()):.3e}")
print("VERDICT:", "bit-identical" if worst == 0.0 and float(du.max()) == 0.0 else f"numeric diff {worst:.3e}")

# ---- 变体对比：找出哪一种写法与节点版逐位一致 ----
import torch.nn.functional as F  # noqa: E402

tile_u8 = torch.from_numpy(np.ascontiguousarray(frames_u8[:, 0:tile, 0:tile, :]))
ref_in = tile_u8.float().div_(255.0)
ref, tH, tW, Fn = mod.prepare_input_tensor(ref_in, "cuda", scale=scale, dtype=torch.bfloat16)
ref_c = ref.float()
sW, sH, tw2, th2 = mod.compute_scaled_and_target_dims(tile, tile, scale=scale, multiple=128)
left, top = max(0, (sW - tw2) // 2), max(0, (sH - th2) // 2)


def _crop(x):
    return x[:, :, top:top + th2, left:left + tw2]


def v_batch(cast_then_scale=True):
    src = ref_in.to("cuda")
    idx = torch.arange(Fn, device=src.device).clamp_(max=ref_in.shape[0] - 1)
    x = src.index_select(0, idx).permute(0, 3, 1, 2)
    x = _crop(F.interpolate(x, size=(sH, sW), mode="bicubic", align_corners=False))
    if cast_then_scale:
        y = x.to(torch.bfloat16) * 2.0 - 1.0
    else:
        y = (x * 2.0 - 1.0).to(torch.bfloat16)
    return y.permute(1, 0, 2, 3).unsqueeze(0)


def v_perframe(cast_then_scale=True, scale_on_cpu=False):
    parts = []
    for i in range(Fn):
        fr = ref_in[min(i, ref_in.shape[0] - 1)].to("cuda")           # (h,w,c)
        x = fr.permute(2, 0, 1).unsqueeze(0)
        x = _crop(F.interpolate(x, size=(sH, sW), mode="bicubic", align_corners=False))
        if scale_on_cpu:
            y = x.to("cpu").to(torch.bfloat16) * 2.0 - 1.0
        elif cast_then_scale:
            y = x.to(torch.bfloat16) * 2.0 - 1.0
        else:
            y = (x * 2.0 - 1.0).to(torch.bfloat16)
        parts.append(y.squeeze(0))
    return torch.stack(parts, 0).permute(1, 0, 2, 3).unsqueeze(0)


for name, fn in (("batch/bf16先乘后减", lambda: v_batch(True)),
                 ("batch/fp32先算后cast", lambda: v_batch(False)),
                 ("perframe/bf16先乘后减", lambda: v_perframe(True, False)),
                 ("perframe/fp32先算后cast", lambda: v_perframe(False, False)),
                 ("perframe/CPU上做bf16", lambda: v_perframe(True, True))):
    y = fn().float()
    d = (y - ref_c.to(y.device)).abs()
    print(f"{name:28s} max_d={float(d.max()):.3e} mean={float(d.mean()):.3e} "
          f"diff_elems={int((d > 0).sum())}/{d.numel()}", flush=True)

