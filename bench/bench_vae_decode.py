# -*- coding: utf-8 -*-
"""VAE 解码速度：旧实现 vs 新实现（生产设置：TF32 默认开、bf16 权重）。

latent [1,16,T,H,W] → decode → 视频。取中位数，并记录峰值显存。
"""
import importlib
import os
import statistics
import sys
import time
import types

MODEL = r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast\src\models"
CKPT = r"D:\ComfyUI_windows_portable\ComfyUI\models\FlashVSR-v1.1\Wan2.1_VAE.pth"

sys.path.insert(0, r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes")
pkg = types.ModuleType("fvnode")
pkg.__path__ = [r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast"]
sys.modules["fvnode"] = pkg

import torch  # noqa: E402

NEW = importlib.import_module("fvnode.src.models.wan_video_vae")


def load_orig():
    tmp = os.path.join(MODEL, "_vae_orig.py")
    with open(os.path.join(MODEL, "wan_video_vae.py.orig"), "r", encoding="utf-8") as f:
        s = f.read()
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(s)
    return importlib.import_module("fvnode.src.models._vae_orig")


def build(mod, sd, dtype):
    vae = mod.WanVideoVAE(z_dim=16, dim=96)
    vae.model.load_state_dict(sd, strict=False)
    return vae.to(dtype=dtype).cuda().eval()


def bench(fn, n=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(n):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / n


def main():
    orig = load_orig()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    dtype = torch.bfloat16
    v_new = build(NEW, sd, dtype)
    v_old = build(orig, sd, dtype)

    for T, hw in ((9, 32), (9, 40)):
        z = torch.randn(1, 16, T, hw, hw, device="cuda", dtype=dtype)
        with torch.no_grad():
            torch.cuda.reset_peak_memory_stats()
            t_new = bench(lambda: NEW.WanVideoVAE.decode.__get__(v_new)([z[0]], "cuda"))
            pk_new = torch.cuda.max_memory_allocated() / 2 ** 20
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t_old = bench(lambda: orig.WanVideoVAE.decode.__get__(v_old)([z[0]], "cuda"))
            pk_old = torch.cuda.max_memory_allocated() / 2 ** 20
            torch.cuda.empty_cache()
        out_w = hw * 8
        print("latent %dx%dx%d → 输出约 %dpx：old %.1f ms / %.0f MiB   new %.1f ms / %.0f MiB   %+.1f%%"
              % (T, hw, hw, out_w, t_old, pk_old, t_new, pk_new, 100 * (t_new / t_old - 1)))

    os.remove(os.path.join(MODEL, "_vae_orig.py"))


if __name__ == "__main__":
    main()
