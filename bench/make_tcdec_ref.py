# -*- coding: utf-8 -*-
"""把 work/tcdec/ref.pt 转成 safetensors（C++ 端用同一个读取器）。"""
import os
import sys

import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "work", "tcdec")


def main():
    ref = torch.load(os.path.join(D, "ref.pt"), map_location="cpu", weights_only=False)
    out = {}
    for k in ("x", "cond", "y"):
        if ref.get(k) is not None:
            out[k] = ref[k].float().contiguous()
    save_file(out, os.path.join(D, "ref.safetensors"))

    # 权重也转成 safetensors：C++ 的 InputArchive 没有枚举 key 的接口
    ckpt = r"D:\ComfyUI_windows_portable\ComfyUI\models\FlashVSR-v1.1\TCDecoder.ckpt"
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    w = {k: v.float().contiguous() for k, v in sd.items()}
    save_file(w, os.path.join(D, "weights.safetensors"))
    print(f"  weights: {len(w)} tensors")

    for k, v in out.items():
        print(f"  {k:5s} {tuple(v.shape)}")
    print("saved:", os.path.join(D, "ref.safetensors"))


if __name__ == "__main__":
    sys.exit(main())
