# -*- coding: utf-8 -*-
"""把 ref.pt 转成 safetensors，供 C++ 端用同一个流式读取器读。

只保留张量（config/seed/topk 这类标量另写到 ref_meta.json）。
"""
import json
import os

import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "work", "dit_ref")


def main():
    ref = torch.load(os.path.join(DIR, "ref.pt"), map_location="cpu", weights_only=False)
    tensors, meta = {}, {}
    for k, v in ref.items():
        if torch.is_tensor(v):
            tensors[k] = v.detach().float().contiguous()
        else:
            meta[k] = v
    save_file(tensors, os.path.join(DIR, "ref.safetensors"))
    with open(os.path.join(DIR, "ref_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("tensors:", len(tensors), sorted(tensors)[:6], "...")
    for k in ("x", "x_patch", "y", "mask"):
        if k in tensors:
            print(f"  {k:8s} {tuple(tensors[k].shape)} {tensors[k].dtype}")
    print("saved:", os.path.join(DIR, "ref.safetensors"))


if __name__ == "__main__":
    main()
