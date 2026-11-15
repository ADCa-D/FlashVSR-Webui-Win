# -*- coding: utf-8 -*-
"""量化后权重到底占多少显存（只加载、不推理）。用法：python bench/quant_mem.py mxfp8"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

import engine as E

q = sys.argv[1] if len(sys.argv) > 1 else "none"


def main():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", quant=q)
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 2 ** 30
    reserved = torch.cuda.memory_reserved() / 2 ** 30
    free, total = torch.cuda.mem_get_info()
    print(f"quant={q:6s} 权重+缓冲 allocated={alloc:.2f} GiB  reserved={reserved:.2f} GiB  "
          f"空闲={free/2**30:.2f}/{total/2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
