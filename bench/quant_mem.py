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
    # init_pipeline() 结尾会把 DiT offload 到 CPU；静态 bf16 若直接量只看到缓冲，
    # 会得出"bf16 只占 0.1 GiB"的错误结论。统一搬回 GPU 再量。
    try:
        E.dit_to_cuda(eng.pipe)
        E.dit_to_cuda(eng.pipe)  # 二次幂等，确保 LQ_proj_in 等子模块也在
    except Exception as e:  # noqa
        print(f"[warn] 搬回 GPU 失败：{e}", flush=True)
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 2 ** 30
    reserved = torch.cuda.memory_reserved() / 2 ** 30
    free, total = torch.cuda.mem_get_info()
    print(f"quant={q:6s} 权重+缓冲 allocated={alloc:.2f} GiB  reserved={reserved:.2f} GiB  "
          f"空闲={free/2**30:.2f}/{total/2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
