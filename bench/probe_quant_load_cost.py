# -*- coding: utf-8 -*-
"""量化到底占了多少加载时间：quant=none vs quant=mxfp8 的加载耗时对比。"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import engine as E  # noqa: E402

which = sys.argv[1] if len(sys.argv) > 1 else "mxfp8"
t0 = time.perf_counter()
eng = E.Engine()
eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant=which)
dt = time.perf_counter() - t0
import torch
free_b, total_b = torch.cuda.mem_get_info()
print(f"RESULT quant={which:6s} 加载 {dt:6.2f}s  显存 {(total_b-free_b)/2**20:.0f} MiB")
