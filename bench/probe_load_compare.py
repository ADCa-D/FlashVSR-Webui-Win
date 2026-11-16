# -*- coding: utf-8 -*-
"""对标 C++ 加载器：Python 侧把同一个 DiT 以 bf16 装进显存，吃多少 RAM/VRAM。

和 cpp/load_dit.cpp 是同一条路径的两种写法：
  C++：mmap + 64 MB 分块 f32->bf16 直转 GPU，峰值 = bf16 权重
  Python：from_pretrained(torch_dtype=bf16) + .to(cuda)，峰值 = f32 页 + bf16 副本
"""
import ctypes
import json
import os
import struct
import sys
import time
import types

import torch

CKPT = r"D:\ComfyUI_windows_portable\ComfyUI\models\FlashVSR-v1.1\diffusion_pytorch_model_streaming_dmd.safetensors"


def priv_mib(peak=False):
    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return -1.0
    return (pmc.PeakPagefileUsage if peak else pmc.PrivateUsage) / 2**20


def ws_mib(peak=False):
    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return (pmc.PeakWorkingSetSize if peak else pmc.WorkingSetSize) / 2**20


def header():
    with open(CKPT, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main():
    hdr = header()
    keys = [k for k in hdr if k != "__metadata__"]
    print(f"tensors={len(keys)}  commit(base)={priv_mib():.0f} MiB")

    # 1) f32 -> bf16，逐张量（模拟 from_pretrained 的 torch_dtype 行为）
    t0 = time.perf_counter()
    from safetensors import safe_open
    out = {}
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        for k in keys:
            out[k] = f.get_tensor(k).to(torch.bfloat16)
    t1 = time.perf_counter()
    print(f"cpu bf16 就绪 {t1-t0:.2f}s  commit={priv_mib():.0f} / peak={priv_mib(True):.0f} MiB")
    print(f"           working set={ws_mib():.0f} / peak={ws_mib(True):.0f} MiB")

    # 2) 搬到显存
    gpu = {k: v.to("cuda") for k, v in out.items()}
    del out
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    free_b, total_b = torch.cuda.mem_get_info()
    print(f"搬到 GPU 完成 {t2-t1:.2f}s  合计 {t2-t0:.2f}s")
    print(f"vram: allocated {torch.cuda.memory_allocated()/2**20:.0f} MiB / "
          f"reserved {torch.cuda.memory_reserved()/2**20:.0f} MiB / "
          f"device peak {(total_b-free_b)/2**20:.0f} MiB")
    print(f"ram : commit={priv_mib():.0f} / peak={priv_mib(True):.0f} MiB")
    print(f"ram : working set={ws_mib():.0f} / peak={ws_mib(True):.0f} MiB")

    cs = sum(gpu[k].flatten()[:1024].float().sum().item() for k in keys)
    print(f"checksum(first 1024 of each tensor) = {cs:.6f}")


if __name__ == "__main__":
    sys.exit(main())
