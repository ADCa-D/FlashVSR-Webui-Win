# -*- coding: utf-8 -*-
"""面板（Python）这条路加载模型时，host 内存峰值到底是多少。

决定"要不要做流式加载"的唯一依据：现在是多少、C++ 那条是 542 MiB。
"""
import ctypes
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _pmc():
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
    p = PMC()
    p.cb = ctypes.sizeof(PMC)
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(p), p.cb)
    return p


def ws_mib():
    p = _pmc()
    return p.WorkingSetSize / 2 ** 20, p.PeakWorkingSetSize / 2 ** 20


def commit_mib():
    p = _pmc()
    return p.PrivateUsage / 2 ** 20, p.PeakPagefileUsage / 2 ** 20


def main():
    import torch
    import engine as E

    cur, peak = ws_mib()
    cc, cp = commit_mib()
    print(f"启动后: WS {cur:.0f} / peak {peak:.0f} MiB   commit {cc:.0f} / peak {cp:.0f} MiB")
    t0 = time.perf_counter()
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
    dt = time.perf_counter() - t0
    cur, peak = ws_mib()
    cc, cp = commit_mib()
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"加载完: {dt:.1f}s")
    print(f"  WS     {cur:.0f} / peak {peak:.0f} MiB   （含 mmap 的文件页，可回收）")
    print(f"  commit {cc:.0f} / peak {cp:.0f} MiB   （真实占用，含显存）")
    print(f"  显存   {(total_b - free_b) / 2 ** 20:.0f} MiB")


if __name__ == "__main__":
    sys.exit(main())
