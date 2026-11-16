// cuda_jit.h -- compile CUDA C at runtime with the NVRTC that torch already
// ships, then launch it through the driver API. No CUDA Toolkit, no nvcc:
// everything is loaded with LoadLibrary + GetProcAddress, so the only runtime
// dependency is what torch itself needs anyway.
#pragma once

#define NOMINMAX
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace cj {

// ---- driver API (declared by hand: we have no cuda.h, and the enums we need
// are stable ABI) -----------------------------------------------------------
using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using CUdeviceptr = unsigned long long;

using PFN_cuCtxGetDevice = CUresult (*)(CUdevice*);
using PFN_cuDeviceGetAttribute = CUresult (*)(int*, int, CUdevice);
using PFN_cuModuleLoadData = CUresult (*)(CUmodule*, const void*);
using PFN_cuModuleGetFunction = CUresult (*)(CUfunction*, CUmodule, const char*);
using PFN_cuLaunchKernel = CUresult (*)(CUfunction, unsigned, unsigned, unsigned, unsigned,
                                        unsigned, unsigned, unsigned, CUstream, void**, void**);
using PFN_cuCtxSynchronize = CUresult (*)();
using PFN_cuModuleUnload = CUresult (*)(CUmodule);
using PFN_cuGetErrorName = CUresult (*)(CUresult, const char**);

// CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / MINOR
constexpr int kAttrMajor = 75;
constexpr int kAttrMinor = 76;

// ---- NVRTC -----------------------------------------------------------------
using nvrtcProgram = void*;
using PFN_nvrtcCreateProgram = int (*)(nvrtcProgram*, const char*, const char*, int,
                                       const char* const*, const char* const*);
using PFN_nvrtcCompileProgram = int (*)(nvrtcProgram, int, const char* const*);
using PFN_nvrtcGetCUBINSize = int (*)(nvrtcProgram, size_t*);
using PFN_nvrtcGetCUBIN = int (*)(nvrtcProgram, char*);
using PFN_nvrtcGetPTXSize = int (*)(nvrtcProgram, size_t*);
using PFN_nvrtcGetPTX = int (*)(nvrtcProgram, char*);
using PFN_nvrtcGetProgramLogSize = int (*)(nvrtcProgram, size_t*);
using PFN_nvrtcGetProgramLog = int (*)(nvrtcProgram, char*);
using PFN_nvrtcDestroyProgram = int (*)(nvrtcProgram*);

struct Driver {
    HMODULE dll = nullptr;
    PFN_cuCtxGetDevice ctx_get_device = nullptr;
    PFN_cuDeviceGetAttribute dev_attr = nullptr;
    PFN_cuModuleLoadData module_load = nullptr;
    PFN_cuModuleGetFunction module_get_fn = nullptr;
    PFN_cuLaunchKernel launch = nullptr;
    PFN_cuCtxSynchronize sync = nullptr;
    PFN_cuModuleUnload module_unload = nullptr;
    PFN_cuGetErrorName err_name = nullptr;

    PFN_nvrtcCreateProgram nvrtc_create = nullptr;
    PFN_nvrtcCompileProgram nvrtc_compile = nullptr;
    PFN_nvrtcGetCUBINSize nvrtc_cubin_size = nullptr;
    PFN_nvrtcGetCUBIN nvrtc_cubin = nullptr;
    PFN_nvrtcGetPTXSize nvrtc_ptx_size = nullptr;
    PFN_nvrtcGetPTX nvrtc_ptx = nullptr;
    PFN_nvrtcGetProgramLogSize nvrtc_log_size = nullptr;
    PFN_nvrtcGetProgramLog nvrtc_log = nullptr;
    PFN_nvrtcDestroyProgram nvrtc_destroy = nullptr;
};

inline Driver& driver() {
    static Driver d;
    return d;
}

template <typename T>
inline bool grab(HMODULE h, const char* name, T& out) {
    out = (T)GetProcAddress(h, name);
    if (!out) std::printf("[cj] missing symbol %s\n", name);
    return out != nullptr;
}

inline bool init() {
    Driver& d = driver();
    if (d.dll) return true;

    d.dll = LoadLibraryA("nvcuda.dll");
    if (!d.dll) {
        std::printf("[cj] nvcuda.dll not loadable (%lu)\n", GetLastError());
        return false;
    }
    bool ok = true;
    ok &= grab(d.dll, "cuCtxGetDevice", d.ctx_get_device);
    ok &= grab(d.dll, "cuDeviceGetAttribute", d.dev_attr);
    ok &= grab(d.dll, "cuModuleLoadData", d.module_load);
    ok &= grab(d.dll, "cuModuleGetFunction", d.module_get_fn);
    ok &= grab(d.dll, "cuLaunchKernel", d.launch);
    ok &= grab(d.dll, "cuCtxSynchronize", d.sync);
    ok &= grab(d.dll, "cuModuleUnload", d.module_unload);
    ok &= grab(d.dll, "cuGetErrorName", d.err_name);

    HMODULE nv = LoadLibraryA("nvrtc64_130_0.dll");
    if (!nv) nv = LoadLibraryA("nvrtc64_120_0.dll");
    if (!nv) {
        std::printf("[cj] nvrtc dll not loadable (%lu)\n", GetLastError());
        return false;
    }
    ok &= grab(nv, "nvrtcCreateProgram", d.nvrtc_create);
    ok &= grab(nv, "nvrtcCompileProgram", d.nvrtc_compile);
    ok &= grab(nv, "nvrtcGetCUBINSize", d.nvrtc_cubin_size);
    ok &= grab(nv, "nvrtcGetCUBIN", d.nvrtc_cubin);
    ok &= grab(nv, "nvrtcGetPTXSize", d.nvrtc_ptx_size);
    ok &= grab(nv, "nvrtcGetPTX", d.nvrtc_ptx);
    ok &= grab(nv, "nvrtcGetProgramLogSize", d.nvrtc_log_size);
    ok &= grab(nv, "nvrtcGetProgramLog", d.nvrtc_log);
    ok &= grab(nv, "nvrtcDestroyProgram", d.nvrtc_destroy);
    if (!ok) d.dll = nullptr;
    return ok;
}

// Compute capability of the device torch is already using (context exists).
inline bool arch(int* major, int* minor) {
    Driver& d = driver();
    CUdevice dev = 0;
    if (d.ctx_get_device(&dev) != 0) return false;
    if (d.dev_attr(major, kAttrMajor, dev) != 0) return false;
    if (d.dev_attr(minor, kAttrMinor, dev) != 0) return false;
    return true;
}

struct Module {
    CUmodule mod = nullptr;
    CUfunction fn = nullptr;
};

// Compile `src` for `entry`, targeting the current device's SM.
inline bool compile(const char* src, const char* entry, Module* out, bool verbose = true) {
    Driver& d = driver();
    if (!init()) return false;
    int maj = 0, min = 0;
    if (!arch(&maj, &min)) {
        std::printf("[cj] cannot query compute capability\n");
        return false;
    }
    const std::string gpu = "--gpu-architecture=sm_" + std::to_string(maj) + std::to_string(min);
    // fast-math off deliberately: we want the same fp32 rounding as the reference
    const char* opts[] = {gpu.c_str(), "--std=c++17", "-default-device",
                          "--device-int128"};

    nvrtcProgram prog = nullptr;
    if (d.nvrtc_create(&prog, src, "kernel.cu", 0, nullptr, nullptr) != 0) {
        std::printf("[cj] nvrtcCreateProgram failed\n");
        return false;
    }
    const int rc = d.nvrtc_compile(prog, 4, opts);
    size_t logsz = 0;
    if (d.nvrtc_log_size(prog, &logsz) == 0 && logsz > 1) {
        std::string log(logsz, '\0');
        d.nvrtc_log(prog, log.data());
        if (verbose || rc != 0) std::printf("[nvrtc] %s\n", log.c_str());
    }
    if (rc != 0) {
        std::printf("[cj] nvrtcCompileProgram failed (%d) for %s\n", rc, gpu.c_str());
        d.nvrtc_destroy(&prog);
        return false;
    }

    size_t binsz = 0;
    if (d.nvrtc_cubin_size(prog, &binsz) != 0) {
        std::printf("[cj] no cubin (need --gpu-architecture=sm_XX)\n");
        d.nvrtc_destroy(&prog);
        return false;
    }
    std::vector<char> bin(binsz);
    d.nvrtc_cubin(prog, bin.data());
    d.nvrtc_destroy(&prog);

    if (d.module_load(&out->mod, bin.data()) != 0) {
        std::printf("[cj] cuModuleLoadData failed (cubin %zu bytes)\n", binsz);
        return false;
    }
    if (d.module_get_fn(&out->fn, out->mod, entry) != 0) {
        std::printf("[cj] cuModuleGetFunction(%s) failed\n", entry);
        return false;
    }
    return true;
}

inline bool launch(Module& m, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                   unsigned bz, unsigned shmem, void** args, const char* tag) {
    Driver& d = driver();
    const CUresult r = d.launch(m.fn, gx, gy, gz, bx, by, bz, shmem, nullptr, args, nullptr);
    if (r != 0) {
        const char* name = "?";
        if (d.err_name) d.err_name(r, &name);
        std::printf("[cj] launch %s failed: %s (%d)\n", tag, name, r);
        return false;
    }
    return true;
}

}  // namespace cj
