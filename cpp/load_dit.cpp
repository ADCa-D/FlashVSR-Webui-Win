// load_dit.cpp -- C++ safetensors loader for the FlashVSR DiT (1419 M params).
//
// Why this exists: the framework path (HF from_pretrained + .to(cuda)) peaks at
// "f32 copy + bf16 copy + GPU copy". Here we do one sequential pass over the
// file, convert 64 MB at a time into ONE reusable staging buffer, and push
// straight to VRAM. Peak RAM = staging buffer + CUDA context, peak VRAM = bf16
// weights only.
//
// Build: cpp\build.bat load_dit
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <immintrin.h>

#define NOMINMAX
#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct TensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    int64_t begin = 0, end = 0;
};

// ---- minimal JSON reader, just enough for the safetensors header ----------
struct JParser {
    const char* p;
    const char* e;

    void ws() {
        while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    }
    void expect(char c) {
        ws();
        if (p < e && *p == c) ++p;
    }
    std::string str() {
        ws();
        std::string s;
        if (p < e && *p == '"') ++p;
        while (p < e && *p != '"') {
            if (*p == '\\') ++p;
            s.push_back(*p++);
        }
        if (p < e) ++p;
        return s;
    }
    double num() {
        char* q = nullptr;
        double v = std::strtod(p, &q);
        if (q) p = q;
        return v;
    }
    void skip_val() {
        ws();
        if (p >= e) return;
        if (*p == '"') { str(); return; }
        if (*p == '[' || *p == '{') {
            char open = *p, close = (open == '[') ? ']' : '}';
            int depth = 0;
            while (p < e) {
                if (*p == '"') { str(); continue; }
                if (*p == open) ++depth;
                else if (*p == close) {
                    ++p;
                    if (--depth == 0) return;
                    continue;
                }
                ++p;
            }
            return;
        }
        while (p < e && *p != ',' && *p != '}' && *p != ']') ++p;
    }

    std::map<std::string, TensorInfo> header() {
        std::map<std::string, TensorInfo> out;
        expect('{');
        while (p < e) {
            ws();
            if (p < e && *p == ',') { ++p; continue; }
            if (*p == '}') { ++p; break; }
            std::string key = str();
            expect(':');
            ws();
            if (*p != '{') { skip_val(); continue; }   // __metadata__
            ++p;
            TensorInfo ti;
            while (p < e) {
                ws();
                if (p < e && *p == ',') { ++p; continue; }
                if (*p == '}') { ++p; break; }
                std::string f = str();
                expect(':');
                ws();
                if (f == "dtype") {
                    ti.dtype = str();
                } else if (f == "shape") {
                    expect('[');
                    while (p < e) {
                        ws();
                        if (*p == ']') { ++p; break; }
                        ti.shape.push_back((int64_t)num());
                        ws();
                        if (p < e && *p == ',') ++p;
                    }
                } else if (f == "data_offsets") {
                    expect('[');
                    ti.begin = (int64_t)num();
                    expect(',');
                    ti.end = (int64_t)num();
                    expect(']');
                } else {
                    skip_val();
                }
            }
            out[key] = ti;
        }
        return out;
    }
};

// f32 -> bf16 with round-to-nearest-even (same rule as torch's .to(bfloat16))
inline void f32_to_bf16_rne(const float* in, uint16_t* out, int64_t n) {
#if defined(__AVX2__)
    const __m256i bias = _mm256_set1_epi32(0x7FFF);
    const __m256i one = _mm256_set1_epi32(1);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 f = _mm256_loadu_ps(in + i);
        __m256i u = _mm256_castps_si256(f);
        __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), one);
        u = _mm256_add_epi32(u, _mm256_add_epi32(bias, lsb));
        u = _mm256_srli_epi32(u, 16);
        __m128i lo = _mm256_castsi256_si128(u);
        __m128i hi = _mm256_extracti128_si256(u, 1);
        _mm_storeu_si128((__m128i*)(out + i), _mm_packus_epi32(lo, hi));
    }
#else
    int64_t i = 0;
#endif
    for (; i < n; ++i) {
        uint32_t u;
        std::memcpy(&u, in + i, 4);
        u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16;
        out[i] = (uint16_t)u;
    }
}

size_t working_set_mib() {
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc)))
        return pmc.WorkingSetSize / 1048576;
    return 0;
}

size_t private_mib(bool peak) {
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc)))
        return (peak ? pmc.PeakPagefileUsage : pmc.PrivateUsage) / 1048576;
    return 0;
}

struct Entry {
    std::string key;
    TensorInfo info;
};

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (!LoadLibraryA("torch_cuda.dll")) {
        std::printf("LoadLibrary(torch_cuda.dll) failed: %lu\n", GetLastError());
        return 1;
    }
    const char* path =
        (argc > 1) ? argv[1]
                   : "D:\\ComfyUI_windows_portable\\ComfyUI\\models\\FlashVSR-v1.1"
                     "\\diffusion_pytorch_model_streaming_dmd.safetensors";
    if (!torch::cuda::is_available()) {
        std::printf("CUDA not available\n");
        return 1;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    FILE* fp = std::fopen(path, "rb");
    if (!fp) {
        std::printf("open failed\n");
        return 1;
    }
    uint64_t hlen = 0;
    if (std::fread(&hlen, 8, 1, fp) != 1) {
        std::printf("read header length failed\n");
        return 1;
    }
    std::string hdr((size_t)hlen, '\0');
    if (std::fread(hdr.data(), 1, (size_t)hlen, fp) != hlen) {
        std::printf("read header failed\n");
        return 1;
    }
    JParser jp{hdr.c_str(), hdr.c_str() + hdr.size()};
    auto infos = jp.header();

    std::vector<Entry> entries;
    entries.reserve(infos.size());
    int64_t data_begin = -1;
    for (auto& kv : infos) entries.push_back(Entry{kv.first, kv.second});
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) { return a.info.begin < b.info.begin; });
    for (auto& e : entries) data_begin = (data_begin < 0) ? e.info.begin : std::min(data_begin, e.info.begin);
    if (data_begin != 0) std::printf("warning: first tensor starts at %lld\n", (long long)data_begin);

    std::printf("file   : %.0f MiB, header %.0f KiB, %zu tensors  (commit %.0f MiB)\n",
                (double)hlen, hlen / 1024.0, entries.size(), (double)private_mib(false));

    // ---- one sequential pass, one reusable staging buffer -------------------
    constexpr int64_t CHUNK = 16ll * 1024 * 1024;   // floats per step (64 MB in, 32 MB out)
    auto opt_gpu_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    std::vector<float> stage_f32((size_t)CHUNK);
    std::vector<uint16_t> stage_bf16((size_t)CHUNK);

    std::map<std::string, torch::Tensor> weights;
    int64_t total_elems = 0;
    double checksum = 0.0;
    int done = 0;

    const bool no_gpu = std::getenv("LOAD_NO_GPU") != nullptr;
    std::printf("mode   : %s\n", no_gpu ? "read-only (no GPU alloc)" : "f32->bf16 into VRAM");
    for (auto& e : entries) {
        const TensorInfo& ti = e.info;
        int64_t n = 1;
        for (auto d : ti.shape) n *= d;
        total_elems += n;

        auto dst_flat = no_gpu ? torch::Tensor() : torch::empty({n}, opt_gpu_bf16);
        auto* gp = no_gpu ? nullptr : (char*)dst_flat.data_ptr();

        // sizes on disk vs the bf16 target
        int64_t in_bytes = (ti.dtype == "F32") ? 4 : 2;
        if (ti.end - ti.begin != n * in_bytes) {
            std::printf("size mismatch on %s\n", e.key.c_str());
            return 1;
        }
        for (int64_t off = 0; off < n; off += CHUNK) {
            int64_t cnt = std::min(CHUNK, n - off);
            if (std::fread(stage_f32.data(), (size_t)in_bytes, (size_t)cnt, fp) != (size_t)cnt) {
                std::printf("short read on %s\n", e.key.c_str());
                return 1;
            }
            if (ti.dtype == "F32") {
                f32_to_bf16_rne(stage_f32.data(), stage_bf16.data(), cnt);
                if (!no_gpu)
                    cudaMemcpy(gp + off * 2, stage_bf16.data(), (size_t)cnt * 2, cudaMemcpyHostToDevice);
            } else if (ti.dtype == "BF16") {
                if (!no_gpu)
                    cudaMemcpy(gp + off * 2, stage_f32.data(), (size_t)cnt * 2, cudaMemcpyHostToDevice);
            } else {
                std::printf("unsupported dtype %s\n", ti.dtype.c_str());
                return 1;
            }
        }
        if (!no_gpu)
            checksum += dst_flat.slice(0, 0, std::min<int64_t>(1024, n))
                            .to(torch::kFloat32)
                            .sum()
                            .item<float>();
        if (!no_gpu) weights.emplace(e.key, dst_flat.reshape(ti.shape));
        if ((++done % 400) == 0)
            std::printf("  .. %d tensors: commit %.0f MiB\n", done, (double)private_mib(false));
    }
    std::fclose(fp);
    cudaDeviceSynchronize();

    auto t_end = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t_end - t_start).count();

    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    auto* alloc = c10::cuda::CUDACachingAllocator::get();
    size_t reserved = 0, allocated = 0;
    if (!no_gpu && alloc) {
        reserved = alloc->getDeviceStats(0).reserved_bytes[0].current;
        allocated = alloc->getDeviceStats(0).allocated_bytes[0].current;
    }

    std::printf("loaded : %.1f M params as bf16 in %.2f s (%.0f MiB/s)\n", total_elems / 1e6,
                sec, total_elems * 2 / 1048576.0 / sec);
    std::printf("vram   : allocated %.0f MiB / reserved %.0f MiB / device peak %.0f MiB\n",
                allocated / 1048576.0, reserved / 1048576.0, (total_b - free_b) / 1048576.0);
    std::printf("ram    : working set %.0f MiB\n", (double)working_set_mib());
    std::printf("ram    : private commit %.0f MiB / peak %.0f MiB\n", (double)private_mib(false),
                (double)private_mib(true));
    std::printf("checksum(first 1024 of each tensor) = %.6f\n", checksum);
    return 0;
}
