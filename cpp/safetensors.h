// safetensors.h -- streaming reader (header-only, shared by the C++ tools).
//
// One sequential pass over the file, one reusable staging buffer, f32 -> target
// dtype converted on the CPU with the runtime-dispatched SIMD kernel. No f32
// copy of the checkpoint ever exists in memory.
#pragma once

#include <torch/torch.h>
#include <cuda_runtime.h>

#include "simd/simd_bf16.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace st {

struct Info {
    std::string dtype;
    std::vector<int64_t> shape;
    int64_t begin = 0, end = 0;
};

inline int64_t numel(const std::vector<int64_t>& s) {
    int64_t n = 1;
    for (auto d : s) n *= d;
    return n;
}

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

    std::map<std::string, Info> header() {
        std::map<std::string, Info> out;
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
            Info ti;
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

// Reads the whole file into tensors of `opts` (dtype/device), streaming.
inline std::map<std::string, torch::Tensor> load_all(const char* path, torch::TensorOptions opts,
                                                     bool verbose = false) {
    std::map<std::string, torch::Tensor> out;
    FILE* fp = std::fopen(path, "rb");
    if (!fp) {
        std::printf("[st] open failed: %s\n", path);
        return out;
    }
    uint64_t hlen = 0;
    if (std::fread(&hlen, 8, 1, fp) != 1) {
        std::fclose(fp);
        std::printf("[st] bad header length\n");
        return out;
    }
    std::string hdr((size_t)hlen, '\0');
    if (std::fread(hdr.data(), 1, (size_t)hlen, fp) != hlen) {
        std::fclose(fp);
        std::printf("[st] short header\n");
        return out;
    }
    JParser jp{hdr.c_str(), hdr.c_str() + hdr.size()};
    auto infos = jp.header();

    std::vector<std::pair<int64_t, const std::string*>> order;
    order.reserve(infos.size());
    for (auto& kv : infos) order.push_back({kv.second.begin, &kv.first});
    std::sort(order.begin(), order.end(),
              [](auto& a, auto& b) { return a.first < b.first; });

    constexpr int64_t CHUNK = 16ll * 1024 * 1024;   // floats per step (64 MB in, 32 MB out)
    std::vector<float> stage32((size_t)CHUNK);
    std::vector<uint16_t> stage16((size_t)CHUNK);
    const bool want_bf16 = opts.dtype() == torch::kBFloat16;
    const simd::ConvertFn to_bf16 = want_bf16 ? simd::f32_to_bf16() : nullptr;

    int64_t done = 0;
    for (auto& e : order) {
        const std::string& key = *e.second;
        const Info& ti = infos[key];
        const int64_t n = numel(ti.shape);
        const int64_t esz = (ti.dtype == "F32") ? 4 : 2;
        if (ti.end - ti.begin != n * esz) {
            std::printf("[st] size mismatch on %s\n", key.c_str());
            std::fclose(fp);
            return out;
        }
        if (_fseeki64(fp, (long long)(8 + hlen + ti.begin), SEEK_SET) != 0) {
            std::printf("[st] seek failed on %s\n", key.c_str());
            std::fclose(fp);
            return out;
        }

        auto dst = torch::empty(ti.shape, opts);
        auto* gp = (char*)dst.data_ptr();

        for (int64_t off = 0; off < n; off += CHUNK) {
            const int64_t cnt = std::min(CHUNK, n - off);
            if (std::fread(stage32.data(), (size_t)esz, (size_t)cnt, fp) != (size_t)cnt) {
                std::printf("[st] short read on %s\n", key.c_str());
                std::fclose(fp);
                return out;
            }
            if (ti.dtype == "F32" && want_bf16) {
                to_bf16(stage32.data(), stage16.data(), cnt);
                cudaMemcpy(gp + off * 2, stage16.data(), (size_t)cnt * 2, cudaMemcpyHostToDevice);
            } else if (ti.dtype == "F32") {
                cudaMemcpy(gp + off * 4, stage32.data(), (size_t)cnt * 4, cudaMemcpyHostToDevice);
            } else if (ti.dtype == "BF16" && want_bf16) {
                cudaMemcpy(gp + off * 2, stage32.data(), (size_t)cnt * 2, cudaMemcpyHostToDevice);
            } else {
                std::printf("[st] dtype %s -> target not supported\n", ti.dtype.c_str());
                std::fclose(fp);
                return out;
            }
        }
        out.emplace(key, dst);
        if (verbose && (++done % 200) == 0) std::printf("  [st] %lld tensors\n", (long long)done);
    }
    std::fclose(fp);
    return out;
}

}  // namespace st
