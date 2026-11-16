// bench_bf16.cpp -- is the f32->bf16 conversion worth hand-tuning at all?
//
// Runs each kernel on the same 64 MB buffer, checks they agree bit-for-bit, and
// reports GB/s of input consumed. Compare that against the loader's end-to-end
// ~1.3 GiB/s to see where the time actually goes.
#include "simd/simd_bf16.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

void run(const char* tag, simd::ConvertFn fn, const std::vector<float>& src,
         std::vector<uint16_t>& out, const std::vector<uint16_t>& ref, int iters) {
    fn(src.data(), out.data(), (int64_t)src.size());   // warm up
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn(src.data(), out.data(), (int64_t)src.size());
    auto t1 = std::chrono::high_resolution_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    const double gbs = (double)src.size() * 4 * iters / sec / 1e9;
    const bool same = std::memcmp(ref.data(), out.data(), (size_t)src.size() * 2) == 0;
    std::printf("  %-14s %7.1f GB/s   %6.2f ms/次   一致:%s\n", tag, gbs, sec / iters * 1e3,
                same ? "yes" : "NO");
}

}  // namespace

int main() {
    const simd::Isa best = simd::detected_isa();
    std::printf("[isa] detected=%s\n", simd::isa_name(best));

    const int64_t n = 16ll * 1024 * 1024;   // 64 MB of f32 -> 32 MB of bf16
    std::vector<float> src((size_t)n);
    for (int64_t i = 0; i < n; ++i) src[(size_t)i] = (float)((i % 4096) - 2048) * 0.00037f;

    std::vector<uint16_t> ref((size_t)n), out((size_t)n);
    simd::f32_to_bf16_base(src.data(), ref.data(), n);

    const int iters = 20;
    std::printf("转换 %lld M 元素（读 64 MB / 写 32 MB）× %d 次\n", (long long)(n / 1000000), iters);
    run("base", &simd::f32_to_bf16_base, src, out, ref, iters);
    if (best != simd::Isa::Base) run("avx2", &simd::f32_to_bf16_avx2, src, out, ref, iters);
    if (best == simd::Isa::Avx512) run("avx512", &simd::f32_to_bf16_avx512, src, out, ref, iters);
    else std::printf("  %-14s 跳过（本机 CPU 无 AVX-512）\n", "avx512");

    std::printf("\n对照：加载器端到端约 1.3 GB/s（磁盘读 + 页式 cudaMemcpy 才是瓶颈）\n");
    return 0;
}
