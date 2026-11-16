// AVX2 path: compiled with /arch:AVX2, so it must never be called unless the
// dispatcher proved AVX2 + OS YMM state support.
#include "simd_bf16.h"

#include <immintrin.h>

#include <cstring>

namespace simd {

void f32_to_bf16_avx2(const float* in, uint16_t* out, int64_t n) {
    const __m256i bias = _mm256_set1_epi32(0x7FFF);
    const __m256i one = _mm256_set1_epi32(1);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i u = _mm256_castps_si256(_mm256_loadu_ps(in + i));
        __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), one);
        u = _mm256_srli_epi32(_mm256_add_epi32(u, _mm256_add_epi32(bias, lsb)), 16);
        __m128i packed = _mm_packus_epi32(_mm256_castsi256_si128(u), _mm256_extracti128_si256(u, 1));
        _mm_storeu_si128((__m128i*)(out + i), packed);
    }
    for (; i < n; ++i) {
        uint32_t u;
        std::memcpy(&u, in + i, 4);
        u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16;
        out[i] = (uint16_t)u;
    }
}

}  // namespace simd
