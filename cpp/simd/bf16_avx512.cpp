// AVX-512 path (F + BW): 16 floats per step, 32->16 pack in one instruction.
// Compiled with /arch:AVX512; only reachable after the dispatcher checks
// AVX512F + AVX512BW + XCR0 opmask/zmm state.
#include "simd_bf16.h"

#include <immintrin.h>

#include <cstring>

namespace simd {

void f32_to_bf16_avx512(const float* in, uint16_t* out, int64_t n) {
    const __m512i bias = _mm512_set1_epi32(0x7FFF);
    const __m512i one = _mm512_set1_epi32(1);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512i u = _mm512_castps_si512(_mm512_loadu_ps(in + i));
        __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(u, 16), one);
        u = _mm512_srli_epi32(_mm512_add_epi32(u, _mm512_add_epi32(bias, lsb)), 16);
        _mm256_storeu_si256((__m256i*)(out + i), _mm512_cvtepi32_epi16(u));
    }
    for (; i < n; ++i) {
        uint32_t u;
        std::memcpy(&u, in + i, 4);
        u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16;
        out[i] = (uint16_t)u;
    }
}

}  // namespace simd
