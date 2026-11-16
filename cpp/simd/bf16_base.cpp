// Baseline path: plain x64, no ISA extensions. Used on anything older than
// Haswell, and as the reference implementation for the other kernels.
#include "simd_bf16.h"

#include <cstring>

namespace simd {

void f32_to_bf16_base(const float* in, uint16_t* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        uint32_t u;
        std::memcpy(&u, in + i, 4);
        u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16;   // round-to-nearest-even
        out[i] = (uint16_t)u;
    }
}

}  // namespace simd
