// Runtime ISA dispatch for the f32 -> bf16 conversion.
//
// MSVC has no per-function target attributes on x86, so each ISA level lives in
// its own translation unit compiled with its own /arch: flag; this file only
// decides which one to call. Detection is CPUID + XGETBV (the XCR0 check is not
// optional -- without it you call AVX-512 on a kernel that never saved the state
// and take #UD).
#pragma once

#include <cstdint>

namespace simd {

using ConvertFn = void (*)(const float* in, uint16_t* out, int64_t n);

enum class Isa { Base = 0, Avx2 = 1, Avx512 = 2 };

const char* isa_name(Isa isa);

// What the CPU + OS actually support (no env override applied).
Isa detected_isa();

// Picks the fastest supported kernel once. FLASHVSR_ISA=base|avx2|avx512 forces
// a level (useful for benchmarking); unsupported requests fall back with a note.
ConvertFn f32_to_bf16();

// Individual kernels (each in its own TU).
void f32_to_bf16_base(const float* in, uint16_t* out, int64_t n);
void f32_to_bf16_avx2(const float* in, uint16_t* out, int64_t n);
void f32_to_bf16_avx512(const float* in, uint16_t* out, int64_t n);

}  // namespace simd
