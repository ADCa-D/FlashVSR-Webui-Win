// CPUID / XGETBV feature detection + one-time kernel selection.
#include "simd_bf16.h"

#include <intrin.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace simd {
namespace {

struct Features {
    bool avx2 = false;
    bool avx512f = false;
    bool avx512bw = false;
    bool avx512vnni = false;
    bool avx512bf16 = false;
    bool os_ymm = false;   // XCR0: SSE + AVX state
    bool os_zmm = false;   // XCR0: + opmask / zmm_hi256 / hi16_zmm
};

Features detect_features() {
    Features f{};
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    const int max_leaf = regs[0];
    __cpuid(regs, 1);
    const bool osxsave = (regs[2] >> 27) & 1;
    const bool avx_cpu = (regs[2] >> 28) & 1;

    if (osxsave) {
        const unsigned long long xcr0 = _xgetbv(0);
        f.os_ymm = (xcr0 & 0x6) == 0x6;
        f.os_zmm = (xcr0 & 0xE6) == 0xE6;
    }

    if (max_leaf >= 7) {
        __cpuidex(regs, 7, 0);
        f.avx2 = ((regs[1] >> 5) & 1) && avx_cpu && f.os_ymm;        // EBX.5
        f.avx512f = ((regs[0] >> 16) & 1) && f.os_zmm;               // EAX.16
        f.avx512bw = ((regs[0] >> 30) & 1) && f.os_zmm;              // EAX.30
        f.avx512vnni = ((regs[2] >> 11) & 1) && f.avx512f;            // ECX.11
        __cpuidex(regs, 7, 1);
        f.avx512bf16 = ((regs[0] >> 5) & 1) && f.avx512f;             // subleaf 1, EAX.5
    }
    return f;
}

Features& features() {
    static Features f = detect_features();
    return f;
}

}  // namespace

const char* isa_name(Isa isa) {
    switch (isa) {
        case Isa::Avx512: return "avx512f+bw";
        case Isa::Avx2: return "avx2";
        default: return "base";
    }
}

Isa detected_isa() {
    const Features& f = features();
    if (f.avx512f && f.avx512bw) return Isa::Avx512;
    if (f.avx2) return Isa::Avx2;
    return Isa::Base;
}

ConvertFn f32_to_bf16() {
    static ConvertFn fn = [] {
        const Features& f = features();
        const Isa best = detected_isa();
        Isa want = best;

        if (const char* env = std::getenv("FLASHVSR_ISA")) {
            if (std::strcmp(env, "base") == 0) want = Isa::Base;
            else if (std::strcmp(env, "avx2") == 0) want = Isa::Avx2;
            else if (std::strcmp(env, "avx512") == 0) want = Isa::Avx512;
        }
        if (want == Isa::Avx512 && !(f.avx512f && f.avx512bw)) {
            std::printf("[isa] avx512 requested but unsupported, falling back to %s\n",
                        isa_name(best));
            want = best;
        }
        if (want == Isa::Avx2 && !f.avx2) {
            std::printf("[isa] avx2 requested but unsupported, falling back to %s\n",
                        isa_name(best));
            want = best;
        }
        std::printf("[isa] detected=%s chosen=%s (avx2=%d avx512f=%d avx512bw=%d vnni=%d "
                    "bf16=%d)\n",
                    isa_name(best), isa_name(want), (int)f.avx2, (int)f.avx512f,
                    (int)f.avx512bw, (int)f.avx512vnni, (int)f.avx512bf16);

        switch (want) {
            case Isa::Avx512: return &f32_to_bf16_avx512;
            case Isa::Avx2: return &f32_to_bf16_avx2;
            default: return &f32_to_bf16_base;
        }
    }();
    return fn;
}

}  // namespace simd
