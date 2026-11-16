// isa_dump.cpp -- raw CPUID / XCR0 dump, so we can see what the OS really exposes.
#include <intrin.h>

#include <cstdio>
#include <cstring>

int main() {
    int regs[4] = {0, 0, 0, 0};

    __cpuid(regs, 0);
    char vendor[13] = {0};
    std::memcpy(vendor, &regs[1], 4);
    std::memcpy(vendor + 4, &regs[3], 4);
    std::memcpy(vendor + 8, &regs[2], 4);
    const int max_leaf = regs[0];
    std::printf("vendor=%s  max_leaf=%d\n", vendor, max_leaf);

    __cpuid(regs, 1);
    std::printf("leaf1 : EAX=%08X EBX=%08X ECX=%08X EDX=%08X\n", regs[0], regs[1], regs[2],
                regs[3]);
    const bool osxsave = (regs[2] >> 27) & 1;
    const bool avx_cpu = (regs[2] >> 28) & 1;
    const bool hypervisor = (regs[2] >> 31) & 1;
    std::printf("        OSXSAVE=%d AVX=%d HYPERVISOR_PRESENT=%d\n", (int)osxsave, (int)avx_cpu,
                (int)hypervisor);

    if (hypervisor) {
        char hv[13] = {0};
        __cpuid(regs, 0x40000000);
        std::memcpy(hv, &regs[1], 4);
        std::memcpy(hv + 4, &regs[2], 4);
        std::memcpy(hv + 8, &regs[3], 4);
        std::printf("        hypervisor vendor = %s\n", hv);
    }

    if (osxsave) {
        const unsigned long long xcr0 = _xgetbv(0);
        std::printf("        XCR0=0x%llX  (ymm=%d zmm_state=%d)\n", xcr0,
                    (int)((xcr0 & 0x6) == 0x6), (int)((xcr0 & 0xE6) == 0xE6));
    } else {
        std::printf("        XCR0: OSXSAVE off, _xgetbv would fault -- skipped\n");
    }

    if (max_leaf >= 7) {
        __cpuidex(regs, 7, 0);
        std::printf("leaf7.0: EAX=%08X EBX=%08X ECX=%08X EDX=%08X\n", regs[0], regs[1], regs[2],
                    regs[3]);
        std::printf("        AVX2(EBX.5)=%d AVX512F(EAX.16)=%d AVX512BW(EAX.30)=%d "
                    "AVX512CD(EAX.28)=%d AVX512VL(EBX.31)=%d\n",
                    (int)((regs[1] >> 5) & 1), (int)((regs[0] >> 16) & 1),
                    (int)((regs[0] >> 30) & 1), (int)((regs[0] >> 28) & 1),
                    (int)((regs[1] >> 31) & 1));
        std::printf("        AVX512_VNNI(ECX.11)=%d AVX512_VBMI(ECX.1)=%d\n",
                    (int)((regs[2] >> 11) & 1), (int)((regs[2] >> 1) & 1));

        __cpuidex(regs, 7, 1);
        std::printf("leaf7.1: EAX=%08X EBX=%08X ECX=%08X EDX=%08X  (AVX512_BF16 EAX.5=%d)\n",
                    regs[0], regs[1], regs[2], regs[3], (int)((regs[0] >> 5) & 1));
    }

    // extended leaves: brand string lives at 0x80000002..0x80000004
    int regs2[4] = {0, 0, 0, 0};
    __cpuid(regs2, 0x80000000);
    std::printf("ext max_leaf = %08X\n", regs2[0]);
    if ((unsigned)regs2[0] >= 0x80000004u) {
        char brand[49] = {0};
        for (int i = 0; i < 3; ++i) {
            __cpuidex(regs2, 0x80000002 + i, 0);
            std::memcpy(brand + i * 16, regs2, 16);
        }
        std::printf("brand  = %s\n", brand);
    }
    return 0;
}
