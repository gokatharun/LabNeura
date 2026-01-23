#include "labneura/backends/cpu_features.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

namespace labneura {

// =======================
// CPU Feature Detection
// =======================
CPUFeatures detect_cpu_features() {
    CPUFeatures features{};

    #if defined(__x86_64__) || defined(_M_X64)
        unsigned int eax, ebx, ecx, edx;

        // Leaf 1: SSE4.1
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
            features.sse41 = (ecx & bit_SSE4_1) != 0;
        }

        // Leaf 7: AVX2
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            features.avx2 = (ebx & bit_AVX2) != 0;
        }
    #endif

    return features;
}

// =======================
// x86 AVX2 detection
// =======================
bool cpu_supports_avx2() {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int eax, ebx, ecx, edx;

    if (!__get_cpuid_max(0, nullptr))
        return false;

    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1 << 5));  // AVX2 bit
#else
    return false;
#endif
}

// =======================
// ARM NEON detection
// =======================
bool cpu_supports_neon() {
#if defined(__aarch64__)
    // AArch64 guarantees NEON
    return true;
#elif defined(__ARM_NEON)
    return true;
#else
    return false;
#endif
}

} // namespace labneura