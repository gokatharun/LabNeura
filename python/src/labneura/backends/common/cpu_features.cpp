#include "labneura/backends/common/cpu_features.h"
#include "cpu_features_internal.h"

#include <atomic>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

#if defined(__linux__) && defined(__arm__) && !defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace labneura {

namespace {

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
void cpuid(unsigned int leaf, unsigned int subleaf, int registers[4]) {
    __cpuidex(registers, static_cast<int>(leaf), static_cast<int>(subleaf));
}

unsigned long long xcr0() {
    return _xgetbv(0);
}
#elif defined(__x86_64__) || defined(__i386__)
bool cpuid(unsigned int leaf, unsigned int subleaf, unsigned int& eax,
           unsigned int& ebx, unsigned int& ecx, unsigned int& edx) {
    return __get_cpuid_count(leaf, subleaf, &eax, &ebx, &ecx, &edx) != 0;
}

unsigned long long xcr0() {
    unsigned int low;
    unsigned int high;
    __asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
    return (static_cast<unsigned long long>(high) << 32) | low;
}
#endif

bool detect_sse41() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int registers[4];
    cpuid(1, 0, registers);
    return (registers[2] & (1 << 19)) != 0;
#elif defined(__x86_64__) || defined(__i386__)
    unsigned int eax, ebx, ecx, edx;
    return cpuid(1, 0, eax, ebx, ecx, edx) && (ecx & bit_SSE4_1) != 0;
#else
    return false;
#endif
}

bool detect_avx2() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int registers[4];
    cpuid(1, 0, registers);
    constexpr int avx = 1 << 28;
    constexpr int osxsave = 1 << 27;
    if ((registers[2] & (avx | osxsave)) != (avx | osxsave) ||
        (xcr0() & 0x6) != 0x6) {
        return false;
    }
    cpuid(7, 0, registers);
    return (registers[1] & (1 << 5)) != 0;
#elif defined(__x86_64__) || defined(__i386__)
    unsigned int eax, ebx, ecx, edx;
    constexpr unsigned int avx = 1U << 28;
    constexpr unsigned int osxsave = 1U << 27;
    if (!cpuid(1, 0, eax, ebx, ecx, edx) ||
        (ecx & (avx | osxsave)) != (avx | osxsave) ||
        (xcr0() & 0x6U) != 0x6U) {
        return false;
    }
    return cpuid(7, 0, eax, ebx, ecx, edx) &&
           (ebx & bit_AVX2) != 0;
#else
    return false;
#endif
}

bool detect_f16c() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int registers[4];
    cpuid(1, 0, registers);
    constexpr int avx = 1 << 28;
    constexpr int osxsave = 1 << 27;
    constexpr int f16c = 1 << 29;
    return (registers[2] & (avx | osxsave | f16c)) == (avx | osxsave | f16c) &&
           (xcr0() & 0x6) == 0x6;
#elif defined(__x86_64__) || defined(__i386__)
    unsigned int eax, ebx, ecx, edx;
    constexpr unsigned int avx = 1U << 28;
    constexpr unsigned int osxsave = 1U << 27;
    constexpr unsigned int f16c = 1U << 29;
    return cpuid(1, 0, eax, ebx, ecx, edx) &&
           (ecx & (avx | osxsave | f16c)) == (avx | osxsave | f16c) &&
           (xcr0() & 0x6U) == 0x6U;
#else
    return false;
#endif
}

bool detect_neon() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return true;
#elif defined(__linux__) && defined(__arm__)
    return (getauxval(AT_HWCAP) & HWCAP_NEON) != 0;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    return true;
#else
    return false;
#endif
}

struct DetectedFeatures {
    CPUFeatures features;
    bool neon;
};

std::atomic<std::size_t> g_detection_runs{0};

// Detection runs once per process (thread-safe function-local static).
const DetectedFeatures& cached_features() {
    static const DetectedFeatures detected = [] {
        g_detection_runs.fetch_add(1, std::memory_order_relaxed);
        return DetectedFeatures{{detect_sse41(), detect_avx2(), detect_f16c()},
                                detect_neon()};
    }();
    return detected;
}

}  // namespace

namespace detail {

std::size_t cpu_feature_detection_count() {
    return g_detection_runs.load(std::memory_order_relaxed);
}

}  // namespace detail

CPUFeatures detect_cpu_features() {
    return cached_features().features;
}

bool cpu_supports_avx2() {
    return cached_features().features.avx2;
}

bool cpu_supports_f16c() {
    return cached_features().features.f16c;
}

bool cpu_supports_neon() {
    return cached_features().neon;
}

} // namespace labneura
