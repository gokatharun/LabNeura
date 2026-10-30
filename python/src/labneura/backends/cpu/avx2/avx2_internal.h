#pragma once

#if defined(__AVX2__)

#include <immintrin.h>
#include <algorithm>
#include <cstdlib>

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace labneura {

// =======================
// Register detection utility
// =======================

// Detect number of available AVX2 (YMM) registers using CPUID
// Returns 32 if AVX-512 is supported (extends YMM registers)
// Returns 16 for standard x86_64 (default)
static inline int detect_avx2_registers() {
    int num_registers = 16;  // Default: standard x86_64

    try {
        #if defined(__GNUC__) || defined(__clang__)
        unsigned int eax, ebx, ecx, edx;
        
        // Check CPUID support
        if (__get_cpuid_max(0, nullptr) >= 7) {
            // CPUID leaf 7, subleaf 0: Extended Features
            if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
                // Check EBX bit 16: AVX-512F (Foundation)
                // AVX-512 extends YMM registers from 16 to 32
                if (ebx & (1 << 16)) {
                    num_registers = 32;
                }
            }
        }
        #elif defined(_MSC_VER)
        int cpuInfo[4];
        __cpuidex(cpuInfo, 7, 0);
        // Check EBX bit 16: AVX-512F
        if (cpuInfo[1] & (1 << 16)) {
            num_registers = 32;
        }
        #endif
    } catch (...) {
        // If any error occurs, use safe default
        num_registers = 16;
    }

    return num_registers;
}

// Get optimized unroll factor: half of available registers
static inline int get_avx2_unroll_factor() {
    return detect_avx2_registers() / 2;  // 8 registers per operand
}

// FP32: 8 elements per YMM, unroll_factor=8 → 64 floats per iteration
static inline std::size_t get_avx2_fp32_chunk() {
    return get_avx2_unroll_factor() * 8;
}

static inline __m256i divide_int32_avx2(__m256i numerator, __m256i denominator) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i zero_mask = _mm256_cmpeq_epi32(denominator, zero);
    const __m256i safe_denominator = _mm256_blendv_epi8(
        denominator, _mm256_set1_epi32(1), zero_mask);
    const __m256i abs_numerator = _mm256_abs_epi32(numerator);
    const __m256i abs_denominator = _mm256_abs_epi32(safe_denominator);
    __m256i quotient = _mm256_cvttps_epi32(_mm256_div_ps(
        _mm256_cvtepi32_ps(abs_numerator),
        _mm256_cvtepi32_ps(abs_denominator)));

    const __m256i product = _mm256_mullo_epi32(quotient, abs_denominator);
    const __m256i too_large = _mm256_cmpgt_epi32(product, abs_numerator);
    quotient = _mm256_sub_epi32(
        quotient, _mm256_and_si256(too_large, _mm256_set1_epi32(1)));

    const __m256i next_product = _mm256_mullo_epi32(
        _mm256_add_epi32(quotient, _mm256_set1_epi32(1)), abs_denominator);
    const __m256i too_small = _mm256_andnot_si256(
        _mm256_cmpgt_epi32(next_product, abs_numerator), _mm256_set1_epi32(1));
    quotient = _mm256_add_epi32(quotient, too_small);

    const __m256i negative = _mm256_cmpgt_epi32(
        zero, _mm256_xor_si256(numerator, denominator));
    quotient = _mm256_blendv_epi8(
        quotient, _mm256_sub_epi32(zero, quotient), negative);
    return _mm256_andnot_si256(zero_mask, quotient);
}

} // namespace labneura

#endif // __AVX2__
