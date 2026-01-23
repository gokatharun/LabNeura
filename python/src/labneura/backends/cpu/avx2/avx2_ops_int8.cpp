#if defined(__AVX2__)

#include "labneura/backends/cpu/avx2.h"
#include "avx2_internal.h"
#include "labneura/utils/alignment.hpp"
#include <algorithm>
#include <stdexcept>

namespace labneura {

void AVX2Backend::operation_int8(const TensorBackend& other, OperationType op_type) {
    const int8_t* other_data = other.data_int8();
    int8_t* this_data = data_int8();
    const std::size_t vec_len = (size_ / 32) * 32;  // AVX2 = 32 int8

    if (op_type == OperationType::ADD) {
        for (std::size_t i = 0; i < vec_len; i += 32) {
            const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
            const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
            const __m256i r = _mm256_adds_epi8(va, vb);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), r);
        }
        for (std::size_t i = vec_len; i < size_; ++i) {
            int16_t sum = static_cast<int16_t>(this_data[i]) + static_cast<int16_t>(other_data[i]);
            this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(sum))));
        }
        return;
    }
    if (op_type == OperationType::MUL) {
        for (std::size_t i = 0; i < vec_len; i += 32) {
            const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
            const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
            const __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 0));
            const __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
            const __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 0));
            const __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
            const __m256i prod_lo = _mm256_mullo_epi16(va_lo, vb_lo);
            const __m256i prod_hi = _mm256_mullo_epi16(va_hi, vb_hi);
            const __m256i r = _mm256_packs_epi16(prod_lo, prod_hi);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), r);
        }
        for (std::size_t i = vec_len; i < size_; ++i) {
            int16_t prod = static_cast<int16_t>(this_data[i]) * static_cast<int16_t>(other_data[i]);
            this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(prod))));
        }
        return;
    }
    if (op_type == OperationType::DIV) {
        // INT8 division: Convert int8->int16->int32, SIMD divide via FP32, pack back
        const std::size_t vec_len = (size_ / 32) * 32;  // 32 int8s per iteration
        constexpr std::size_t PREFETCH_DISTANCE = 128;
        std::size_t i = 0;
        
        // SIMD main loop: process 32 int8s per iteration
        while (i < vec_len) {
            if (i + PREFETCH_DISTANCE < size_) {
                _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
            }
            
            // Load 32 int8s
            const __m256i va8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
            const __m256i vb8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
            
            // Expand int8 to int16 (extract and extend)
            const __m256i va_lo16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va8, 0));
            const __m256i va_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va8, 1));
            const __m256i vb_lo16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb8, 0));
            const __m256i vb_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb8, 1));
            
            // Expand int16 to int32 and process in 4 chunks per original 32
            // Chunk 1: va_lo16[0..3] -> int32
            const __m256i va_00 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(va_lo16));
            const __m256i vb_00 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vb_lo16));
            // Chunk 2: va_lo16[4..7] -> int32
            const __m256i va_01 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va_lo16, 1));
            const __m256i vb_01 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vb_lo16, 1));
            // Chunk 3: va_hi16[0..3] -> int32
            const __m256i va_10 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(va_hi16));
            const __m256i vb_10 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vb_hi16));
            // Chunk 4: va_hi16[4..7] -> int32
            const __m256i va_11 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va_hi16, 1));
            const __m256i vb_11 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vb_hi16, 1));
            
            // Convert to FP32 and divide
            auto safe_div = [](const __m256i& a, const __m256i& b) -> __m256i {
                const __m256 fa = _mm256_cvtepi32_ps(a);
                const __m256 fb = _mm256_cvtepi32_ps(b);
                const __m256 eps = _mm256_set1_ps(1e-10f);
                const __m256 fb_safe = _mm256_add_ps(fb, _mm256_and_ps(_mm256_cmp_ps(fb, _mm256_setzero_ps(), _CMP_EQ_OQ), eps));
                const __m256 res = _mm256_div_ps(fa, fb_safe);
                return _mm256_cvtps_epi32(res);
            };
            
            const __m256i res_00 = safe_div(va_00, vb_00);
            const __m256i res_01 = safe_div(va_01, vb_01);
            const __m256i res_10 = safe_div(va_10, vb_10);
            const __m256i res_11 = safe_div(va_11, vb_11);
            
            // Pack int32 back to int16
            const __m256i res_lo16 = _mm256_packs_epi32(res_00, res_01);
            const __m256i res_hi16 = _mm256_packs_epi32(res_10, res_11);
            
            // Pack int16 back to int8
            const __m256i result = _mm256_packs_epi16(res_lo16, res_hi16);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), result);
            i += 32;
        }
        
        // Scalar tail
        while (i < size_) {
            if (other_data[i] == 0) {
                this_data[i] = 0;
            } else {
                int16_t quot = static_cast<int16_t>(this_data[i]) / static_cast<int16_t>(other_data[i]);
                this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(quot))));
            }
            ++i;
        }
        return;
    }
    // SUB (default)
    for (std::size_t i = 0; i < vec_len; i += 32) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
        const __m256i r = _mm256_subs_epi8(va, vb);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), r);
    }
    for (std::size_t i = vec_len; i < size_; ++i) {
        int16_t diff = static_cast<int16_t>(this_data[i]) - static_cast<int16_t>(other_data[i]);
        this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(diff))));
    }
}

// =======================
// FP16 and INT16 kernels (optimized with SIMD, prefetch, and tail chains)
// =======================

} // namespace labneura

#endif // __AVX2__