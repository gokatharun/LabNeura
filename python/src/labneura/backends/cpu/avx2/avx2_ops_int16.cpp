#if defined(__AVX2__)

#include "labneura/backends/cpu/avx2.h"
#include "avx2_internal.h"
#include "labneura/utils/alignment.hpp"
#include <algorithm>
#include <stdexcept>

namespace labneura {

void AVX2Backend::operation_int16(const TensorBackend& other, OperationType op_type) {
    const int16_t* other_data = other.data_int16();
    int16_t* this_data = data_int16();
    std::size_t i = 0;
    const std::size_t size = size_;
    
    // DIV: Convert int16 to int32, divide with SIMD, pack back
    if (op_type == OperationType::DIV) {
        const bool aligned32 = labneura::util::is_aligned(this_data, labneura::util::AVX2_ALIGN) &&
                               labneura::util::is_aligned(other_data, labneura::util::AVX2_ALIGN);
        constexpr std::size_t PREFETCH_DISTANCE = 64;
        const std::size_t vec_len_16 = (size / 16) * 16;  // 16 int16s per iteration
        std::size_t i = 0;
        
        if (aligned32) {
            // Aligned SIMD: process 16 int16s per iteration
            while (i < vec_len_16) {
                if (i + PREFETCH_DISTANCE < size) {
                    _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                }
                // Load 16 int16s (256 bits)
                const __m256i va = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i));
                
                // Extract low 8 int16s, convert to int32
                const __m256i va_lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 0));
                const __m256i va_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 1));
                const __m256i vb_lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vb, 0));
                const __m256i vb_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vb, 1));
                
                // Convert int32 to FP32 for division
                const __m256 fa_lo = _mm256_cvtepi32_ps(va_lo);
                const __m256 fa_hi = _mm256_cvtepi32_ps(va_hi);
                const __m256 fb_lo = _mm256_cvtepi32_ps(vb_lo);
                const __m256 fb_hi = _mm256_cvtepi32_ps(vb_hi);
                
                // Add small epsilon to avoid division by zero
                const __m256 eps = _mm256_set1_ps(1e-10f);
                const __m256 fb_lo_safe = _mm256_add_ps(fb_lo, _mm256_and_ps(_mm256_cmp_ps(fb_lo, _mm256_setzero_ps(), _CMP_EQ_OQ), eps));
                const __m256 fb_hi_safe = _mm256_add_ps(fb_hi, _mm256_and_ps(_mm256_cmp_ps(fb_hi, _mm256_setzero_ps(), _CMP_EQ_OQ), eps));
                
                // SIMD division
                const __m256 res_lo = _mm256_div_ps(fa_lo, fb_lo_safe);
                const __m256 res_hi = _mm256_div_ps(fa_hi, fb_hi_safe);
                
                // Convert back to int32
                const __m256i res_lo_i = _mm256_cvtps_epi32(res_lo);
                const __m256i res_hi_i = _mm256_cvtps_epi32(res_hi);
                
                // Pack int32 back to int16 with saturation
                const __m256i result = _mm256_packs_epi32(res_lo_i, res_hi_i);
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i), result);
                i += 16;
            }
        } else {
            // Unaligned SIMD
            while (i < vec_len_16) {
                if (i + PREFETCH_DISTANCE < size) {
                    _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                }
                const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
                
                const __m256i va_lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 0));
                const __m256i va_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 1));
                const __m256i vb_lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vb, 0));
                const __m256i vb_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vb, 1));
                
                const __m256 fa_lo = _mm256_cvtepi32_ps(va_lo);
                const __m256 fa_hi = _mm256_cvtepi32_ps(va_hi);
                const __m256 fb_lo = _mm256_cvtepi32_ps(vb_lo);
                const __m256 fb_hi = _mm256_cvtepi32_ps(vb_hi);
                
                const __m256 eps = _mm256_set1_ps(1e-10f);
                const __m256 fb_lo_safe = _mm256_add_ps(fb_lo, _mm256_and_ps(_mm256_cmp_ps(fb_lo, _mm256_setzero_ps(), _CMP_EQ_OQ), eps));
                const __m256 fb_hi_safe = _mm256_add_ps(fb_hi, _mm256_and_ps(_mm256_cmp_ps(fb_hi, _mm256_setzero_ps(), _CMP_EQ_OQ), eps));
                
                const __m256 res_lo = _mm256_div_ps(fa_lo, fb_lo_safe);
                const __m256 res_hi = _mm256_div_ps(fa_hi, fb_hi_safe);
                
                const __m256i res_lo_i = _mm256_cvtps_epi32(res_lo);
                const __m256i res_hi_i = _mm256_cvtps_epi32(res_hi);
                
                const __m256i result = _mm256_packs_epi32(res_lo_i, res_hi_i);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), result);
                i += 16;
            }
        }
        // Scalar tail
        while (i < size_) {
            if (other_data[i] == 0) {
                this_data[i] = 0;
            } else {
                int32_t quot = static_cast<int32_t>(this_data[i]) / static_cast<int32_t>(other_data[i]);
                this_data[i] = static_cast<int16_t>(std::max(-32768, std::min(32767, quot)));
            }
            ++i;
        }
        return;
    }
    
    // Check alignment: 32 bytes for AVX2
    const bool aligned32 = labneura::util::is_aligned(this_data, labneura::util::AVX2_ALIGN) &&
                           labneura::util::is_aligned(other_data, labneura::util::AVX2_ALIGN);
    
    // Prefetch distance for INT16: 64 int16s = 128 bytes
    constexpr std::size_t PREFETCH_DISTANCE = 64;
    
    // AVX2 processes 16 int16 values per __m256i (256 bits / 16 bits = 16 elements)
    const std::size_t vec_len_64 = (size / 64) * 64;  // 64 int16s = 4 × __m256i
    const std::size_t vec_len_32 = (size / 32) * 32;  // 32 int16s = 2 × __m256i
    const std::size_t vec_len_16 = (size / 16) * 16;  // 16 int16s = 1 × __m256i
    
    if (op_type == OperationType::ADD) {
        if (aligned32) {
            // Aligned path: 64 int16s per iteration
            while (i < vec_len_64) {
                if (i + PREFETCH_DISTANCE < size) {
                    _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                }
                const __m256i va0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i + 0));
                const __m256i vb0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i + 0));
                const __m256i va1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
                const __m256i vb1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
                const __m256i va2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i + 32));
                const __m256i vb2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i + 32));
                const __m256i va3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i + 48));
                const __m256i vb3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i + 48));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 0), _mm256_adds_epi16(va0, vb0));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_adds_epi16(va1, vb1));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 32), _mm256_adds_epi16(va2, vb2));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 48), _mm256_adds_epi16(va3, vb3));
                i += 64;
            }
            if (i < vec_len_32) {
                const __m256i va0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i));
                const __m256i va1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
                const __m256i vb1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_adds_epi16(va0, vb0));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_adds_epi16(va1, vb1));
                i += 32;
            }
            if (i < vec_len_16) {
                const __m256i va = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_adds_epi16(va, vb));
                i += 16;
            }
        } else {
            // Unaligned path: 64 int16s per iteration
            while (i < vec_len_64) {
                if (i + PREFETCH_DISTANCE < size) {
                    _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                }
                const __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 0));
                const __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 0));
                const __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
                const __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
                const __m256i va2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 32));
                const __m256i vb2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 32));
                const __m256i va3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 48));
                const __m256i vb3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 48));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 0), _mm256_adds_epi16(va0, vb0));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_adds_epi16(va1, vb1));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 32), _mm256_adds_epi16(va2, vb2));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 48), _mm256_adds_epi16(va3, vb3));
                i += 64;
            }
            if (i < vec_len_32) {
                const __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
                const __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
                const __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_adds_epi16(va0, vb0));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_adds_epi16(va1, vb1));
                i += 32;
            }
            if (i < vec_len_16) {
                const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_adds_epi16(va, vb));
                i += 16;
            }
        }
        if (i + 8 <= size) {
            const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(this_data + i));
            const __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(other_data + i));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(this_data + i), _mm_adds_epi16(va, vb));
            i += 8;
        }
        if (i + 4 <= size) {
            for (int j = 0; j < 4; ++j) {
                int32_t sum = static_cast<int32_t>(this_data[i + j]) + static_cast<int32_t>(other_data[i + j]);
                this_data[i + j] = static_cast<int16_t>(std::max(-32768, std::min(32767, sum)));
            }
            i += 4;
        }
        while (i < size) {
            int32_t sum = static_cast<int32_t>(this_data[i]) + static_cast<int32_t>(other_data[i]);
            this_data[i] = static_cast<int16_t>(std::max(-32768, std::min(32767, sum)));
            ++i;
        }
    } else if (op_type == OperationType::MUL) {
        if (aligned32) {
            while (i < vec_len_64) {
                if (i + PREFETCH_DISTANCE < size) {
                    _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                }
                const __m256i va0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i + 0));
                const __m256i vb0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i + 0));
                const __m256i va1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
                const __m256i vb1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
                const __m256i va2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i + 32));
                const __m256i vb2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i + 32));
                const __m256i va3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i + 48));
                const __m256i vb3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i + 48));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 0), _mm256_mulhrs_epi16(va0, vb0));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_mulhrs_epi16(va1, vb1));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 32), _mm256_mulhrs_epi16(va2, vb2));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 48), _mm256_mulhrs_epi16(va3, vb3));
                i += 64;
            }
            if (i < vec_len_32) {
                const __m256i va0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i));
                const __m256i va1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
                const __m256i vb1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_mulhrs_epi16(va0, vb0));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_mulhrs_epi16(va1, vb1));
                i += 32;
            }
            if (i < vec_len_16) {
                const __m256i va = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_mulhrs_epi16(va, vb));
                i += 16;
            }
        } else {
            while (i < vec_len_64) {
                if (i + PREFETCH_DISTANCE < size) {
                    _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                }
                const __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 0));
                const __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 0));
                const __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
                const __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
                const __m256i va2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 32));
                const __m256i vb2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 32));
                const __m256i va3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 48));
                const __m256i vb3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 48));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 0), _mm256_mulhrs_epi16(va0, vb0));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_mulhrs_epi16(va1, vb1));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 32), _mm256_mulhrs_epi16(va2, vb2));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 48), _mm256_mulhrs_epi16(va3, vb3));
                i += 64;
            }
            if (i < vec_len_32) {
                const __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
                const __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
                const __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_mulhrs_epi16(va0, vb0));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_mulhrs_epi16(va1, vb1));
                i += 32;
            }
            if (i < vec_len_16) {
                const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_mulhrs_epi16(va, vb));
                i += 16;
            }
        }
        if (i + 8 <= size) {
            const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(this_data + i));
            const __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(other_data + i));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(this_data + i), _mm_mulhrs_epi16(va, vb));
            i += 8;
        }
        if (i + 4 <= size) {
            for (int j = 0; j < 4; ++j) {
                int32_t product = static_cast<int32_t>(this_data[i + j]) * static_cast<int32_t>(other_data[i + j]);
                this_data[i + j] = static_cast<int16_t>(std::max(-32768, std::min(32767, product)));
            }
            i += 4;
        }
        while (i < size) {
            int32_t product = static_cast<int32_t>(this_data[i]) * static_cast<int32_t>(other_data[i]);
            this_data[i] = static_cast<int16_t>(std::max(-32768, std::min(32767, product)));
            ++i;
        }
    } else {  // SUB
        if (aligned32) {
            while (i < vec_len_64) {
                if (i + PREFETCH_DISTANCE < size) {
                    _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                }
                const __m256i va0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i + 0));
                const __m256i vb0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i + 0));
                const __m256i va1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
                const __m256i vb1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
                const __m256i va2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i + 32));
                const __m256i vb2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i + 32));
                const __m256i va3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i + 48));
                const __m256i vb3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i + 48));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 0), _mm256_subs_epi16(va0, vb0));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_subs_epi16(va1, vb1));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 32), _mm256_subs_epi16(va2, vb2));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 48), _mm256_subs_epi16(va3, vb3));
                i += 64;
            }
            if (i < vec_len_32) {
                const __m256i va0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i));
                const __m256i va1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
                const __m256i vb1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_subs_epi16(va0, vb0));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_subs_epi16(va1, vb1));
                i += 32;
            }
            if (i < vec_len_16) {
                const __m256i va = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_subs_epi16(va, vb));
                i += 16;
            }
        } else {
            while (i < vec_len_64) {
                if (i + PREFETCH_DISTANCE < size) {
                    _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                }
                const __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 0));
                const __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 0));
                const __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
                const __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
                const __m256i va2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 32));
                const __m256i vb2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 32));
                const __m256i va3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 48));
                const __m256i vb3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 48));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 0), _mm256_subs_epi16(va0, vb0));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_subs_epi16(va1, vb1));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 32), _mm256_subs_epi16(va2, vb2));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 48), _mm256_subs_epi16(va3, vb3));
                i += 64;
            }
            if (i < vec_len_32) {
                const __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
                const __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
                const __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_subs_epi16(va0, vb0));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_subs_epi16(va1, vb1));
                i += 32;
            }
            if (i < vec_len_16) {
                const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_subs_epi16(va, vb));
                i += 16;
            }
        }
        if (i + 8 <= size) {
            const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(this_data + i));
            const __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(other_data + i));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(this_data + i), _mm_subs_epi16(va, vb));
            i += 8;
        }
        if (i + 4 <= size) {
            for (int j = 0; j < 4; ++j) {
                int32_t diff = static_cast<int32_t>(this_data[i + j]) - static_cast<int32_t>(other_data[i + j]);
                this_data[i + j] = static_cast<int16_t>(std::max(-32768, std::min(32767, diff)));
            }
            i += 4;
        }
        while (i < size) {
            int32_t diff = static_cast<int32_t>(this_data[i]) - static_cast<int32_t>(other_data[i]);
            this_data[i] = static_cast<int16_t>(std::max(-32768, std::min(32767, diff)));
            ++i;
        }
    }
}

} // namespace labneura

#endif // __AVX2__