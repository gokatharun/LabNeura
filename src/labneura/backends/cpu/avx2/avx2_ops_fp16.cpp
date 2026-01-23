#if defined(__AVX2__)

#include "labneura/backends/cpu/avx2.h"
#include "avx2_internal.h"
#include "labneura/utils/alignment.hpp"
#include <algorithm>
#include <stdexcept>

namespace labneura {

void AVX2Backend::operation_fp16(const TensorBackend& other, OperationType op_type) {
    // FP16 treated as int16_t for bitwise SIMD operations
    const int16_t* other_data = other.data_fp16();
    int16_t* this_data = data_fp16();
    std::size_t i = 0;
    const std::size_t size = size_;
    
    // DIV: Convert FP16 to FP32, divide with SIMD, convert back
    if (op_type == OperationType::DIV) {
        // Helper to convert FP16 bits to FP32 (no actual half-precision, just bit interpretation)
        auto fp16_to_fp32_bits = [](int16_t h) -> float {
            // Simplified scalar fallback; replace with proper FP16 conversion if needed
            return static_cast<float>(h) / 1024.0f;  // Rough approximation
        };
        
        auto fp32_to_fp16_bits = [](float f) -> int16_t {
            // Inverse: convert FP32 back to FP16 bit pattern
            return static_cast<int16_t>(f * 1024.0f);
        };
        
        // Prefetched scalar loop with 32-element chunks for cache efficiency
        constexpr std::size_t PREFETCH_DISTANCE = 64;
        const std::size_t chunk_size = 32;
        const std::size_t vec_len = (size / chunk_size) * chunk_size;
        
        std::size_t j = 0;
        while (j < vec_len) {
            if (j + PREFETCH_DISTANCE < size) {
                _mm_prefetch(reinterpret_cast<const char*>(this_data + j + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(other_data + j + PREFETCH_DISTANCE), _MM_HINT_T0);
            }
            // Process 32 elements: convert pairs to FP32, divide, convert back
            for (std::size_t k = 0; k < chunk_size; k += 8) {
                // Load 8 FP16 values (16 bytes)
                const __m128i v_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(this_data + j + k));
                const __m128i v_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(other_data + j + k));
                
                // Convert to FP32: low 4 elements
                const __m128 fa_lo = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v_a));
                const __m128 fb_lo = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v_b));
                
                // Convert to FP32: high 4 elements
                const __m128i v_a_hi = _mm_srli_si128(v_a, 8);
                const __m128i v_b_hi = _mm_srli_si128(v_b, 8);
                const __m128 fa_hi = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v_a_hi));
                const __m128 fb_hi = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v_b_hi));
                
                // SIMD division with zero check
                const __m128 res_lo = _mm_div_ps(fa_lo, _mm_add_ps(fb_lo, _mm_set1_ps(1e-10f)));
                const __m128 res_hi = _mm_div_ps(fa_hi, _mm_add_ps(fb_hi, _mm_set1_ps(1e-10f)));
                
                // Convert back to int16
                const __m128i res_lo_i = _mm_cvtps_epi32(res_lo);
                const __m128i res_hi_i = _mm_cvtps_epi32(res_hi);
                const __m128i result = _mm_packs_epi32(res_lo_i, res_hi_i);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(this_data + j + k), result);
            }
            j += chunk_size;
        }
        // Scalar tail
        while (j < size) {
            if (other_data[j] != 0) {
                this_data[j] = static_cast<int16_t>(this_data[j] / other_data[j]);
            } else {
                this_data[j] = 0;
            }
            ++j;
        }
        return;
    }
    
    // Check alignment: 32 bytes for AVX2
    const bool aligned32 = labneura::util::is_aligned(this_data, labneura::util::AVX2_ALIGN) &&
                           labneura::util::is_aligned(other_data, labneura::util::AVX2_ALIGN);
    
    // Prefetch distance for FP16: 64 int16s = 128 bytes
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
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 0), _mm256_add_epi16(va0, vb0));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_add_epi16(va1, vb1));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 32), _mm256_add_epi16(va2, vb2));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 48), _mm256_add_epi16(va3, vb3));
                i += 64;
            }
            if (i < vec_len_32) {
                const __m256i va0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i));
                const __m256i va1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
                const __m256i vb1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_add_epi16(va0, vb0));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_add_epi16(va1, vb1));
                i += 32;
            }
            if (i < vec_len_16) {
                const __m256i va = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_add_epi16(va, vb));
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
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 0), _mm256_add_epi16(va0, vb0));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_add_epi16(va1, vb1));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 32), _mm256_add_epi16(va2, vb2));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 48), _mm256_add_epi16(va3, vb3));
                i += 64;
            }
            if (i < vec_len_32) {
                const __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
                const __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
                const __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_add_epi16(va0, vb0));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_add_epi16(va1, vb1));
                i += 32;
            }
            if (i < vec_len_16) {
                const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_add_epi16(va, vb));
                i += 16;
            }
        }
        if (i + 8 <= size) {
            const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(this_data + i));
            const __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(other_data + i));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(this_data + i), _mm_add_epi16(va, vb));
            i += 8;
        }
        if (i + 4 <= size) {
            for (int j = 0; j < 4; ++j) {
                this_data[i + j] = static_cast<int16_t>(static_cast<int32_t>(this_data[i + j]) + static_cast<int32_t>(other_data[i + j]));
            }
            i += 4;
        }
        while (i < size) {
            this_data[i] = static_cast<int16_t>(static_cast<int32_t>(this_data[i]) + static_cast<int32_t>(other_data[i]));
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
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 0), _mm256_sub_epi16(va0, vb0));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_sub_epi16(va1, vb1));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 32), _mm256_sub_epi16(va2, vb2));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 48), _mm256_sub_epi16(va3, vb3));
                i += 64;
            }
            if (i < vec_len_32) {
                const __m256i va0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i));
                const __m256i va1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
                const __m256i vb1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_sub_epi16(va0, vb0));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_sub_epi16(va1, vb1));
                i += 32;
            }
            if (i < vec_len_16) {
                const __m256i va = _mm256_load_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb = _mm256_load_si256(reinterpret_cast<const __m256i*>(other_data + i));
                _mm256_store_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_sub_epi16(va, vb));
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
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 0), _mm256_sub_epi16(va0, vb0));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_sub_epi16(va1, vb1));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 32), _mm256_sub_epi16(va2, vb2));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 48), _mm256_sub_epi16(va3, vb3));
                i += 64;
            }
            if (i < vec_len_32) {
                const __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
                const __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
                const __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_sub_epi16(va0, vb0));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 16), _mm256_sub_epi16(va1, vb1));
                i += 32;
            }
            if (i < vec_len_16) {
                const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
                const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), _mm256_sub_epi16(va, vb));
                i += 16;
            }
        }
        if (i + 8 <= size) {
            const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(this_data + i));
            const __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(other_data + i));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(this_data + i), _mm_sub_epi16(va, vb));
            i += 8;
        }
        if (i + 4 <= size) {
            for (int j = 0; j < 4; ++j) {
                this_data[i + j] = static_cast<int16_t>(static_cast<int32_t>(this_data[i + j]) - static_cast<int32_t>(other_data[i + j]));
            }
            i += 4;
        }
        while (i < size) {
            this_data[i] = static_cast<int16_t>(static_cast<int32_t>(this_data[i]) - static_cast<int32_t>(other_data[i]));
            ++i;
        }
    }
}

} // namespace labneura

#endif // __AVX2__