#if defined(__AVX2__)

#include "labneura/backends/cpu/avx2.h"
#include "avx2_internal.h"
#include "labneura/utils/alignment.hpp"
#include <algorithm>
#include <stdexcept>

namespace labneura {

void AVX2Backend::operation_fp32(const TensorBackend& other, OperationType op_type) {
    const float* other_data = other.data_fp32();
    float* this_data = data_fp32();
    std::size_t i = 0;
    const std::size_t size = size_;
    const bool aligned32 = labneura::util::is_aligned(this_data, labneura::util::AVX2_ALIGN) &&
                           labneura::util::is_aligned(other_data, labneura::util::AVX2_ALIGN);

    // Prefetch distance: 512 bytes (8 cache lines) ahead for medium-sized arrays
    constexpr std::size_t PREFETCH_DISTANCE = 128;  // 128 floats = 512 bytes

    if (op_type == OperationType::ADD) {
        if (aligned32) {
            // Aligned main loop: 64 floats/iteration with prefetching
            while (i + 64 <= size) {
                // Prefetch data ahead for better cache utilization (helps 100K-1M range)
                if (i + PREFETCH_DISTANCE < size) {
                    _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                }
                
                const __m256 va0 = _mm256_load_ps(this_data + i + 0);
                const __m256 vb0 = _mm256_load_ps(other_data + i + 0);
                const __m256 va1 = _mm256_load_ps(this_data + i + 8);
                const __m256 vb1 = _mm256_load_ps(other_data + i + 8);
                const __m256 va2 = _mm256_load_ps(this_data + i + 16);
                const __m256 vb2 = _mm256_load_ps(other_data + i + 16);
                const __m256 va3 = _mm256_load_ps(this_data + i + 24);
                const __m256 vb3 = _mm256_load_ps(other_data + i + 24);
                const __m256 va4 = _mm256_load_ps(this_data + i + 32);
                const __m256 vb4 = _mm256_load_ps(other_data + i + 32);
                const __m256 va5 = _mm256_load_ps(this_data + i + 40);
                const __m256 vb5 = _mm256_load_ps(other_data + i + 40);
                const __m256 va6 = _mm256_load_ps(this_data + i + 48);
                const __m256 vb6 = _mm256_load_ps(other_data + i + 48);
                const __m256 va7 = _mm256_load_ps(this_data + i + 56);
                const __m256 vb7 = _mm256_load_ps(other_data + i + 56);
                _mm256_store_ps(this_data + i + 0,  _mm256_add_ps(va0, vb0));
                _mm256_store_ps(this_data + i + 8,  _mm256_add_ps(va1, vb1));
                _mm256_store_ps(this_data + i + 16, _mm256_add_ps(va2, vb2));
                _mm256_store_ps(this_data + i + 24, _mm256_add_ps(va3, vb3));
                _mm256_store_ps(this_data + i + 32, _mm256_add_ps(va4, vb4));
                _mm256_store_ps(this_data + i + 40, _mm256_add_ps(va5, vb5));
                _mm256_store_ps(this_data + i + 48, _mm256_add_ps(va6, vb6));
                _mm256_store_ps(this_data + i + 56, _mm256_add_ps(va7, vb7));
                i += 64;
            }
            if (i + 32 <= size) {
                const __m256 va0 = _mm256_load_ps(this_data + i);
                const __m256 vb0 = _mm256_load_ps(other_data + i);
                const __m256 va1 = _mm256_load_ps(this_data + i + 8);
                const __m256 vb1 = _mm256_load_ps(other_data + i + 8);
                const __m256 va2 = _mm256_load_ps(this_data + i + 16);
                const __m256 vb2 = _mm256_load_ps(other_data + i + 16);
                const __m256 va3 = _mm256_load_ps(this_data + i + 24);
                const __m256 vb3 = _mm256_load_ps(other_data + i + 24);
                _mm256_store_ps(this_data + i, _mm256_add_ps(va0, vb0));
                _mm256_store_ps(this_data + i + 8, _mm256_add_ps(va1, vb1));
                _mm256_store_ps(this_data + i + 16, _mm256_add_ps(va2, vb2));
                _mm256_store_ps(this_data + i + 24, _mm256_add_ps(va3, vb3));
                i += 32;
            }
            if (i + 16 <= size) {
                const __m256 va0 = _mm256_load_ps(this_data + i);
                const __m256 vb0 = _mm256_load_ps(other_data + i);
                const __m256 va1 = _mm256_load_ps(this_data + i + 8);
                const __m256 vb1 = _mm256_load_ps(other_data + i + 8);
                _mm256_store_ps(this_data + i, _mm256_add_ps(va0, vb0));
                _mm256_store_ps(this_data + i + 8, _mm256_add_ps(va1, vb1));
                i += 16;
            }
            if (i + 8 <= size) {
                const __m256 va = _mm256_load_ps(this_data + i);
                const __m256 vb = _mm256_load_ps(other_data + i);
                _mm256_store_ps(this_data + i, _mm256_add_ps(va, vb));
                i += 8;
            }
            if (i + 4 <= size) {
                const __m128 va = _mm_load_ps(this_data + i);
                const __m128 vb = _mm_load_ps(other_data + i);
                _mm_store_ps(this_data + i, _mm_add_ps(va, vb));
                i += 4;
            }
        } else {
            // Unaligned path with prefetching
            while (i + 64 <= size) {
                // Prefetch for unaligned access
                if (i + PREFETCH_DISTANCE < size) {
                    _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                }
                
                const __m256 va0 = _mm256_loadu_ps(this_data + i + 0);
                const __m256 vb0 = _mm256_loadu_ps(other_data + i + 0);
                const __m256 va1 = _mm256_loadu_ps(this_data + i + 8);
                const __m256 vb1 = _mm256_loadu_ps(other_data + i + 8);
                const __m256 va2 = _mm256_loadu_ps(this_data + i + 16);
                const __m256 vb2 = _mm256_loadu_ps(other_data + i + 16);
                const __m256 va3 = _mm256_loadu_ps(this_data + i + 24);
                const __m256 vb3 = _mm256_loadu_ps(other_data + i + 24);
                const __m256 va4 = _mm256_loadu_ps(this_data + i + 32);
                const __m256 vb4 = _mm256_loadu_ps(other_data + i + 32);
                const __m256 va5 = _mm256_loadu_ps(this_data + i + 40);
                const __m256 vb5 = _mm256_loadu_ps(other_data + i + 40);
                const __m256 va6 = _mm256_loadu_ps(this_data + i + 48);
                const __m256 vb6 = _mm256_loadu_ps(other_data + i + 48);
                const __m256 va7 = _mm256_loadu_ps(this_data + i + 56);
                const __m256 vb7 = _mm256_loadu_ps(other_data + i + 56);
                _mm256_storeu_ps(this_data + i + 0, _mm256_add_ps(va0, vb0));
                _mm256_storeu_ps(this_data + i + 8, _mm256_add_ps(va1, vb1));
                _mm256_storeu_ps(this_data + i + 16, _mm256_add_ps(va2, vb2));
                _mm256_storeu_ps(this_data + i + 24, _mm256_add_ps(va3, vb3));
                _mm256_storeu_ps(this_data + i + 32, _mm256_add_ps(va4, vb4));
                _mm256_storeu_ps(this_data + i + 40, _mm256_add_ps(va5, vb5));
                _mm256_storeu_ps(this_data + i + 48, _mm256_add_ps(va6, vb6));
                _mm256_storeu_ps(this_data + i + 56, _mm256_add_ps(va7, vb7));
                i += 64;
            }
            if (i + 32 <= size) {
                const __m256 va0 = _mm256_loadu_ps(this_data + i);
                const __m256 vb0 = _mm256_loadu_ps(other_data + i);
                const __m256 va1 = _mm256_loadu_ps(this_data + i + 8);
                const __m256 vb1 = _mm256_loadu_ps(other_data + i + 8);
                const __m256 va2 = _mm256_loadu_ps(this_data + i + 16);
                const __m256 vb2 = _mm256_loadu_ps(other_data + i + 16);
                const __m256 va3 = _mm256_loadu_ps(this_data + i + 24);
                const __m256 vb3 = _mm256_loadu_ps(other_data + i + 24);
                _mm256_storeu_ps(this_data + i, _mm256_add_ps(va0, vb0));
                _mm256_storeu_ps(this_data + i + 8, _mm256_add_ps(va1, vb1));
                _mm256_storeu_ps(this_data + i + 16, _mm256_add_ps(va2, vb2));
                _mm256_storeu_ps(this_data + i + 24, _mm256_add_ps(va3, vb3));
                i += 32;
            }
            if (i + 16 <= size) {
                const __m256 va0 = _mm256_loadu_ps(this_data + i);
                const __m256 vb0 = _mm256_loadu_ps(other_data + i);
                const __m256 va1 = _mm256_loadu_ps(this_data + i + 8);
                const __m256 vb1 = _mm256_loadu_ps(other_data + i + 8);
                _mm256_storeu_ps(this_data + i, _mm256_add_ps(va0, vb0));
                _mm256_storeu_ps(this_data + i + 8, _mm256_add_ps(va1, vb1));
                i += 16;
            }
            if (i + 8 <= size) {
                const __m256 va = _mm256_loadu_ps(this_data + i);
                const __m256 vb = _mm256_loadu_ps(other_data + i);
                _mm256_storeu_ps(this_data + i, _mm256_add_ps(va, vb));
                i += 8;
            }
            if (i + 4 <= size) {
                const __m128 va = _mm_loadu_ps(this_data + i);
                const __m128 vb = _mm_loadu_ps(other_data + i);
                _mm_storeu_ps(this_data + i, _mm_add_ps(va, vb));
                i += 4;
            }
        }
        while (i < size) { this_data[i] += other_data[i]; ++i; }
        return;
    }
    if (op_type == OperationType::MUL) {
        if (aligned32) {
            while (i + 64 <= size) {
                // Prefetch for multiply operation
                if (i + PREFETCH_DISTANCE < size) {
                    _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                }
                
                const __m256 va0 = _mm256_load_ps(this_data + i + 0);
                const __m256 vb0 = _mm256_load_ps(other_data + i + 0);
                const __m256 va1 = _mm256_load_ps(this_data + i + 8);
                const __m256 vb1 = _mm256_load_ps(other_data + i + 8);
                const __m256 va2 = _mm256_load_ps(this_data + i + 16);
                const __m256 vb2 = _mm256_load_ps(other_data + i + 16);
                const __m256 va3 = _mm256_load_ps(this_data + i + 24);
                const __m256 vb3 = _mm256_load_ps(other_data + i + 24);
                const __m256 va4 = _mm256_load_ps(this_data + i + 32);
                const __m256 vb4 = _mm256_load_ps(other_data + i + 32);
                const __m256 va5 = _mm256_load_ps(this_data + i + 40);
                const __m256 vb5 = _mm256_load_ps(other_data + i + 40);
                const __m256 va6 = _mm256_load_ps(this_data + i + 48);
                const __m256 vb6 = _mm256_load_ps(other_data + i + 48);
                const __m256 va7 = _mm256_load_ps(this_data + i + 56);
                const __m256 vb7 = _mm256_load_ps(other_data + i + 56);
                _mm256_store_ps(this_data + i + 0,  _mm256_mul_ps(va0, vb0));
                _mm256_store_ps(this_data + i + 8,  _mm256_mul_ps(va1, vb1));
                _mm256_store_ps(this_data + i + 16, _mm256_mul_ps(va2, vb2));
                _mm256_store_ps(this_data + i + 24, _mm256_mul_ps(va3, vb3));
                _mm256_store_ps(this_data + i + 32, _mm256_mul_ps(va4, vb4));
                _mm256_store_ps(this_data + i + 40, _mm256_mul_ps(va5, vb5));
                _mm256_store_ps(this_data + i + 48, _mm256_mul_ps(va6, vb6));
                _mm256_store_ps(this_data + i + 56, _mm256_mul_ps(va7, vb7));
                i += 64;
            }
            if (i + 32 <= size) {
                const __m256 va0 = _mm256_load_ps(this_data + i);
                const __m256 vb0 = _mm256_load_ps(other_data + i);
                const __m256 va1 = _mm256_load_ps(this_data + i + 8);
                const __m256 vb1 = _mm256_load_ps(other_data + i + 8);
                const __m256 va2 = _mm256_load_ps(this_data + i + 16);
                const __m256 vb2 = _mm256_load_ps(other_data + i + 16);
                const __m256 va3 = _mm256_load_ps(this_data + i + 24);
                const __m256 vb3 = _mm256_load_ps(other_data + i + 24);
                _mm256_store_ps(this_data + i, _mm256_mul_ps(va0, vb0));
                _mm256_store_ps(this_data + i + 8, _mm256_mul_ps(va1, vb1));
                _mm256_store_ps(this_data + i + 16, _mm256_mul_ps(va2, vb2));
                _mm256_store_ps(this_data + i + 24, _mm256_mul_ps(va3, vb3));
                i += 32;
            }
            if (i + 16 <= size) {
                const __m256 va0 = _mm256_load_ps(this_data + i);
                const __m256 vb0 = _mm256_load_ps(other_data + i);
                const __m256 va1 = _mm256_load_ps(this_data + i + 8);
                const __m256 vb1 = _mm256_load_ps(other_data + i + 8);
                _mm256_store_ps(this_data + i, _mm256_mul_ps(va0, vb0));
                _mm256_store_ps(this_data + i + 8, _mm256_mul_ps(va1, vb1));
                i += 16;
            }
            if (i + 8 <= size) {
                const __m256 va = _mm256_load_ps(this_data + i);
                const __m256 vb = _mm256_load_ps(other_data + i);
                _mm256_store_ps(this_data + i, _mm256_mul_ps(va, vb));
                i += 8;
            }
            if (i + 4 <= size) {
                const __m128 va = _mm_load_ps(this_data + i);
                const __m128 vb = _mm_load_ps(other_data + i);
                _mm_store_ps(this_data + i, _mm_mul_ps(va, vb));
                i += 4;
            }
        } else {
            while (i + 64 <= size) {
                // Prefetch for unaligned multiply
                if (i + PREFETCH_DISTANCE < size) {
                    _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                }
                
                const __m256 va0 = _mm256_loadu_ps(this_data + i + 0);
                const __m256 vb0 = _mm256_loadu_ps(other_data + i + 0);
                const __m256 va1 = _mm256_loadu_ps(this_data + i + 8);
                const __m256 vb1 = _mm256_loadu_ps(other_data + i + 8);
                const __m256 va2 = _mm256_loadu_ps(this_data + i + 16);
                const __m256 vb2 = _mm256_loadu_ps(other_data + i + 16);
                const __m256 va3 = _mm256_loadu_ps(this_data + i + 24);
                const __m256 vb3 = _mm256_loadu_ps(other_data + i + 24);
                const __m256 va4 = _mm256_loadu_ps(this_data + i + 32);
                const __m256 vb4 = _mm256_loadu_ps(other_data + i + 32);
                const __m256 va5 = _mm256_loadu_ps(this_data + i + 40);
                const __m256 vb5 = _mm256_loadu_ps(other_data + i + 40);
                const __m256 va6 = _mm256_loadu_ps(this_data + i + 48);
                const __m256 vb6 = _mm256_loadu_ps(other_data + i + 48);
                const __m256 va7 = _mm256_loadu_ps(this_data + i + 56);
                const __m256 vb7 = _mm256_loadu_ps(other_data + i + 56);
                _mm256_storeu_ps(this_data + i + 0, _mm256_mul_ps(va0, vb0));
                _mm256_storeu_ps(this_data + i + 8, _mm256_mul_ps(va1, vb1));
                _mm256_storeu_ps(this_data + i + 16, _mm256_mul_ps(va2, vb2));
                _mm256_storeu_ps(this_data + i + 24, _mm256_mul_ps(va3, vb3));
                _mm256_storeu_ps(this_data + i + 32, _mm256_mul_ps(va4, vb4));
                _mm256_storeu_ps(this_data + i + 40, _mm256_mul_ps(va5, vb5));
                _mm256_storeu_ps(this_data + i + 48, _mm256_mul_ps(va6, vb6));
                _mm256_storeu_ps(this_data + i + 56, _mm256_mul_ps(va7, vb7));
                i += 64;
            }
            if (i + 32 <= size) {
                const __m256 va0 = _mm256_loadu_ps(this_data + i);
                const __m256 vb0 = _mm256_loadu_ps(other_data + i);
                const __m256 va1 = _mm256_loadu_ps(this_data + i + 8);
                const __m256 vb1 = _mm256_loadu_ps(other_data + i + 8);
                const __m256 va2 = _mm256_loadu_ps(this_data + i + 16);
                const __m256 vb2 = _mm256_loadu_ps(other_data + i + 16);
                const __m256 va3 = _mm256_loadu_ps(this_data + i + 24);
                const __m256 vb3 = _mm256_loadu_ps(other_data + i + 24);
                _mm256_storeu_ps(this_data + i, _mm256_mul_ps(va0, vb0));
                _mm256_storeu_ps(this_data + i + 8, _mm256_mul_ps(va1, vb1));
                _mm256_storeu_ps(this_data + i + 16, _mm256_mul_ps(va2, vb2));
                _mm256_storeu_ps(this_data + i + 24, _mm256_mul_ps(va3, vb3));
                i += 32;
            }
            if (i + 16 <= size) {
                const __m256 va0 = _mm256_loadu_ps(this_data + i);
                const __m256 vb0 = _mm256_loadu_ps(other_data + i);
                const __m256 va1 = _mm256_loadu_ps(this_data + i + 8);
                const __m256 vb1 = _mm256_loadu_ps(other_data + i + 8);
                _mm256_storeu_ps(this_data + i, _mm256_mul_ps(va0, vb0));
                _mm256_storeu_ps(this_data + i + 8, _mm256_mul_ps(va1, vb1));
                i += 16;
            }
            if (i + 8 <= size) {
                const __m256 va = _mm256_loadu_ps(this_data + i);
                const __m256 vb = _mm256_loadu_ps(other_data + i);
                _mm256_storeu_ps(this_data + i, _mm256_mul_ps(va, vb));
                i += 8;
            }
            if (i + 4 <= size) {
                const __m128 va = _mm_loadu_ps(this_data + i);
                const __m128 vb = _mm_loadu_ps(other_data + i);
                _mm_storeu_ps(this_data + i, _mm_mul_ps(va, vb));
                i += 4;
            }
        }
        while (i < size) { this_data[i] *= other_data[i]; ++i; }
        return;
    }
    if (op_type == OperationType::DIV) {
        if (aligned32) {
            while (i + 64 <= size) {
                // Prefetch for divide operation
                if (i + PREFETCH_DISTANCE < size) {
                    _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                }
                
                const __m256 va0 = _mm256_load_ps(this_data + i + 0);
                const __m256 vb0 = _mm256_load_ps(other_data + i + 0);
                const __m256 va1 = _mm256_load_ps(this_data + i + 8);
                const __m256 vb1 = _mm256_load_ps(other_data + i + 8);
                const __m256 va2 = _mm256_load_ps(this_data + i + 16);
                const __m256 vb2 = _mm256_load_ps(other_data + i + 16);
                const __m256 va3 = _mm256_load_ps(this_data + i + 24);
                const __m256 vb3 = _mm256_load_ps(other_data + i + 24);
                const __m256 va4 = _mm256_load_ps(this_data + i + 32);
                const __m256 vb4 = _mm256_load_ps(other_data + i + 32);
                const __m256 va5 = _mm256_load_ps(this_data + i + 40);
                const __m256 vb5 = _mm256_load_ps(other_data + i + 40);
                const __m256 va6 = _mm256_load_ps(this_data + i + 48);
                const __m256 vb6 = _mm256_load_ps(other_data + i + 48);
                const __m256 va7 = _mm256_load_ps(this_data + i + 56);
                const __m256 vb7 = _mm256_load_ps(other_data + i + 56);
                _mm256_store_ps(this_data + i + 0,  _mm256_div_ps(va0, vb0));
                _mm256_store_ps(this_data + i + 8,  _mm256_div_ps(va1, vb1));
                _mm256_store_ps(this_data + i + 16, _mm256_div_ps(va2, vb2));
                _mm256_store_ps(this_data + i + 24, _mm256_div_ps(va3, vb3));
                _mm256_store_ps(this_data + i + 32, _mm256_div_ps(va4, vb4));
                _mm256_store_ps(this_data + i + 40, _mm256_div_ps(va5, vb5));
                _mm256_store_ps(this_data + i + 48, _mm256_div_ps(va6, vb6));
                _mm256_store_ps(this_data + i + 56, _mm256_div_ps(va7, vb7));
                i += 64;
            }
            if (i + 32 <= size) {
                const __m256 va0 = _mm256_load_ps(this_data + i);
                const __m256 vb0 = _mm256_load_ps(other_data + i);
                const __m256 va1 = _mm256_load_ps(this_data + i + 8);
                const __m256 vb1 = _mm256_load_ps(other_data + i + 8);
                const __m256 va2 = _mm256_load_ps(this_data + i + 16);
                const __m256 vb2 = _mm256_load_ps(other_data + i + 16);
                const __m256 va3 = _mm256_load_ps(this_data + i + 24);
                const __m256 vb3 = _mm256_load_ps(other_data + i + 24);
                _mm256_store_ps(this_data + i, _mm256_div_ps(va0, vb0));
                _mm256_store_ps(this_data + i + 8, _mm256_div_ps(va1, vb1));
                _mm256_store_ps(this_data + i + 16, _mm256_div_ps(va2, vb2));
                _mm256_store_ps(this_data + i + 24, _mm256_div_ps(va3, vb3));
                i += 32;
            }
            if (i + 16 <= size) {
                const __m256 va0 = _mm256_load_ps(this_data + i);
                const __m256 vb0 = _mm256_load_ps(other_data + i);
                const __m256 va1 = _mm256_load_ps(this_data + i + 8);
                const __m256 vb1 = _mm256_load_ps(other_data + i + 8);
                _mm256_store_ps(this_data + i, _mm256_div_ps(va0, vb0));
                _mm256_store_ps(this_data + i + 8, _mm256_div_ps(va1, vb1));
                i += 16;
            }
            if (i + 8 <= size) {
                const __m256 va = _mm256_load_ps(this_data + i);
                const __m256 vb = _mm256_load_ps(other_data + i);
                _mm256_store_ps(this_data + i, _mm256_div_ps(va, vb));
                i += 8;
            }
            if (i + 4 <= size) {
                const __m128 va = _mm_load_ps(this_data + i);
                const __m128 vb = _mm_load_ps(other_data + i);
                _mm_store_ps(this_data + i, _mm_div_ps(va, vb));
                i += 4;
            }
        } else {
            while (i + 64 <= size) {
                // Prefetch for unaligned divide
                if (i + PREFETCH_DISTANCE < size) {
                    _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                }
                
                const __m256 va0 = _mm256_loadu_ps(this_data + i + 0);
                const __m256 vb0 = _mm256_loadu_ps(other_data + i + 0);
                const __m256 va1 = _mm256_loadu_ps(this_data + i + 8);
                const __m256 vb1 = _mm256_loadu_ps(other_data + i + 8);
                const __m256 va2 = _mm256_loadu_ps(this_data + i + 16);
                const __m256 vb2 = _mm256_loadu_ps(other_data + i + 16);
                const __m256 va3 = _mm256_loadu_ps(this_data + i + 24);
                const __m256 vb3 = _mm256_loadu_ps(other_data + i + 24);
                const __m256 va4 = _mm256_loadu_ps(this_data + i + 32);
                const __m256 vb4 = _mm256_loadu_ps(other_data + i + 32);
                const __m256 va5 = _mm256_loadu_ps(this_data + i + 40);
                const __m256 vb5 = _mm256_loadu_ps(other_data + i + 40);
                const __m256 va6 = _mm256_loadu_ps(this_data + i + 48);
                const __m256 vb6 = _mm256_loadu_ps(other_data + i + 48);
                const __m256 va7 = _mm256_loadu_ps(this_data + i + 56);
                const __m256 vb7 = _mm256_loadu_ps(other_data + i + 56);
                _mm256_storeu_ps(this_data + i + 0,  _mm256_div_ps(va0, vb0));
                _mm256_storeu_ps(this_data + i + 8,  _mm256_div_ps(va1, vb1));
                _mm256_storeu_ps(this_data + i + 16, _mm256_div_ps(va2, vb2));
                _mm256_storeu_ps(this_data + i + 24, _mm256_div_ps(va3, vb3));
                _mm256_storeu_ps(this_data + i + 32, _mm256_div_ps(va4, vb4));
                _mm256_storeu_ps(this_data + i + 40, _mm256_div_ps(va5, vb5));
                _mm256_storeu_ps(this_data + i + 48, _mm256_div_ps(va6, vb6));
                _mm256_storeu_ps(this_data + i + 56, _mm256_div_ps(va7, vb7));
                i += 64;
            }
            if (i + 32 <= size) {
                const __m256 va0 = _mm256_loadu_ps(this_data + i);
                const __m256 vb0 = _mm256_loadu_ps(other_data + i);
                const __m256 va1 = _mm256_loadu_ps(this_data + i + 8);
                const __m256 vb1 = _mm256_loadu_ps(other_data + i + 8);
                const __m256 va2 = _mm256_loadu_ps(this_data + i + 16);
                const __m256 vb2 = _mm256_loadu_ps(other_data + i + 16);
                const __m256 va3 = _mm256_loadu_ps(this_data + i + 24);
                const __m256 vb3 = _mm256_loadu_ps(other_data + i + 24);
                _mm256_storeu_ps(this_data + i, _mm256_div_ps(va0, vb0));
                _mm256_storeu_ps(this_data + i + 8, _mm256_div_ps(va1, vb1));
                _mm256_storeu_ps(this_data + i + 16, _mm256_div_ps(va2, vb2));
                _mm256_storeu_ps(this_data + i + 24, _mm256_div_ps(va3, vb3));
                i += 32;
            }
            if (i + 16 <= size) {
                const __m256 va0 = _mm256_loadu_ps(this_data + i);
                const __m256 vb0 = _mm256_loadu_ps(other_data + i);
                const __m256 va1 = _mm256_loadu_ps(this_data + i + 8);
                const __m256 vb1 = _mm256_loadu_ps(other_data + i + 8);
                _mm256_storeu_ps(this_data + i, _mm256_div_ps(va0, vb0));
                _mm256_storeu_ps(this_data + i + 8, _mm256_div_ps(va1, vb1));
                i += 16;
            }
            if (i + 8 <= size) {
                const __m256 va = _mm256_loadu_ps(this_data + i);
                const __m256 vb = _mm256_loadu_ps(other_data + i);
                _mm256_storeu_ps(this_data + i, _mm256_div_ps(va, vb));
                i += 8;
            }
            if (i + 4 <= size) {
                const __m128 va = _mm_loadu_ps(this_data + i);
                const __m128 vb = _mm_loadu_ps(other_data + i);
                _mm_storeu_ps(this_data + i, _mm_div_ps(va, vb));
                i += 4;
            }
        }
        while (i < size) { this_data[i] /= other_data[i]; ++i; }
        return;
    }
    // SUB (default)
    if (aligned32) {
        while (i + 64 <= size) {
            // Prefetch for subtract operation
            if (i + PREFETCH_DISTANCE < size) {
                _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
            }
            
            const __m256 va0 = _mm256_load_ps(this_data + i + 0);
            const __m256 vb0 = _mm256_load_ps(other_data + i + 0);
            const __m256 va1 = _mm256_load_ps(this_data + i + 8);
            const __m256 vb1 = _mm256_load_ps(other_data + i + 8);
            const __m256 va2 = _mm256_load_ps(this_data + i + 16);
            const __m256 vb2 = _mm256_load_ps(other_data + i + 16);
            const __m256 va3 = _mm256_load_ps(this_data + i + 24);
            const __m256 vb3 = _mm256_load_ps(other_data + i + 24);
            const __m256 va4 = _mm256_load_ps(this_data + i + 32);
            const __m256 vb4 = _mm256_load_ps(other_data + i + 32);
            const __m256 va5 = _mm256_load_ps(this_data + i + 40);
            const __m256 vb5 = _mm256_load_ps(other_data + i + 40);
            const __m256 va6 = _mm256_load_ps(this_data + i + 48);
            const __m256 vb6 = _mm256_load_ps(other_data + i + 48);
            const __m256 va7 = _mm256_load_ps(this_data + i + 56);
            const __m256 vb7 = _mm256_load_ps(other_data + i + 56);
            _mm256_store_ps(this_data + i + 0,  _mm256_sub_ps(va0, vb0));
            _mm256_store_ps(this_data + i + 8,  _mm256_sub_ps(va1, vb1));
            _mm256_store_ps(this_data + i + 16, _mm256_sub_ps(va2, vb2));
            _mm256_store_ps(this_data + i + 24, _mm256_sub_ps(va3, vb3));
            _mm256_store_ps(this_data + i + 32, _mm256_sub_ps(va4, vb4));
            _mm256_store_ps(this_data + i + 40, _mm256_sub_ps(va5, vb5));
            _mm256_store_ps(this_data + i + 48, _mm256_sub_ps(va6, vb6));
            _mm256_store_ps(this_data + i + 56, _mm256_sub_ps(va7, vb7));
            i += 64;
        }
        if (i + 32 <= size) {
            const __m256 va0 = _mm256_load_ps(this_data + i);
            const __m256 vb0 = _mm256_load_ps(other_data + i);
            const __m256 va1 = _mm256_load_ps(this_data + i + 8);
            const __m256 vb1 = _mm256_load_ps(other_data + i + 8);
            const __m256 va2 = _mm256_load_ps(this_data + i + 16);
            const __m256 vb2 = _mm256_load_ps(other_data + i + 16);
            const __m256 va3 = _mm256_load_ps(this_data + i + 24);
            const __m256 vb3 = _mm256_load_ps(other_data + i + 24);
            _mm256_store_ps(this_data + i, _mm256_sub_ps(va0, vb0));
            _mm256_store_ps(this_data + i + 8, _mm256_sub_ps(va1, vb1));
            _mm256_store_ps(this_data + i + 16, _mm256_sub_ps(va2, vb2));
            _mm256_store_ps(this_data + i + 24, _mm256_sub_ps(va3, vb3));
            i += 32;
        }
        if (i + 16 <= size) {
            const __m256 va0 = _mm256_load_ps(this_data + i);
            const __m256 vb0 = _mm256_load_ps(other_data + i);
            const __m256 va1 = _mm256_load_ps(this_data + i + 8);
            const __m256 vb1 = _mm256_load_ps(other_data + i + 8);
            _mm256_store_ps(this_data + i, _mm256_sub_ps(va0, vb0));
            _mm256_store_ps(this_data + i + 8, _mm256_sub_ps(va1, vb1));
            i += 16;
        }
        if (i + 8 <= size) {
            const __m256 va = _mm256_load_ps(this_data + i);
            const __m256 vb = _mm256_load_ps(other_data + i);
            _mm256_store_ps(this_data + i, _mm256_sub_ps(va, vb));
            i += 8;
        }
        if (i + 4 <= size) {
            const __m128 va = _mm_load_ps(this_data + i);
            const __m128 vb = _mm_load_ps(other_data + i);
            _mm_store_ps(this_data + i, _mm_sub_ps(va, vb));
            i += 4;
        }
    } else {
        while (i + 64 <= size) {
            // Prefetch for unaligned subtract
            if (i + PREFETCH_DISTANCE < size) {
                _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
            }
            
            const __m256 va0 = _mm256_loadu_ps(this_data + i + 0);
            const __m256 vb0 = _mm256_loadu_ps(other_data + i + 0);
            const __m256 va1 = _mm256_loadu_ps(this_data + i + 8);
            const __m256 vb1 = _mm256_loadu_ps(other_data + i + 8);
            const __m256 va2 = _mm256_loadu_ps(this_data + i + 16);
            const __m256 vb2 = _mm256_loadu_ps(other_data + i + 16);
            const __m256 va3 = _mm256_loadu_ps(this_data + i + 24);
            const __m256 vb3 = _mm256_loadu_ps(other_data + i + 24);
            const __m256 va4 = _mm256_loadu_ps(this_data + i + 32);
            const __m256 vb4 = _mm256_loadu_ps(other_data + i + 32);
            const __m256 va5 = _mm256_loadu_ps(this_data + i + 40);
            const __m256 vb5 = _mm256_loadu_ps(other_data + i + 40);
            const __m256 va6 = _mm256_loadu_ps(this_data + i + 48);
            const __m256 vb6 = _mm256_loadu_ps(other_data + i + 48);
            const __m256 va7 = _mm256_loadu_ps(this_data + i + 56);
            const __m256 vb7 = _mm256_loadu_ps(other_data + i + 56);
            _mm256_storeu_ps(this_data + i + 0,  _mm256_sub_ps(va0, vb0));
            _mm256_storeu_ps(this_data + i + 8,  _mm256_sub_ps(va1, vb1));
            _mm256_storeu_ps(this_data + i + 16, _mm256_sub_ps(va2, vb2));
            _mm256_storeu_ps(this_data + i + 24, _mm256_sub_ps(va3, vb3));
            _mm256_storeu_ps(this_data + i + 32, _mm256_sub_ps(va4, vb4));
            _mm256_storeu_ps(this_data + i + 40, _mm256_sub_ps(va5, vb5));
            _mm256_storeu_ps(this_data + i + 48, _mm256_sub_ps(va6, vb6));
            _mm256_storeu_ps(this_data + i + 56, _mm256_sub_ps(va7, vb7));
            i += 64;
        }
        if (i + 32 <= size) {
            const __m256 va0 = _mm256_loadu_ps(this_data + i);
            const __m256 vb0 = _mm256_loadu_ps(other_data + i);
            const __m256 va1 = _mm256_loadu_ps(this_data + i + 8);
            const __m256 vb1 = _mm256_loadu_ps(other_data + i + 8);
            const __m256 va2 = _mm256_loadu_ps(this_data + i + 16);
            const __m256 vb2 = _mm256_loadu_ps(other_data + i + 16);
            const __m256 va3 = _mm256_loadu_ps(this_data + i + 24);
            const __m256 vb3 = _mm256_loadu_ps(other_data + i + 24);
            _mm256_storeu_ps(this_data + i, _mm256_sub_ps(va0, vb0));
            _mm256_storeu_ps(this_data + i + 8, _mm256_sub_ps(va1, vb1));
            _mm256_storeu_ps(this_data + i + 16, _mm256_sub_ps(va2, vb2));
            _mm256_storeu_ps(this_data + i + 24, _mm256_sub_ps(va3, vb3));
            i += 32;
        }
        if (i + 16 <= size) {
            const __m256 va0 = _mm256_loadu_ps(this_data + i);
            const __m256 vb0 = _mm256_loadu_ps(other_data + i);
            const __m256 va1 = _mm256_loadu_ps(this_data + i + 8);
            const __m256 vb1 = _mm256_loadu_ps(other_data + i + 8);
            _mm256_storeu_ps(this_data + i, _mm256_sub_ps(va0, vb0));
            _mm256_storeu_ps(this_data + i + 8, _mm256_sub_ps(va1, vb1));
            i += 16;
        }
        if (i + 8 <= size) {
            const __m256 va = _mm256_loadu_ps(this_data + i);
            const __m256 vb = _mm256_loadu_ps(other_data + i);
            _mm256_storeu_ps(this_data + i, _mm256_sub_ps(va, vb));
            i += 8;
        }
        if (i + 4 <= size) {
            const __m128 va = _mm_loadu_ps(this_data + i);
            const __m128 vb = _mm_loadu_ps(other_data + i);
            _mm_storeu_ps(this_data + i, _mm_sub_ps(va, vb));
            i += 4;
        }
    }
    while (i < size) { this_data[i] -= other_data[i]; ++i; }
}

// =======================
// INT8 kernels
// =======================

} // namespace labneura

#endif // __AVX2__