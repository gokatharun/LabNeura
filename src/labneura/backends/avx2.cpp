#if defined(__AVX2__)

#include "labneura/backends/avx2.h"
#include <immintrin.h>
#include <algorithm>
#include <cstdlib>
#include "labneura/utils/alignment.hpp"

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

// Use alignment utility from labneura::util

// =======================
// Constructor
// =======================

AVX2Backend::AVX2Backend(std::size_t size, QuantizationMode mode)
    : size_(size), quantization_mode_(mode) {

    // AVX2: 256-bit registers → 8 FP32 elements
    aligned_size_ = ((size + 7) / 8) * 8;

    if (mode == QuantizationMode::FP32) {
        float* p = labneura::util::allocate_aligned<float>(aligned_size_, labneura::util::AVX2_ALIGN);
        data_fp32_ = std::unique_ptr<float, void(*)(float*)>(p, labneura::util::free_aligned_float);
        std::fill(data_fp32_.get() + size_, data_fp32_.get() + aligned_size_, 0.0f);
    } else if (mode == QuantizationMode::FP16) {
        int16_t* p = labneura::util::allocate_aligned<int16_t>(aligned_size_, labneura::util::AVX2_ALIGN);
        auto fp16_deleter = [](int16_t* ptr){ labneura::util::aligned_free(ptr); };
        data_fp16_ = std::unique_ptr<int16_t, void(*)(int16_t*)>(p, fp16_deleter);
        std::fill(data_fp16_.get() + size_, data_fp16_.get() + aligned_size_, static_cast<int16_t>(0));
    } else if (mode == QuantizationMode::INT16) {
        int16_t* p = labneura::util::allocate_aligned<int16_t>(aligned_size_, labneura::util::AVX2_ALIGN);
        auto int16_deleter = [](int16_t* ptr){ labneura::util::aligned_free(ptr); };
        data_int16_ = std::unique_ptr<int16_t, void(*)(int16_t*)>(p, int16_deleter);
        std::fill(data_int16_.get() + size_, data_int16_.get() + aligned_size_, static_cast<int16_t>(0));
    } else {
        int8_t* p = labneura::util::allocate_aligned<int8_t>(aligned_size_, labneura::util::AVX2_ALIGN);
        data_int8_ = std::unique_ptr<int8_t, void(*)(int8_t*)>(p, labneura::util::free_aligned_int8);
        std::fill(data_int8_.get() + size_, data_int8_.get() + aligned_size_, 0);
    }
}

// =======================
// Metadata
// =======================

std::size_t AVX2Backend::size() const {
    return size_;
}

QuantizationMode AVX2Backend::quantization_mode() const {
    return quantization_mode_;
}

// =======================
// Data access
// =======================

float* AVX2Backend::data_fp32() {
    if (quantization_mode_ != QuantizationMode::FP32) {
        throw std::runtime_error("Tensor is not in FP32 mode");
    }
    return data_fp32_.get();
}

const float* AVX2Backend::data_fp32() const {
    if (quantization_mode_ != QuantizationMode::FP32) {
        throw std::runtime_error("Tensor is not in FP32 mode");
    }
    return data_fp32_.get();
}

int16_t* AVX2Backend::data_fp16() {
    if (quantization_mode_ != QuantizationMode::FP16) {
        throw std::runtime_error("Tensor is not in FP16 mode");
    }
    return data_fp16_.get();
}

const int16_t* AVX2Backend::data_fp16() const {
    if (quantization_mode_ != QuantizationMode::FP16) {
        throw std::runtime_error("Tensor is not in FP16 mode");
    }
    return data_fp16_.get();
}

int16_t* AVX2Backend::data_int16() {
    if (quantization_mode_ != QuantizationMode::INT16) {
        throw std::runtime_error("Tensor is not in INT16 mode");
    }
    return data_int16_.get();
}

const int16_t* AVX2Backend::data_int16() const {
    if (quantization_mode_ != QuantizationMode::INT16) {
        throw std::runtime_error("Tensor is not in INT16 mode");
    }
    return data_int16_.get();
}

int8_t* AVX2Backend::data_int8() {
    if (quantization_mode_ != QuantizationMode::INT8) {
        throw std::runtime_error("Tensor is not in INT8 mode");
    }
    return data_int8_.get();
}

const int8_t* AVX2Backend::data_int8() const {
    if (quantization_mode_ != QuantizationMode::INT8) {
        throw std::runtime_error("Tensor is not in INT8 mode");
    }
    return data_int8_.get();
}

// =======================
// Clone
// =======================

std::unique_ptr<TensorBackend> AVX2Backend::clone() const {
    auto backend = std::make_unique<AVX2Backend>(size_, quantization_mode_);
    if (quantization_mode_ == QuantizationMode::FP32) {
        std::copy(data_fp32_.get(), data_fp32_.get() + size_, backend->data_fp32());
    } else if (quantization_mode_ == QuantizationMode::FP16) {
        std::copy(data_fp16_.get(), data_fp16_.get() + size_, backend->data_fp16());
    } else if (quantization_mode_ == QuantizationMode::INT16) {
        std::copy(data_int16_.get(), data_int16_.get() + size_, backend->data_int16());
    } else {
        std::copy(data_int8_.get(), data_int8_.get() + size_, backend->data_int8());
    }
    return backend;
}

// =======================
// FP32 kernels
// =======================

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

void AVX2Backend::operation_fp16(const TensorBackend& other, OperationType op_type) {
    // FP16 treated as int16_t for bitwise SIMD operations
    const int16_t* other_data = other.data_fp16();
    int16_t* this_data = data_fp16();
    std::size_t i = 0;
    const std::size_t size = size_;
    
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

void AVX2Backend::operation_int16(const TensorBackend& other, OperationType op_type) {
    const int16_t* other_data = other.data_int16();
    int16_t* this_data = data_int16();
    std::size_t i = 0;
    const std::size_t size = size_;
    
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
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), r);
            i += 16;
        }
        if (i + 8 <= size) {
            const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(this_data + i));
            const __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(other_data + i));
            const __m128i r = _mm_adds_epi16(va, vb);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(this_data + i), r);
            i += 8;
        }
        if (i + 4 <= size) {
            for (int j = 0; j < 4; ++j) {
                int32_t sum = static_cast<int32_t>(this_data[i + j]) + static_cast<int32_t>(other_data[i + j]);
                this_data[i + j] = static_cast<int16_t>(std::max(-32768, std::min(32767, sum)));
            }
            i += 4;
        }
        
        // Scalar remainder
        while (i < size) {
            int32_t sum = static_cast<int32_t>(this_data[i]) + static_cast<int32_t>(other_data[i]);
            this_data[i] = static_cast<int16_t>(std::max(-32768, std::min(32767, sum)));
            ++i;
        }
    } else if (op_type == OperationType::MUL) {
        // Main loop: 64 int16s per iteration with widening multiply
        while (i < vec_len_64) {
            // Prefetch ahead
            if (i + PREFETCH_DISTANCE < size) {
                _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
            }
            
            // Process 64 int16s = 4 × __m256i
            // Strategy: Load 16 int16s, widen to 32-bit, multiply, saturate narrow back to 16-bit
            const __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 0));
            const __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 0));
            const __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
            const __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
            const __m256i va2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 32));
            const __m256i vb2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 32));
            const __m256i va3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 48));
            const __m256i vb3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 48));
            
            // Multiply with saturation using _mm256_mulhrs_epi16 (signed multiply high with round and saturate)
            const __m256i r0 = _mm256_mulhrs_epi16(va0, vb0);
            const __m256i r1 = _mm256_mulhrs_epi16(va1, vb1);
            const __m256i r2 = _mm256_mulhrs_epi16(va2, vb2);
            const __m256i r3 = _mm256_mulhrs_epi16(va3, vb3);
            
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 0), r0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 16), r1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 32), r2);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 48), r3);
            
            i += 64;
        }
        
        // Tail chain: 32 -> 16 -> 8 -> 4 -> scalar
        if (i < vec_len_32) {
            const __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
            const __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
            const __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
            const __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
            const __m256i r0 = _mm256_mulhrs_epi16(va0, vb0);
            const __m256i r1 = _mm256_mulhrs_epi16(va1, vb1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), r0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 16), r1);
            i += 32;
        }
        if (i < vec_len_16) {
            const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
            const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
            const __m256i r = _mm256_mulhrs_epi16(va, vb);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), r);
            i += 16;
        }
        if (i + 8 <= size) {
            const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(this_data + i));
            const __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(other_data + i));
            const __m128i r = _mm_mulhrs_epi16(va, vb);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(this_data + i), r);
            i += 8;
        }
        if (i + 4 <= size) {
            for (int j = 0; j < 4; ++j) {
                int32_t product = static_cast<int32_t>(this_data[i + j]) * static_cast<int32_t>(other_data[i + j]);
                this_data[i + j] = static_cast<int16_t>(std::max(-32768, std::min(32767, product)));
            }
            i += 4;
        }
        
        // Scalar remainder
        while (i < size) {
            int32_t product = static_cast<int32_t>(this_data[i]) * static_cast<int32_t>(other_data[i]);
            this_data[i] = static_cast<int16_t>(std::max(-32768, std::min(32767, product)));
            ++i;
        }
    } else {  // SUB
        // Main loop: 64 int16s per iteration
        while (i < vec_len_64) {
            // Prefetch ahead
            if (i + PREFETCH_DISTANCE < size) {
                _mm_prefetch(reinterpret_cast<const char*>(this_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(other_data + i + PREFETCH_DISTANCE), _MM_HINT_T0);
            }
            
            // Process 64 int16s = 4 × __m256i
            const __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 0));
            const __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 0));
            const __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
            const __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
            const __m256i va2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 32));
            const __m256i vb2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 32));
            const __m256i va3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 48));
            const __m256i vb3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 48));
            
            // Saturating subtract for int16
            const __m256i r0 = _mm256_subs_epi16(va0, vb0);
            const __m256i r1 = _mm256_subs_epi16(va1, vb1);
            const __m256i r2 = _mm256_subs_epi16(va2, vb2);
            const __m256i r3 = _mm256_subs_epi16(va3, vb3);
            
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 0), r0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 16), r1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 32), r2);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 48), r3);
            
            i += 64;
        }
        
        // Tail chain: 32 -> 16 -> 8 -> 4 -> scalar
        if (i < vec_len_32) {
            const __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
            const __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
            const __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i + 16));
            const __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i + 16));
            const __m256i r0 = _mm256_subs_epi16(va0, vb0);
            const __m256i r1 = _mm256_subs_epi16(va1, vb1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), r0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i + 16), r1);
            i += 32;
        }
        if (i < vec_len_16) {
            const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
            const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
            const __m256i r = _mm256_subs_epi16(va, vb);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), r);
            i += 16;
        }
        if (i + 8 <= size) {
            const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(this_data + i));
            const __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(other_data + i));
            const __m128i r = _mm_subs_epi16(va, vb);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(this_data + i), r);
            i += 8;
        }
        if (i + 4 <= size) {
            for (int j = 0; j < 4; ++j) {
                int32_t diff = static_cast<int32_t>(this_data[i + j]) - static_cast<int32_t>(other_data[i + j]);
                this_data[i + j] = static_cast<int16_t>(std::max(-32768, std::min(32767, diff)));
            }
            i += 4;
        }
        
        // Scalar remainder
        while (i < size) {
            int32_t diff = static_cast<int32_t>(this_data[i]) - static_cast<int32_t>(other_data[i]);
            this_data[i] = static_cast<int16_t>(std::max(-32768, std::min(32767, diff)));
            ++i;
        }
    }
}

// =======================
// Public dispatch
// =======================

} // namespace labneura

#endif // __AVX2__