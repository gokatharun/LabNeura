#if defined(__SSE4_1__)

#include "labneura/backends/sse41.h"
#include <smmintrin.h>
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

// Detect number of available SSE4.1 (XMM) registers using CPUID
// Returns 32 if AVX-512 is supported (extends XMM registers)
// Returns 16 for standard x86_64 (default)
static inline int detect_sse41_registers() {
    int num_registers = 16;  // Default: standard x86_64

    try {
#if defined(__GNUC__) || defined(__clang__)
        unsigned int eax, ebx, ecx, edx;
        
        // Check CPUID support
        if (__get_cpuid_max(0, nullptr) >= 7) {
            // CPUID leaf 7, subleaf 0: Extended Features
            if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
                // Check EBX bit 16: AVX-512F (Foundation)
                // AVX-512 extends XMM registers from 16 to 32
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
static inline int get_sse41_unroll_factor() {
    return detect_sse41_registers() / 2;  // 8 registers per operand
}

// FP32: 4 elements per XMM, unroll_factor=8 → 32 floats per iteration
static inline std::size_t get_sse41_fp32_chunk() {
    return get_sse41_unroll_factor() * 4;
}

// Use alignment utility from labneura::util

// =======================
// Constructor
// =======================

SSE41Backend::SSE41Backend(std::size_t size, QuantizationMode mode)
    : size_(size), quantization_mode_(mode) {

    // SSE = 128-bit → 4 FP32
    aligned_size_ = ((size + 3) / 4) * 4;

    if (mode == QuantizationMode::FP32) {
        float* p = labneura::util::allocate_aligned<float>(aligned_size_, labneura::util::SSE_ALIGN);
        data_fp32_ = std::unique_ptr<float, void(*)(float*)>(p, labneura::util::free_aligned_float);
        std::fill(data_fp32_.get() + size_, data_fp32_.get() + aligned_size_, 0.0f);
    } else {
        int8_t* p = labneura::util::allocate_aligned<int8_t>(aligned_size_, labneura::util::SSE_ALIGN);
        data_int8_ = std::unique_ptr<int8_t, void(*)(int8_t*)>(p, labneura::util::free_aligned_int8);
        std::fill(data_int8_.get() + size_, data_int8_.get() + aligned_size_, 0);
    }
}

// =======================
// Metadata
// =======================

std::size_t SSE41Backend::size() const {
    return size_;
}

QuantizationMode SSE41Backend::quantization_mode() const {
    return quantization_mode_;
}

// =======================
// Data access
// =======================

float* SSE41Backend::data_fp32() {
    if (quantization_mode_ != QuantizationMode::FP32) {
        throw std::runtime_error("Tensor is not in FP32 mode");
    }
    return data_fp32_.get();
}

const float* SSE41Backend::data_fp32() const {
    if (quantization_mode_ != QuantizationMode::FP32) {
        throw std::runtime_error("Tensor is not in FP32 mode");
    }
    return data_fp32_.get();
}

int8_t* SSE41Backend::data_int8() {
    if (quantization_mode_ != QuantizationMode::INT8) {
        throw std::runtime_error("Tensor is not in INT8 mode");
    }
    return data_int8_.get();
}

const int8_t* SSE41Backend::data_int8() const {
    if (quantization_mode_ != QuantizationMode::INT8) {
        throw std::runtime_error("Tensor is not in INT8 mode");
    }
    return data_int8_.get();
}

// =======================
// Clone
// =======================

std::unique_ptr<TensorBackend> SSE41Backend::clone() const {
    auto backend = std::make_unique<SSE41Backend>(size_, quantization_mode_);
    if (quantization_mode_ == QuantizationMode::FP32) {
        std::copy(data_fp32_.get(), data_fp32_.get() + size_, backend->data_fp32());
    } else {
        std::copy(data_int8_.get(), data_int8_.get() + size_, backend->data_int8());
    }
    return backend;
}

// =======================
// FP32 kernels
// =======================

void SSE41Backend::operation_fp32(const TensorBackend& other, OperationType op) {
    const float* o = other.data_fp32();
    float* d = data_fp32();
    std::size_t i = 0;
    const std::size_t size = size_;
    const bool aligned16 = labneura::util::is_aligned(d, labneura::util::SSE_ALIGN) &&
                           labneura::util::is_aligned(o, labneura::util::SSE_ALIGN);

    if (op == OperationType::ADD) {
        // Main loop: 32 floats/iteration (8 registers per operand, 4 ops per register)
        if (aligned16) {
                while (i + 32 <= size) {
                    const __m128 va0 = _mm_load_ps(d + i + 0);
                    const __m128 vb0 = _mm_load_ps(o + i + 0);
                    const __m128 va1 = _mm_load_ps(d + i + 4);
                    const __m128 vb1 = _mm_load_ps(o + i + 4);
                    const __m128 va2 = _mm_load_ps(d + i + 8);
                    const __m128 vb2 = _mm_load_ps(o + i + 8);
                    const __m128 va3 = _mm_load_ps(d + i + 12);
                    const __m128 vb3 = _mm_load_ps(o + i + 12);
                    const __m128 va4 = _mm_load_ps(d + i + 16);
                    const __m128 vb4 = _mm_load_ps(o + i + 16);
                    const __m128 va5 = _mm_load_ps(d + i + 20);
                    const __m128 vb5 = _mm_load_ps(o + i + 20);
                    const __m128 va6 = _mm_load_ps(d + i + 24);
                    const __m128 vb6 = _mm_load_ps(o + i + 24);
                    const __m128 va7 = _mm_load_ps(d + i + 28);
                    const __m128 vb7 = _mm_load_ps(o + i + 28);
                    _mm_store_ps(d + i + 0,  _mm_add_ps(va0, vb0));
                    _mm_store_ps(d + i + 4,  _mm_add_ps(va1, vb1));
                    _mm_store_ps(d + i + 8,  _mm_add_ps(va2, vb2));
                    _mm_store_ps(d + i + 12, _mm_add_ps(va3, vb3));
                    _mm_store_ps(d + i + 16, _mm_add_ps(va4, vb4));
                    _mm_store_ps(d + i + 20, _mm_add_ps(va5, vb5));
                    _mm_store_ps(d + i + 24, _mm_add_ps(va6, vb6));
                    _mm_store_ps(d + i + 28, _mm_add_ps(va7, vb7));
                    i += 32;
                }
                if (i + 16 <= size) {
                    const __m128 va0 = _mm_load_ps(d + i);
                    const __m128 vb0 = _mm_load_ps(o + i);
                    const __m128 va1 = _mm_load_ps(d + i + 4);
                    const __m128 vb1 = _mm_load_ps(o + i + 4);
                    const __m128 va2 = _mm_load_ps(d + i + 8);
                    const __m128 vb2 = _mm_load_ps(o + i + 8);
                    const __m128 va3 = _mm_load_ps(d + i + 12);
                    const __m128 vb3 = _mm_load_ps(o + i + 12);
                    _mm_store_ps(d + i, _mm_add_ps(va0, vb0));
                    _mm_store_ps(d + i + 4, _mm_add_ps(va1, vb1));
                    _mm_store_ps(d + i + 8, _mm_add_ps(va2, vb2));
                    _mm_store_ps(d + i + 12, _mm_add_ps(va3, vb3));
                    i += 16;
                }
                if (i + 8 <= size) {
                    const __m128 va0 = _mm_load_ps(d + i);
                    const __m128 vb0 = _mm_load_ps(o + i);
                    const __m128 va1 = _mm_load_ps(d + i + 4);
                    const __m128 vb1 = _mm_load_ps(o + i + 4);
                    _mm_store_ps(d + i, _mm_add_ps(va0, vb0));
                    _mm_store_ps(d + i + 4, _mm_add_ps(va1, vb1));
                    i += 8;
                }
                if (i + 4 <= size) {
                    const __m128 va = _mm_load_ps(d + i);
                    const __m128 vb = _mm_load_ps(o + i);
                    _mm_store_ps(d + i, _mm_add_ps(va, vb));
                    i += 4;
                }
            } else {
                while (i + 32 <= size) {
                    const __m128 va0 = _mm_loadu_ps(d + i + 0);
                    const __m128 vb0 = _mm_loadu_ps(o + i + 0);
                    const __m128 va1 = _mm_loadu_ps(d + i + 4);
                    const __m128 vb1 = _mm_loadu_ps(o + i + 4);
                    const __m128 va2 = _mm_loadu_ps(d + i + 8);
                    const __m128 vb2 = _mm_loadu_ps(o + i + 8);
                    const __m128 va3 = _mm_loadu_ps(d + i + 12);
                    const __m128 vb3 = _mm_loadu_ps(o + i + 12);
                    const __m128 va4 = _mm_loadu_ps(d + i + 16);
                    const __m128 vb4 = _mm_loadu_ps(o + i + 16);
                    const __m128 va5 = _mm_loadu_ps(d + i + 20);
                    const __m128 vb5 = _mm_loadu_ps(o + i + 20);
                    const __m128 va6 = _mm_loadu_ps(d + i + 24);
                    const __m128 vb6 = _mm_loadu_ps(o + i + 24);
                    const __m128 va7 = _mm_loadu_ps(d + i + 28);
                    const __m128 vb7 = _mm_loadu_ps(o + i + 28);
                    _mm_storeu_ps(d + i + 0, _mm_add_ps(va0, vb0));
                    _mm_storeu_ps(d + i + 4, _mm_add_ps(va1, vb1));
                    _mm_storeu_ps(d + i + 8, _mm_add_ps(va2, vb2));
                    _mm_storeu_ps(d + i + 12, _mm_add_ps(va3, vb3));
                    _mm_storeu_ps(d + i + 16, _mm_add_ps(va4, vb4));
                    _mm_storeu_ps(d + i + 20, _mm_add_ps(va5, vb5));
                    _mm_storeu_ps(d + i + 24, _mm_add_ps(va6, vb6));
                    _mm_storeu_ps(d + i + 28, _mm_add_ps(va7, vb7));
                    i += 32;
                }
                if (i + 16 <= size) {
                    const __m128 va0 = _mm_loadu_ps(d + i);
                    const __m128 vb0 = _mm_loadu_ps(o + i);
                    const __m128 va1 = _mm_loadu_ps(d + i + 4);
                    const __m128 vb1 = _mm_loadu_ps(o + i + 4);
                    const __m128 va2 = _mm_loadu_ps(d + i + 8);
                    const __m128 vb2 = _mm_loadu_ps(o + i + 8);
                    const __m128 va3 = _mm_loadu_ps(d + i + 12);
                    const __m128 vb3 = _mm_loadu_ps(o + i + 12);
                    _mm_storeu_ps(d + i, _mm_add_ps(va0, vb0));
                    _mm_storeu_ps(d + i + 4, _mm_add_ps(va1, vb1));
                    _mm_storeu_ps(d + i + 8, _mm_add_ps(va2, vb2));
                    _mm_storeu_ps(d + i + 12, _mm_add_ps(va3, vb3));
                    i += 16;
                }
                if (i + 8 <= size) {
                    const __m128 va0 = _mm_loadu_ps(d + i);
                    const __m128 vb0 = _mm_loadu_ps(o + i);
                    const __m128 va1 = _mm_loadu_ps(d + i + 4);
                    const __m128 vb1 = _mm_loadu_ps(o + i + 4);
                    _mm_storeu_ps(d + i, _mm_add_ps(va0, vb0));
                    _mm_storeu_ps(d + i + 4, _mm_add_ps(va1, vb1));
                    i += 8;
                }
                if (i + 4 <= size) {
                    const __m128 va = _mm_loadu_ps(d + i);
                    const __m128 vb = _mm_loadu_ps(o + i);
                    _mm_storeu_ps(d + i, _mm_add_ps(va, vb));
                    i += 4;
                }
            }
        while (i < size) { d[i] += o[i]; ++i; }
        return;
    }
    if (op == OperationType::MUL) {
        // Main loop: 32 floats/iteration
        while (i + 32 <= size) {
            const __m128 va0 = _mm_loadu_ps(d + i + 0);
            const __m128 vb0 = _mm_loadu_ps(o + i + 0);
            const __m128 va1 = _mm_loadu_ps(d + i + 4);
            const __m128 vb1 = _mm_loadu_ps(o + i + 4);
            const __m128 va2 = _mm_loadu_ps(d + i + 8);
            const __m128 vb2 = _mm_loadu_ps(o + i + 8);
            const __m128 va3 = _mm_loadu_ps(d + i + 12);
            const __m128 vb3 = _mm_loadu_ps(o + i + 12);
            const __m128 va4 = _mm_loadu_ps(d + i + 16);
            const __m128 vb4 = _mm_loadu_ps(o + i + 16);
            const __m128 va5 = _mm_loadu_ps(d + i + 20);
            const __m128 vb5 = _mm_loadu_ps(o + i + 20);
            const __m128 va6 = _mm_loadu_ps(d + i + 24);
            const __m128 vb6 = _mm_loadu_ps(o + i + 24);
            const __m128 va7 = _mm_loadu_ps(d + i + 28);
            const __m128 vb7 = _mm_loadu_ps(o + i + 28);
            if (aligned16) {
                _mm_store_ps(d + i + 0,  _mm_mul_ps(va0, vb0));
                _mm_store_ps(d + i + 4,  _mm_mul_ps(va1, vb1));
                _mm_store_ps(d + i + 8,  _mm_mul_ps(va2, vb2));
                _mm_store_ps(d + i + 12, _mm_mul_ps(va3, vb3));
                _mm_store_ps(d + i + 16, _mm_mul_ps(va4, vb4));
                _mm_store_ps(d + i + 20, _mm_mul_ps(va5, vb5));
                _mm_store_ps(d + i + 24, _mm_mul_ps(va6, vb6));
                _mm_store_ps(d + i + 28, _mm_mul_ps(va7, vb7));
            } else {
                _mm_storeu_ps(d + i + 0, _mm_mul_ps(va0, vb0));
                _mm_storeu_ps(d + i + 4, _mm_mul_ps(va1, vb1));
                _mm_storeu_ps(d + i + 8, _mm_mul_ps(va2, vb2));
                _mm_storeu_ps(d + i + 12, _mm_mul_ps(va3, vb3));
                _mm_storeu_ps(d + i + 16, _mm_mul_ps(va4, vb4));
                _mm_storeu_ps(d + i + 20, _mm_mul_ps(va5, vb5));
                _mm_storeu_ps(d + i + 24, _mm_mul_ps(va6, vb6));
                _mm_storeu_ps(d + i + 28, _mm_mul_ps(va7, vb7));
            }
            i += 32;
        }
        // Tail 16 floats
        if (i + 16 <= size) {
            const __m128 va0 = _mm_loadu_ps(d + i);
            const __m128 vb0 = _mm_loadu_ps(o + i);
            const __m128 va1 = _mm_loadu_ps(d + i + 4);
            const __m128 vb1 = _mm_loadu_ps(o + i + 4);
            const __m128 va2 = _mm_loadu_ps(d + i + 8);
            const __m128 vb2 = _mm_loadu_ps(o + i + 8);
            const __m128 va3 = _mm_loadu_ps(d + i + 12);
            const __m128 vb3 = _mm_loadu_ps(o + i + 12);
            _mm_storeu_ps(d + i, _mm_mul_ps(va0, vb0));
            _mm_storeu_ps(d + i + 4, _mm_mul_ps(va1, vb1));
            _mm_storeu_ps(d + i + 8, _mm_mul_ps(va2, vb2));
            _mm_storeu_ps(d + i + 12, _mm_mul_ps(va3, vb3));
            i += 16;
        }
        // Tail 8 floats
        if (i + 8 <= size) {
            const __m128 va0 = _mm_loadu_ps(d + i);
            const __m128 vb0 = _mm_loadu_ps(o + i);
            const __m128 va1 = _mm_loadu_ps(d + i + 4);
            const __m128 vb1 = _mm_loadu_ps(o + i + 4);
            _mm_storeu_ps(d + i, _mm_mul_ps(va0, vb0));
            _mm_storeu_ps(d + i + 4, _mm_mul_ps(va1, vb1));
            i += 8;
        }
        // Tail 4 floats
        if (i + 4 <= size) {
            const __m128 va = _mm_loadu_ps(d + i);
            const __m128 vb = _mm_loadu_ps(o + i);
            _mm_storeu_ps(d + i, _mm_mul_ps(va, vb));
            i += 4;
        }
        // Scalar tail
        while (i < size) {
            d[i] *= o[i];
            ++i;
        }
        return;
    }
    // SUB (default)
    while (i + 32 <= size) {
        const __m128 va0 = _mm_loadu_ps(d + i + 0);
        const __m128 vb0 = _mm_loadu_ps(o + i + 0);
        const __m128 va1 = _mm_loadu_ps(d + i + 4);
        const __m128 vb1 = _mm_loadu_ps(o + i + 4);
        const __m128 va2 = _mm_loadu_ps(d + i + 8);
        const __m128 vb2 = _mm_loadu_ps(o + i + 8);
        const __m128 va3 = _mm_loadu_ps(d + i + 12);
        const __m128 vb3 = _mm_loadu_ps(o + i + 12);
        const __m128 va4 = _mm_loadu_ps(d + i + 16);
        const __m128 vb4 = _mm_loadu_ps(o + i + 16);
        const __m128 va5 = _mm_loadu_ps(d + i + 20);
        const __m128 vb5 = _mm_loadu_ps(o + i + 20);
        const __m128 va6 = _mm_loadu_ps(d + i + 24);
        const __m128 vb6 = _mm_loadu_ps(o + i + 24);
        const __m128 va7 = _mm_loadu_ps(d + i + 28);
        const __m128 vb7 = _mm_loadu_ps(o + i + 28);
        if (aligned16) {
            _mm_store_ps(d + i + 0,  _mm_sub_ps(va0, vb0));
            _mm_store_ps(d + i + 4,  _mm_sub_ps(va1, vb1));
            _mm_store_ps(d + i + 8,  _mm_sub_ps(va2, vb2));
            _mm_store_ps(d + i + 12, _mm_sub_ps(va3, vb3));
            _mm_store_ps(d + i + 16, _mm_sub_ps(va4, vb4));
            _mm_store_ps(d + i + 20, _mm_sub_ps(va5, vb5));
            _mm_store_ps(d + i + 24, _mm_sub_ps(va6, vb6));
            _mm_store_ps(d + i + 28, _mm_sub_ps(va7, vb7));
        } else {
            _mm_storeu_ps(d + i + 0, _mm_sub_ps(va0, vb0));
            _mm_storeu_ps(d + i + 4, _mm_sub_ps(va1, vb1));
            _mm_storeu_ps(d + i + 8, _mm_sub_ps(va2, vb2));
            _mm_storeu_ps(d + i + 12, _mm_sub_ps(va3, vb3));
            _mm_storeu_ps(d + i + 16, _mm_sub_ps(va4, vb4));
            _mm_storeu_ps(d + i + 20, _mm_sub_ps(va5, vb5));
            _mm_storeu_ps(d + i + 24, _mm_sub_ps(va6, vb6));
            _mm_storeu_ps(d + i + 28, _mm_sub_ps(va7, vb7));
        }
        i += 32;
    }
    // Tail 16 floats
    if (i + 16 <= size) {
        const __m128 va0 = _mm_loadu_ps(d + i);
        const __m128 vb0 = _mm_loadu_ps(o + i);
        const __m128 va1 = _mm_loadu_ps(d + i + 4);
        const __m128 vb1 = _mm_loadu_ps(o + i + 4);
        const __m128 va2 = _mm_loadu_ps(d + i + 8);
        const __m128 vb2 = _mm_loadu_ps(o + i + 8);
        const __m128 va3 = _mm_loadu_ps(d + i + 12);
        const __m128 vb3 = _mm_loadu_ps(o + i + 12);
        _mm_storeu_ps(d + i, _mm_sub_ps(va0, vb0));
        _mm_storeu_ps(d + i + 4, _mm_sub_ps(va1, vb1));
        _mm_storeu_ps(d + i + 8, _mm_sub_ps(va2, vb2));
        _mm_storeu_ps(d + i + 12, _mm_sub_ps(va3, vb3));
        i += 16;
    }
    // Tail 8 floats
    if (i + 8 <= size) {
        const __m128 va0 = _mm_loadu_ps(d + i);
        const __m128 vb0 = _mm_loadu_ps(o + i);
        const __m128 va1 = _mm_loadu_ps(d + i + 4);
        const __m128 vb1 = _mm_loadu_ps(o + i + 4);
        _mm_storeu_ps(d + i, _mm_sub_ps(va0, vb0));
        _mm_storeu_ps(d + i + 4, _mm_sub_ps(va1, vb1));
        i += 8;
    }
    // Tail 4 floats
    if (i + 4 <= size) {
        const __m128 va = _mm_loadu_ps(d + i);
        const __m128 vb = _mm_loadu_ps(o + i);
        _mm_storeu_ps(d + i, _mm_sub_ps(va, vb));
        i += 4;
    }
    // Scalar tail
    while (i < size) {
        d[i] -= o[i];
        ++i;
    }
}

// =======================
// INT8 kernels
// =======================

void SSE41Backend::operation_int8(const TensorBackend& other, OperationType op) {
    const int8_t* o = other.data_int8();
    int8_t* d = data_int8();

    const std::size_t vec_end = (size_ / 16) * 16; // 16 int8 per XMM

    if (op == OperationType::ADD) {
        for (std::size_t i = 0; i < vec_end; i += 16) {
            const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(d + i));
            const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o + i));
            const __m128i r = _mm_adds_epi8(a, b);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(d + i), r);
        }
        for (std::size_t i = vec_end; i < size_; ++i) {
            int v = static_cast<int>(d[i]) + static_cast<int>(o[i]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            d[i] = static_cast<int8_t>(v);
        }
        return;
    }

    if (op == OperationType::MUL) {
        for (std::size_t i = 0; i < vec_end; i += 16) {
            const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(d + i));
            const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o + i));
            const __m128i a_lo = _mm_cvtepi8_epi16(a);
            const __m128i b_lo = _mm_cvtepi8_epi16(b);
            const __m128i a_hi = _mm_cvtepi8_epi16(_mm_srli_si128(a, 8));
            const __m128i b_hi = _mm_cvtepi8_epi16(_mm_srli_si128(b, 8));
            const __m128i prod_lo = _mm_mullo_epi16(a_lo, b_lo);
            const __m128i prod_hi = _mm_mullo_epi16(a_hi, b_hi);
            const __m128i r = _mm_packs_epi16(prod_lo, prod_hi);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(d + i), r);
        }
        for (std::size_t i = vec_end; i < size_; ++i) {
            int v = static_cast<int>(d[i]) * static_cast<int>(o[i]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            d[i] = static_cast<int8_t>(v);
        }
        return;
    }

    // SUB (default) with saturation
    for (std::size_t i = 0; i < vec_end; i += 16) {
        const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(d + i));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o + i));
        const __m128i r = _mm_subs_epi8(a, b);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + i), r);
    }
    for (std::size_t i = vec_end; i < size_; ++i) {
        int v = static_cast<int>(d[i]) - static_cast<int>(o[i]);
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        d[i] = static_cast<int8_t>(v);
    }
}

// =======================
// Public dispatch
// =======================

} // namespace labneura

#endif // __SSE4_1__