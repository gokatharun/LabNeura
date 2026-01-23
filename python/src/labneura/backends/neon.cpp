#if defined(__ARM_NEON)

#include "labneura/backends/neon.h"
#include "labneura/types.hpp"
#include <arm_neon.h>
#include <cstdlib>
#include "labneura/utils/alignment.hpp"

namespace labneura {

// =======================
// Aligned allocation helpers (use central utility)
// =======================

// =======================
// Constructor
// =======================

NEONBackend::NEONBackend(std::size_t size, QuantizationMode mode)
    : size_(size), quantization_mode_(mode) {

    aligned_size_ = ((size + 7) / 8) * 8;  // align to 8 elements for FP16/FP32 safety

    if (mode == QuantizationMode::FP32) {
        float* p = labneura::util::allocate_aligned<float>(aligned_size_, labneura::util::NEON_ALIGN);
        data_fp32_ = std::unique_ptr<float, void(*)(float*)>(p, labneura::util::free_aligned_float);
        std::fill(data_fp32_.get() + size_, data_fp32_.get() + aligned_size_, 0.0f);
    } else if (mode == QuantizationMode::FP16) {
        int16_t* p = labneura::util::allocate_aligned<int16_t>(aligned_size_, labneura::util::NEON_ALIGN);
        auto fp16_deleter = [](int16_t* ptr){ labneura::util::aligned_free(ptr); };
        data_fp16_ = std::unique_ptr<int16_t, void(*)(int16_t*)>(p, fp16_deleter);
        std::fill(data_fp16_.get() + size_, data_fp16_.get() + aligned_size_, static_cast<int16_t>(0));
    } else if (mode == QuantizationMode::INT16) {
        int16_t* p = labneura::util::allocate_aligned<int16_t>(aligned_size_, labneura::util::NEON_ALIGN);
        auto int16_deleter = [](int16_t* ptr){ labneura::util::aligned_free(ptr); };
        data_int16_ = std::unique_ptr<int16_t, void(*)(int16_t*)>(p, int16_deleter);
        std::fill(data_int16_.get() + size_, data_int16_.get() + aligned_size_, static_cast<int16_t>(0));
    } else {
        int8_t* p = labneura::util::allocate_aligned<int8_t>(aligned_size_, labneura::util::NEON_ALIGN);
        data_int8_ = std::unique_ptr<int8_t, void(*)(int8_t*)>(p, labneura::util::free_aligned_int8);
        std::fill(data_int8_.get() + size_, data_int8_.get() + aligned_size_, 0);
    }
}

// =======================
// Metadata
// =======================

std::size_t NEONBackend::size() const {
    return size_;
}

QuantizationMode NEONBackend::quantization_mode() const {
    return quantization_mode_;
}

// =======================
// Data access
// =======================

float* NEONBackend::data_fp32() {
    if (quantization_mode_ != QuantizationMode::FP32) {
        throw std::runtime_error("Tensor is not in FP32 mode");
    }
    return data_fp32_.get();
}

const float* NEONBackend::data_fp32() const {
    if (quantization_mode_ != QuantizationMode::FP32) {
        throw std::runtime_error("Tensor is not in FP32 mode");
    }
    return data_fp32_.get();
}

int8_t* NEONBackend::data_int8() {
    if (quantization_mode_ != QuantizationMode::INT8) {
        throw std::runtime_error("Tensor is not in INT8 mode");
    }
    return data_int8_.get();
}

const int8_t* NEONBackend::data_int8() const {
    if (quantization_mode_ != QuantizationMode::INT8) {
        throw std::runtime_error("Tensor is not in INT8 mode");
    }
    return data_int8_.get();
}

int16_t* NEONBackend::data_fp16() {
    if (quantization_mode_ != QuantizationMode::FP16) {
        throw std::runtime_error("Tensor is not in FP16 mode");
    }
    return data_fp16_.get();
}

const int16_t* NEONBackend::data_fp16() const {
    if (quantization_mode_ != QuantizationMode::FP16) {
        throw std::runtime_error("Tensor is not in FP16 mode");
    }
    return data_fp16_.get();
}

int16_t* NEONBackend::data_int16() {
    if (quantization_mode_ != QuantizationMode::INT16) {
        throw std::runtime_error("Tensor is not in INT16 mode");
    }
    return data_int16_.get();
}

const int16_t* NEONBackend::data_int16() const {
    if (quantization_mode_ != QuantizationMode::INT16) {
        throw std::runtime_error("Tensor is not in INT16 mode");
    }
    return data_int16_.get();
}

// =======================
// Clone
// =======================

std::unique_ptr<TensorBackend> NEONBackend::clone() const {
    auto backend = std::make_unique<NEONBackend>(size_, quantization_mode_);
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

void NEONBackend::operation_fp32(const TensorBackend& other, OperationType op_type) {
    const float* other_data = other.data_fp32();
    float* this_data = data_fp32();

    // Prefetch distance for ARM NEON: 256 bytes (64 floats) ahead
    // M1 has larger caches but benefits from prefetch at 100K-1M sizes
    constexpr std::size_t PREFETCH_DISTANCE = 64;  // 64 floats = 256 bytes

    // Hierarchical processing: 32 → 16 → 8 → 4 → scalar
    const std::size_t len32 = (size_ / 32) * 32;
    const std::size_t len16 = len32 + ((size_ - len32) / 16) * 16;
    const std::size_t len8  = len16 + ((size_ - len16) / 8) * 8;
    const std::size_t len4  = len8  + ((size_ - len8) / 4) * 4;

    if (op_type == OperationType::ADD) {
        // Main loop: 32 floats (8 registers × 4 floats)
        for (std::size_t i = 0; i < len32; i += 32) {
            // Prefetch next iteration's data (ARM intrinsic)
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);  // write, temporal
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1); // read, temporal
            }
            const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
            const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            const float32x4_t va2 = vld1q_f32(&this_data[i + 8]);
            const float32x4_t va3 = vld1q_f32(&this_data[i + 12]);
            const float32x4_t va4 = vld1q_f32(&this_data[i + 16]);
            const float32x4_t va5 = vld1q_f32(&this_data[i + 20]);
            const float32x4_t va6 = vld1q_f32(&this_data[i + 24]);
            const float32x4_t va7 = vld1q_f32(&this_data[i + 28]);
            
            const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
            const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            const float32x4_t vb2 = vld1q_f32(&other_data[i + 8]);
            const float32x4_t vb3 = vld1q_f32(&other_data[i + 12]);
            const float32x4_t vb4 = vld1q_f32(&other_data[i + 16]);
            const float32x4_t vb5 = vld1q_f32(&other_data[i + 20]);
            const float32x4_t vb6 = vld1q_f32(&other_data[i + 24]);
            const float32x4_t vb7 = vld1q_f32(&other_data[i + 28]);
            
            vst1q_f32(&this_data[i + 0],  vaddq_f32(va0, vb0));
            vst1q_f32(&this_data[i + 4],  vaddq_f32(va1, vb1));
            vst1q_f32(&this_data[i + 8],  vaddq_f32(va2, vb2));
            vst1q_f32(&this_data[i + 12], vaddq_f32(va3, vb3));
            vst1q_f32(&this_data[i + 16], vaddq_f32(va4, vb4));
            vst1q_f32(&this_data[i + 20], vaddq_f32(va5, vb5));
            vst1q_f32(&this_data[i + 24], vaddq_f32(va6, vb6));
            vst1q_f32(&this_data[i + 28], vaddq_f32(va7, vb7));
        }
        // Tail: 16 floats (4 registers × 4 floats)
        if (len16 > len32) {
            const std::size_t i = len32;
            const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
            const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            const float32x4_t va2 = vld1q_f32(&this_data[i + 8]);
            const float32x4_t va3 = vld1q_f32(&this_data[i + 12]);
            const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
            const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            const float32x4_t vb2 = vld1q_f32(&other_data[i + 8]);
            const float32x4_t vb3 = vld1q_f32(&other_data[i + 12]);
            vst1q_f32(&this_data[i + 0],  vaddq_f32(va0, vb0));
            vst1q_f32(&this_data[i + 4],  vaddq_f32(va1, vb1));
            vst1q_f32(&this_data[i + 8],  vaddq_f32(va2, vb2));
            vst1q_f32(&this_data[i + 12], vaddq_f32(va3, vb3));
        }
        // Tail: 8 floats (2 registers × 4 floats)
        if (len8 > len16) {
            const std::size_t i = len16;
            const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
            const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
            const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            vst1q_f32(&this_data[i + 0], vaddq_f32(va0, vb0));
            vst1q_f32(&this_data[i + 4], vaddq_f32(va1, vb1));
        }
        // Tail: 4 floats (1 register)
        if (len4 > len8) {
            const std::size_t i = len8;
            const float32x4_t va = vld1q_f32(&this_data[i]);
            const float32x4_t vb = vld1q_f32(&other_data[i]);
            vst1q_f32(&this_data[i], vaddq_f32(va, vb));
        }
        // Scalar tail: remaining 1-3 elements
        for (std::size_t i = len4; i < size_; ++i) {
            this_data[i] += other_data[i];
        }
        return;
    }

    if (op_type == OperationType::MUL) {
        // Main loop: 32 floats
        for (std::size_t i = 0; i < len32; i += 32) {
            // Prefetch for multiply operation
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
            }
            const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
            const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            const float32x4_t va2 = vld1q_f32(&this_data[i + 8]);
            const float32x4_t va3 = vld1q_f32(&this_data[i + 12]);
            const float32x4_t va4 = vld1q_f32(&this_data[i + 16]);
            const float32x4_t va5 = vld1q_f32(&this_data[i + 20]);
            const float32x4_t va6 = vld1q_f32(&this_data[i + 24]);
            const float32x4_t va7 = vld1q_f32(&this_data[i + 28]);
            
            const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
            const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            const float32x4_t vb2 = vld1q_f32(&other_data[i + 8]);
            const float32x4_t vb3 = vld1q_f32(&other_data[i + 12]);
            const float32x4_t vb4 = vld1q_f32(&other_data[i + 16]);
            const float32x4_t vb5 = vld1q_f32(&other_data[i + 20]);
            const float32x4_t vb6 = vld1q_f32(&other_data[i + 24]);
            const float32x4_t vb7 = vld1q_f32(&other_data[i + 28]);
            
            vst1q_f32(&this_data[i + 0],  vmulq_f32(va0, vb0));
            vst1q_f32(&this_data[i + 4],  vmulq_f32(va1, vb1));
            vst1q_f32(&this_data[i + 8],  vmulq_f32(va2, vb2));
            vst1q_f32(&this_data[i + 12], vmulq_f32(va3, vb3));
            vst1q_f32(&this_data[i + 16], vmulq_f32(va4, vb4));
            vst1q_f32(&this_data[i + 20], vmulq_f32(va5, vb5));
            vst1q_f32(&this_data[i + 24], vmulq_f32(va6, vb6));
            vst1q_f32(&this_data[i + 28], vmulq_f32(va7, vb7));
        }
        // Tail: 16 floats
        if (len16 > len32) {
            const std::size_t i = len32;
            const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
            const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            const float32x4_t va2 = vld1q_f32(&this_data[i + 8]);
            const float32x4_t va3 = vld1q_f32(&this_data[i + 12]);
            const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
            const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            const float32x4_t vb2 = vld1q_f32(&other_data[i + 8]);
            const float32x4_t vb3 = vld1q_f32(&other_data[i + 12]);
            vst1q_f32(&this_data[i + 0],  vmulq_f32(va0, vb0));
            vst1q_f32(&this_data[i + 4],  vmulq_f32(va1, vb1));
            vst1q_f32(&this_data[i + 8],  vmulq_f32(va2, vb2));
            vst1q_f32(&this_data[i + 12], vmulq_f32(va3, vb3));
        }
        // Tail: 8 floats
        if (len8 > len16) {
            const std::size_t i = len16;
            const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
            const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
            const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            vst1q_f32(&this_data[i + 0], vmulq_f32(va0, vb0));
            vst1q_f32(&this_data[i + 4], vmulq_f32(va1, vb1));
        }
        // Tail: 4 floats
        if (len4 > len8) {
            const std::size_t i = len8;
            const float32x4_t va = vld1q_f32(&this_data[i]);
            const float32x4_t vb = vld1q_f32(&other_data[i]);
            vst1q_f32(&this_data[i], vmulq_f32(va, vb));
        }
        // Scalar tail
        for (std::size_t i = len4; i < size_; ++i) {
            this_data[i] *= other_data[i];
        }
        return;
    }

    // DIV path for FP32
    if (op_type == OperationType::DIV) {
        // Main loop: 32 floats (8 registers × 4 floats)
        for (std::size_t i = 0; i < len32; i += 32) {
            // Prefetch for division operation
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
            }
            
            const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
            const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            const float32x4_t va2 = vld1q_f32(&this_data[i + 8]);
            const float32x4_t va3 = vld1q_f32(&this_data[i + 12]);
            const float32x4_t va4 = vld1q_f32(&this_data[i + 16]);
            const float32x4_t va5 = vld1q_f32(&this_data[i + 20]);
            const float32x4_t va6 = vld1q_f32(&this_data[i + 24]);
            const float32x4_t va7 = vld1q_f32(&this_data[i + 28]);
            
            const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
            const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            const float32x4_t vb2 = vld1q_f32(&other_data[i + 8]);
            const float32x4_t vb3 = vld1q_f32(&other_data[i + 12]);
            const float32x4_t vb4 = vld1q_f32(&other_data[i + 16]);
            const float32x4_t vb5 = vld1q_f32(&other_data[i + 20]);
            const float32x4_t vb6 = vld1q_f32(&other_data[i + 24]);
            const float32x4_t vb7 = vld1q_f32(&other_data[i + 28]);
            
            // NEON division using vdivq_f32 (ARMv8 and later)
            #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__)
                vst1q_f32(&this_data[i + 0],  vdivq_f32(va0, vb0));
                vst1q_f32(&this_data[i + 4],  vdivq_f32(va1, vb1));
                vst1q_f32(&this_data[i + 8],  vdivq_f32(va2, vb2));
                vst1q_f32(&this_data[i + 12], vdivq_f32(va3, vb3));
                vst1q_f32(&this_data[i + 16], vdivq_f32(va4, vb4));
                vst1q_f32(&this_data[i + 20], vdivq_f32(va5, vb5));
                vst1q_f32(&this_data[i + 24], vdivq_f32(va6, vb6));
                vst1q_f32(&this_data[i + 28], vdivq_f32(va7, vb7));
            #else
                // Fallback using reciprocal estimate + Newton-Raphson (ARMv7 NEON)
                float32x4_t r0 = vrecpeq_f32(vb0);
                float32x4_t r1 = vrecpeq_f32(vb1);
                float32x4_t r2 = vrecpeq_f32(vb2);
                float32x4_t r3 = vrecpeq_f32(vb3);
                float32x4_t r4 = vrecpeq_f32(vb4);
                float32x4_t r5 = vrecpeq_f32(vb5);
                float32x4_t r6 = vrecpeq_f32(vb6);
                float32x4_t r7 = vrecpeq_f32(vb7);
                
                // Newton-Raphson iteration: r = r * (2 - b * r)
                r0 = vmulq_f32(r0, vrecpsq_f32(vb0, r0));
                r1 = vmulq_f32(r1, vrecpsq_f32(vb1, r1));
                r2 = vmulq_f32(r2, vrecpsq_f32(vb2, r2));
                r3 = vmulq_f32(r3, vrecpsq_f32(vb3, r3));
                r4 = vmulq_f32(r4, vrecpsq_f32(vb4, r4));
                r5 = vmulq_f32(r5, vrecpsq_f32(vb5, r5));
                r6 = vmulq_f32(r6, vrecpsq_f32(vb6, r6));
                r7 = vmulq_f32(r7, vrecpsq_f32(vb7, r7));
                
                // Second iteration for better precision
                r0 = vmulq_f32(r0, vrecpsq_f32(vb0, r0));
                r1 = vmulq_f32(r1, vrecpsq_f32(vb1, r1));
                r2 = vmulq_f32(r2, vrecpsq_f32(vb2, r2));
                r3 = vmulq_f32(r3, vrecpsq_f32(vb3, r3));
                r4 = vmulq_f32(r4, vrecpsq_f32(vb4, r4));
                r5 = vmulq_f32(r5, vrecpsq_f32(vb5, r5));
                r6 = vmulq_f32(r6, vrecpsq_f32(vb6, r6));
                r7 = vmulq_f32(r7, vrecpsq_f32(vb7, r7));
                
                vst1q_f32(&this_data[i + 0],  vmulq_f32(va0, r0));
                vst1q_f32(&this_data[i + 4],  vmulq_f32(va1, r1));
                vst1q_f32(&this_data[i + 8],  vmulq_f32(va2, r2));
                vst1q_f32(&this_data[i + 12], vmulq_f32(va3, r3));
                vst1q_f32(&this_data[i + 16], vmulq_f32(va4, r4));
                vst1q_f32(&this_data[i + 20], vmulq_f32(va5, r5));
                vst1q_f32(&this_data[i + 24], vmulq_f32(va6, r6));
                vst1q_f32(&this_data[i + 28], vmulq_f32(va7, r7));
            #endif
        }
        
        // Tail: 16 floats (4 registers × 4 floats)
        if (len16 > len32) {
            const std::size_t i = len32;
            const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
            const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            const float32x4_t va2 = vld1q_f32(&this_data[i + 8]);
            const float32x4_t va3 = vld1q_f32(&this_data[i + 12]);
            const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
            const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            const float32x4_t vb2 = vld1q_f32(&other_data[i + 8]);
            const float32x4_t vb3 = vld1q_f32(&other_data[i + 12]);
            
            #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__)
                vst1q_f32(&this_data[i + 0],  vdivq_f32(va0, vb0));
                vst1q_f32(&this_data[i + 4],  vdivq_f32(va1, vb1));
                vst1q_f32(&this_data[i + 8],  vdivq_f32(va2, vb2));
                vst1q_f32(&this_data[i + 12], vdivq_f32(va3, vb3));
            #else
                float32x4_t r0 = vrecpeq_f32(vb0);
                float32x4_t r1 = vrecpeq_f32(vb1);
                float32x4_t r2 = vrecpeq_f32(vb2);
                float32x4_t r3 = vrecpeq_f32(vb3);
                r0 = vmulq_f32(r0, vrecpsq_f32(vb0, r0));
                r1 = vmulq_f32(r1, vrecpsq_f32(vb1, r1));
                r2 = vmulq_f32(r2, vrecpsq_f32(vb2, r2));
                r3 = vmulq_f32(r3, vrecpsq_f32(vb3, r3));
                r0 = vmulq_f32(r0, vrecpsq_f32(vb0, r0));
                r1 = vmulq_f32(r1, vrecpsq_f32(vb1, r1));
                r2 = vmulq_f32(r2, vrecpsq_f32(vb2, r2));
                r3 = vmulq_f32(r3, vrecpsq_f32(vb3, r3));
                vst1q_f32(&this_data[i + 0],  vmulq_f32(va0, r0));
                vst1q_f32(&this_data[i + 4],  vmulq_f32(va1, r1));
                vst1q_f32(&this_data[i + 8],  vmulq_f32(va2, r2));
                vst1q_f32(&this_data[i + 12], vmulq_f32(va3, r3));
            #endif
        }
        
        // Tail: 8 floats (2 registers × 4 floats)
        if (len8 > len16) {
            const std::size_t i = len16;
            const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
            const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
            const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            
            #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__)
                vst1q_f32(&this_data[i + 0], vdivq_f32(va0, vb0));
                vst1q_f32(&this_data[i + 4], vdivq_f32(va1, vb1));
            #else
                float32x4_t r0 = vrecpeq_f32(vb0);
                float32x4_t r1 = vrecpeq_f32(vb1);
                r0 = vmulq_f32(r0, vrecpsq_f32(vb0, r0));
                r1 = vmulq_f32(r1, vrecpsq_f32(vb1, r1));
                r0 = vmulq_f32(r0, vrecpsq_f32(vb0, r0));
                r1 = vmulq_f32(r1, vrecpsq_f32(vb1, r1));
                vst1q_f32(&this_data[i + 0], vmulq_f32(va0, r0));
                vst1q_f32(&this_data[i + 4], vmulq_f32(va1, r1));
            #endif
        }
        
        // Tail: 4 floats (1 register)
        if (len4 > len8) {
            const std::size_t i = len8;
            const float32x4_t va = vld1q_f32(&this_data[i]);
            const float32x4_t vb = vld1q_f32(&other_data[i]);
            
            #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__)
                vst1q_f32(&this_data[i], vdivq_f32(va, vb));
            #else
                float32x4_t r = vrecpeq_f32(vb);
                r = vmulq_f32(r, vrecpsq_f32(vb, r));
                r = vmulq_f32(r, vrecpsq_f32(vb, r));
                vst1q_f32(&this_data[i], vmulq_f32(va, r));
            #endif
        }
        
        // Scalar tail: remaining 1-3 elements
        for (std::size_t i = len4; i < size_; ++i) {
            this_data[i] /= other_data[i];
        }
        return;
    }

    // SUB path (default)
    for (std::size_t i = 0; i < len32; i += 32) {
        // Prefetch for subtract operation
        if (i + PREFETCH_DISTANCE < size_) {
            __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
            __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
        }
        
        const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
        const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
        const float32x4_t va2 = vld1q_f32(&this_data[i + 8]);
        const float32x4_t va3 = vld1q_f32(&this_data[i + 12]);
        const float32x4_t va4 = vld1q_f32(&this_data[i + 16]);
        const float32x4_t va5 = vld1q_f32(&this_data[i + 20]);
        const float32x4_t va6 = vld1q_f32(&this_data[i + 24]);
        const float32x4_t va7 = vld1q_f32(&this_data[i + 28]);
        
        const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
        const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
        const float32x4_t vb2 = vld1q_f32(&other_data[i + 8]);
        const float32x4_t vb3 = vld1q_f32(&other_data[i + 12]);
        const float32x4_t vb4 = vld1q_f32(&other_data[i + 16]);
        const float32x4_t vb5 = vld1q_f32(&other_data[i + 20]);
        const float32x4_t vb6 = vld1q_f32(&other_data[i + 24]);
        const float32x4_t vb7 = vld1q_f32(&other_data[i + 28]);
        
        vst1q_f32(&this_data[i + 0],  vsubq_f32(va0, vb0));
        vst1q_f32(&this_data[i + 4],  vsubq_f32(va1, vb1));
        vst1q_f32(&this_data[i + 8],  vsubq_f32(va2, vb2));
        vst1q_f32(&this_data[i + 12], vsubq_f32(va3, vb3));
        vst1q_f32(&this_data[i + 16], vsubq_f32(va4, vb4));
        vst1q_f32(&this_data[i + 20], vsubq_f32(va5, vb5));
        vst1q_f32(&this_data[i + 24], vsubq_f32(va6, vb6));
        vst1q_f32(&this_data[i + 28], vsubq_f32(va7, vb7));
    }
    // Tail: 16 floats
    if (len16 > len32) {
        const std::size_t i = len32;
        const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
        const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
        const float32x4_t va2 = vld1q_f32(&this_data[i + 8]);
        const float32x4_t va3 = vld1q_f32(&this_data[i + 12]);
        const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
        const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
        const float32x4_t vb2 = vld1q_f32(&other_data[i + 8]);
        const float32x4_t vb3 = vld1q_f32(&other_data[i + 12]);
        vst1q_f32(&this_data[i + 0],  vsubq_f32(va0, vb0));
        vst1q_f32(&this_data[i + 4],  vsubq_f32(va1, vb1));
        vst1q_f32(&this_data[i + 8],  vsubq_f32(va2, vb2));
        vst1q_f32(&this_data[i + 12], vsubq_f32(va3, vb3));
    }
    // Tail: 8 floats
    if (len8 > len16) {
        const std::size_t i = len16;
        const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
        const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
        const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
        const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
        vst1q_f32(&this_data[i + 0], vsubq_f32(va0, vb0));
        vst1q_f32(&this_data[i + 4], vsubq_f32(va1, vb1));
    }
    // Tail: 4 floats
    if (len4 > len8) {
        const std::size_t i = len8;
        const float32x4_t va = vld1q_f32(&this_data[i]);
        const float32x4_t vb = vld1q_f32(&other_data[i]);
        vst1q_f32(&this_data[i], vsubq_f32(va, vb));
    }
    // Scalar tail
    for (std::size_t i = len4; i < size_; ++i) {
        this_data[i] -= other_data[i];
    }
}

// =======================
// FP16 kernels
// =======================

void NEONBackend::operation_fp16(const TensorBackend& other, OperationType op_type) {
    const float16_t* other_data = reinterpret_cast<const float16_t*>(other.data_fp16());
    float16_t* this_data = reinterpret_cast<float16_t*>(data_fp16());

    // Prefetch distance: 128 float16s (~256 bytes)
    constexpr std::size_t PREFETCH_DISTANCE = 128;

    // Hierarchical processing: 64 → 32 → 16 → 8 → 4 → scalar
    const std::size_t len64 = (size_ / 64) * 64;
    const std::size_t len32 = len64 + ((size_ - len64) / 32) * 32;
    const std::size_t len16 = len32 + ((size_ - len32) / 16) * 16;
    const std::size_t len8  = len16 + ((size_ - len16) / 8) * 8;
    const std::size_t len4  = len8  + ((size_ - len8) / 4) * 4;

    if (op_type == OperationType::ADD) {
        // Main loop: 64 values (8 registers × 8 float16s)
        for (std::size_t i = 0; i < len64; i += 64) {
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
            }
            const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
            const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
            const float16x8_t va2 = vld1q_f16(&this_data[i + 16]);
            const float16x8_t va3 = vld1q_f16(&this_data[i + 24]);
            const float16x8_t va4 = vld1q_f16(&this_data[i + 32]);
            const float16x8_t va5 = vld1q_f16(&this_data[i + 40]);
            const float16x8_t va6 = vld1q_f16(&this_data[i + 48]);
            const float16x8_t va7 = vld1q_f16(&this_data[i + 56]);

            const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
            const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
            const float16x8_t vb2 = vld1q_f16(&other_data[i + 16]);
            const float16x8_t vb3 = vld1q_f16(&other_data[i + 24]);
            const float16x8_t vb4 = vld1q_f16(&other_data[i + 32]);
            const float16x8_t vb5 = vld1q_f16(&other_data[i + 40]);
            const float16x8_t vb6 = vld1q_f16(&other_data[i + 48]);
            const float16x8_t vb7 = vld1q_f16(&other_data[i + 56]);

            vst1q_f16(&this_data[i + 0],  vaddq_f16(va0, vb0));
            vst1q_f16(&this_data[i + 8],  vaddq_f16(va1, vb1));
            vst1q_f16(&this_data[i + 16], vaddq_f16(va2, vb2));
            vst1q_f16(&this_data[i + 24], vaddq_f16(va3, vb3));
            vst1q_f16(&this_data[i + 32], vaddq_f16(va4, vb4));
            vst1q_f16(&this_data[i + 40], vaddq_f16(va5, vb5));
            vst1q_f16(&this_data[i + 48], vaddq_f16(va6, vb6));
            vst1q_f16(&this_data[i + 56], vaddq_f16(va7, vb7));
        }
        // Tail: 32 values (4 registers × 8 float16s)
        if (len32 > len64) {
            const std::size_t i = len64;
            const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
            const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
            const float16x8_t va2 = vld1q_f16(&this_data[i + 16]);
            const float16x8_t va3 = vld1q_f16(&this_data[i + 24]);
            const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
            const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
            const float16x8_t vb2 = vld1q_f16(&other_data[i + 16]);
            const float16x8_t vb3 = vld1q_f16(&other_data[i + 24]);
            vst1q_f16(&this_data[i + 0],  vaddq_f16(va0, vb0));
            vst1q_f16(&this_data[i + 8],  vaddq_f16(va1, vb1));
            vst1q_f16(&this_data[i + 16], vaddq_f16(va2, vb2));
            vst1q_f16(&this_data[i + 24], vaddq_f16(va3, vb3));
        }
        // Tail: 16 values (2 registers × 8 float16s)
        if (len16 > len32) {
            const std::size_t i = len32;
            const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
            const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
            const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
            const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
            vst1q_f16(&this_data[i + 0], vaddq_f16(va0, vb0));
            vst1q_f16(&this_data[i + 8], vaddq_f16(va1, vb1));
        }
        // Tail: 8 values (1 register)
        if (len8 > len16) {
            const std::size_t i = len16;
            const float16x8_t va = vld1q_f16(&this_data[i]);
            const float16x8_t vb = vld1q_f16(&other_data[i]);
            vst1q_f16(&this_data[i], vaddq_f16(va, vb));
        }
        // Tail: 4 values (half register)
        if (len4 > len8) {
            const std::size_t i = len8;
            const float16x4_t va = vld1_f16(&this_data[i]);
            const float16x4_t vb = vld1_f16(&other_data[i]);
            vst1_f16(&this_data[i], vadd_f16(va, vb));
        }
        // Scalar tail
        for (std::size_t i = len4; i < size_; ++i) {
            this_data[i] = this_data[i] + other_data[i];
        }
        return;
    }

    if (op_type == OperationType::MUL) {
        // Main loop: 64 values
        for (std::size_t i = 0; i < len64; i += 64) {
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
            }
            const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
            const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
            const float16x8_t va2 = vld1q_f16(&this_data[i + 16]);
            const float16x8_t va3 = vld1q_f16(&this_data[i + 24]);
            const float16x8_t va4 = vld1q_f16(&this_data[i + 32]);
            const float16x8_t va5 = vld1q_f16(&this_data[i + 40]);
            const float16x8_t va6 = vld1q_f16(&this_data[i + 48]);
            const float16x8_t va7 = vld1q_f16(&this_data[i + 56]);

            const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
            const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
            const float16x8_t vb2 = vld1q_f16(&other_data[i + 16]);
            const float16x8_t vb3 = vld1q_f16(&other_data[i + 24]);
            const float16x8_t vb4 = vld1q_f16(&other_data[i + 32]);
            const float16x8_t vb5 = vld1q_f16(&other_data[i + 40]);
            const float16x8_t vb6 = vld1q_f16(&other_data[i + 48]);
            const float16x8_t vb7 = vld1q_f16(&other_data[i + 56]);

            vst1q_f16(&this_data[i + 0],  vmulq_f16(va0, vb0));
            vst1q_f16(&this_data[i + 8],  vmulq_f16(va1, vb1));
            vst1q_f16(&this_data[i + 16], vmulq_f16(va2, vb2));
            vst1q_f16(&this_data[i + 24], vmulq_f16(va3, vb3));
            vst1q_f16(&this_data[i + 32], vmulq_f16(va4, vb4));
            vst1q_f16(&this_data[i + 40], vmulq_f16(va5, vb5));
            vst1q_f16(&this_data[i + 48], vmulq_f16(va6, vb6));
            vst1q_f16(&this_data[i + 56], vmulq_f16(va7, vb7));
        }
        // Tail: 32 values
        if (len32 > len64) {
            const std::size_t i = len64;
            const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
            const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
            const float16x8_t va2 = vld1q_f16(&this_data[i + 16]);
            const float16x8_t va3 = vld1q_f16(&this_data[i + 24]);
            const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
            const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
            const float16x8_t vb2 = vld1q_f16(&other_data[i + 16]);
            const float16x8_t vb3 = vld1q_f16(&other_data[i + 24]);
            vst1q_f16(&this_data[i + 0],  vmulq_f16(va0, vb0));
            vst1q_f16(&this_data[i + 8],  vmulq_f16(va1, vb1));
            vst1q_f16(&this_data[i + 16], vmulq_f16(va2, vb2));
            vst1q_f16(&this_data[i + 24], vmulq_f16(va3, vb3));
        }
        // Tail: 16 values
        if (len16 > len32) {
            const std::size_t i = len32;
            const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
            const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
            const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
            const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
            vst1q_f16(&this_data[i + 0], vmulq_f16(va0, vb0));
            vst1q_f16(&this_data[i + 8], vmulq_f16(va1, vb1));
        }
        // Tail: 8 values
        if (len8 > len16) {
            const std::size_t i = len16;
            const float16x8_t va = vld1q_f16(&this_data[i]);
            const float16x8_t vb = vld1q_f16(&other_data[i]);
            vst1q_f16(&this_data[i], vmulq_f16(va, vb));
        }
        // Tail: 4 values
        if (len4 > len8) {
            const std::size_t i = len8;
            const float16x4_t va = vld1_f16(&this_data[i]);
            const float16x4_t vb = vld1_f16(&other_data[i]);
            vst1_f16(&this_data[i], vmul_f16(va, vb));
        }
        // Scalar tail
        for (std::size_t i = len4; i < size_; ++i) {
            this_data[i] = this_data[i] * other_data[i];
        }
        return;
    }

    // DIV path for FP16
    if (op_type == OperationType::DIV) {
        // Main loop: 64 values (8 registers × 8 float16s)
        for (std::size_t i = 0; i < len64; i += 64) {
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
            }
            const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
            const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
            const float16x8_t va2 = vld1q_f16(&this_data[i + 16]);
            const float16x8_t va3 = vld1q_f16(&this_data[i + 24]);
            const float16x8_t va4 = vld1q_f16(&this_data[i + 32]);
            const float16x8_t va5 = vld1q_f16(&this_data[i + 40]);
            const float16x8_t va6 = vld1q_f16(&this_data[i + 48]);
            const float16x8_t va7 = vld1q_f16(&this_data[i + 56]);

            const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
            const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
            const float16x8_t vb2 = vld1q_f16(&other_data[i + 16]);
            const float16x8_t vb3 = vld1q_f16(&other_data[i + 24]);
            const float16x8_t vb4 = vld1q_f16(&other_data[i + 32]);
            const float16x8_t vb5 = vld1q_f16(&other_data[i + 40]);
            const float16x8_t vb6 = vld1q_f16(&other_data[i + 48]);
            const float16x8_t vb7 = vld1q_f16(&other_data[i + 56]);

            vst1q_f16(&this_data[i + 0],  vdivq_f16(va0, vb0));
            vst1q_f16(&this_data[i + 8],  vdivq_f16(va1, vb1));
            vst1q_f16(&this_data[i + 16], vdivq_f16(va2, vb2));
            vst1q_f16(&this_data[i + 24], vdivq_f16(va3, vb3));
            vst1q_f16(&this_data[i + 32], vdivq_f16(va4, vb4));
            vst1q_f16(&this_data[i + 40], vdivq_f16(va5, vb5));
            vst1q_f16(&this_data[i + 48], vdivq_f16(va6, vb6));
            vst1q_f16(&this_data[i + 56], vdivq_f16(va7, vb7));
        }
        // Tail: 32 values
        if (len32 > len64) {
            const std::size_t i = len64;
            const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
            const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
            const float16x8_t va2 = vld1q_f16(&this_data[i + 16]);
            const float16x8_t va3 = vld1q_f16(&this_data[i + 24]);
            const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
            const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
            const float16x8_t vb2 = vld1q_f16(&other_data[i + 16]);
            const float16x8_t vb3 = vld1q_f16(&other_data[i + 24]);
            vst1q_f16(&this_data[i + 0],  vdivq_f16(va0, vb0));
            vst1q_f16(&this_data[i + 8],  vdivq_f16(va1, vb1));
            vst1q_f16(&this_data[i + 16], vdivq_f16(va2, vb2));
            vst1q_f16(&this_data[i + 24], vdivq_f16(va3, vb3));
        }
        // Tail: 16 values
        if (len16 > len32) {
            const std::size_t i = len32;
            const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
            const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
            const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
            const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
            vst1q_f16(&this_data[i + 0], vdivq_f16(va0, vb0));
            vst1q_f16(&this_data[i + 8], vdivq_f16(va1, vb1));
        }
        // Tail: 8 values
        if (len8 > len16) {
            const std::size_t i = len16;
            const float16x8_t va = vld1q_f16(&this_data[i]);
            const float16x8_t vb = vld1q_f16(&other_data[i]);
            vst1q_f16(&this_data[i], vdivq_f16(va, vb));
        }
        // Tail: 4 values
        if (len4 > len8) {
            const std::size_t i = len8;
            const float16x4_t va = vld1_f16(&this_data[i]);
            const float16x4_t vb = vld1_f16(&other_data[i]);
            vst1_f16(&this_data[i], vdiv_f16(va, vb));
        }
        // Scalar tail
        for (std::size_t i = len4; i < size_; ++i) {
            // Cast int16_t to uint16_t for fp16 operations
            uint16_t a_bits, b_bits;
            std::memcpy(&a_bits, &this_data[i], sizeof(uint16_t));
            std::memcpy(&b_bits, &other_data[i], sizeof(uint16_t));
            float a = dequantize_fp16(a_bits);
            float b = dequantize_fp16(b_bits);
            uint16_t result = quantize_fp16(a / b);
            std::memcpy(&this_data[i], &result, sizeof(uint16_t));
        }
        return;
    }

    // SUB path (default)
    for (std::size_t i = 0; i < len64; i += 64) {
        if (i + PREFETCH_DISTANCE < size_) {
            __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
            __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
        }

        const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
        const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
        const float16x8_t va2 = vld1q_f16(&this_data[i + 16]);
        const float16x8_t va3 = vld1q_f16(&this_data[i + 24]);
        const float16x8_t va4 = vld1q_f16(&this_data[i + 32]);
        const float16x8_t va5 = vld1q_f16(&this_data[i + 40]);
        const float16x8_t va6 = vld1q_f16(&this_data[i + 48]);
        const float16x8_t va7 = vld1q_f16(&this_data[i + 56]);

        const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
        const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
        const float16x8_t vb2 = vld1q_f16(&other_data[i + 16]);
        const float16x8_t vb3 = vld1q_f16(&other_data[i + 24]);
        const float16x8_t vb4 = vld1q_f16(&other_data[i + 32]);
        const float16x8_t vb5 = vld1q_f16(&other_data[i + 40]);
        const float16x8_t vb6 = vld1q_f16(&other_data[i + 48]);
        const float16x8_t vb7 = vld1q_f16(&other_data[i + 56]);

        vst1q_f16(&this_data[i + 0],  vsubq_f16(va0, vb0));
        vst1q_f16(&this_data[i + 8],  vsubq_f16(va1, vb1));
        vst1q_f16(&this_data[i + 16], vsubq_f16(va2, vb2));
        vst1q_f16(&this_data[i + 24], vsubq_f16(va3, vb3));
        vst1q_f16(&this_data[i + 32], vsubq_f16(va4, vb4));
        vst1q_f16(&this_data[i + 40], vsubq_f16(va5, vb5));
        vst1q_f16(&this_data[i + 48], vsubq_f16(va6, vb6));
        vst1q_f16(&this_data[i + 56], vsubq_f16(va7, vb7));
    }
    // Tail: 32 values
    if (len32 > len64) {
        const std::size_t i = len64;
        const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
        const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
        const float16x8_t va2 = vld1q_f16(&this_data[i + 16]);
        const float16x8_t va3 = vld1q_f16(&this_data[i + 24]);
        const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
        const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
        const float16x8_t vb2 = vld1q_f16(&other_data[i + 16]);
        const float16x8_t vb3 = vld1q_f16(&other_data[i + 24]);
        vst1q_f16(&this_data[i + 0],  vsubq_f16(va0, vb0));
        vst1q_f16(&this_data[i + 8],  vsubq_f16(va1, vb1));
        vst1q_f16(&this_data[i + 16], vsubq_f16(va2, vb2));
        vst1q_f16(&this_data[i + 24], vsubq_f16(va3, vb3));
    }
    // Tail: 16 values
    if (len16 > len32) {
        const std::size_t i = len32;
        const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
        const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
        const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
        const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
        vst1q_f16(&this_data[i + 0], vsubq_f16(va0, vb0));
        vst1q_f16(&this_data[i + 8], vsubq_f16(va1, vb1));
    }
    // Tail: 8 values
    if (len8 > len16) {
        const std::size_t i = len16;
        const float16x8_t va = vld1q_f16(&this_data[i]);
        const float16x8_t vb = vld1q_f16(&other_data[i]);
        vst1q_f16(&this_data[i], vsubq_f16(va, vb));
    }
    // Tail: 4 values
    if (len4 > len8) {
        const std::size_t i = len8;
        const float16x4_t va = vld1_f16(&this_data[i]);
        const float16x4_t vb = vld1_f16(&other_data[i]);
        vst1_f16(&this_data[i], vsub_f16(va, vb));
    }
    // Scalar tail
    for (std::size_t i = len4; i < size_; ++i) {
        this_data[i] = this_data[i] - other_data[i];
    }
}

// =======================
// INT8 kernels
// =======================

void NEONBackend::operation_int8(const TensorBackend& other, OperationType op_type) {
    const int8_t* other_data = other.data_int8();
    int8_t* this_data = data_int8();

    // Hierarchical processing: 128 → 64 → 32 → 16 → 8 → scalar
    const std::size_t len128 = (size_ / 128) * 128;
    const std::size_t len64  = len128 + ((size_ - len128) / 64) * 64;
    const std::size_t len32  = len64  + ((size_ - len64) / 32) * 32;
    const std::size_t len16  = len32  + ((size_ - len32) / 16) * 16;
    const std::size_t len8   = len16  + ((size_ - len16) / 8) * 8;

    // Prefetch distance for INT8: 256 bytes (256 int8s)
    constexpr std::size_t PREFETCH_DISTANCE = 256;

    if (op_type == OperationType::ADD) {
        // Main loop: 128 int8s (8 registers × 16 int8s)
        for (std::size_t i = 0; i < len128; i += 128) {
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
            }
            const int8x16_t va0 = vld1q_s8(&this_data[i + 0]);
            const int8x16_t va1 = vld1q_s8(&this_data[i + 16]);
            const int8x16_t va2 = vld1q_s8(&this_data[i + 32]);
            const int8x16_t va3 = vld1q_s8(&this_data[i + 48]);
            const int8x16_t va4 = vld1q_s8(&this_data[i + 64]);
            const int8x16_t va5 = vld1q_s8(&this_data[i + 80]);
            const int8x16_t va6 = vld1q_s8(&this_data[i + 96]);
            const int8x16_t va7 = vld1q_s8(&this_data[i + 112]);
            
            const int8x16_t vb0 = vld1q_s8(&other_data[i + 0]);
            const int8x16_t vb1 = vld1q_s8(&other_data[i + 16]);
            const int8x16_t vb2 = vld1q_s8(&other_data[i + 32]);
            const int8x16_t vb3 = vld1q_s8(&other_data[i + 48]);
            const int8x16_t vb4 = vld1q_s8(&other_data[i + 64]);
            const int8x16_t vb5 = vld1q_s8(&other_data[i + 80]);
            const int8x16_t vb6 = vld1q_s8(&other_data[i + 96]);
            const int8x16_t vb7 = vld1q_s8(&other_data[i + 112]);
            
            vst1q_s8(&this_data[i + 0],   vqaddq_s8(va0, vb0));
            vst1q_s8(&this_data[i + 16],  vqaddq_s8(va1, vb1));
            vst1q_s8(&this_data[i + 32],  vqaddq_s8(va2, vb2));
            vst1q_s8(&this_data[i + 48],  vqaddq_s8(va3, vb3));
            vst1q_s8(&this_data[i + 64],  vqaddq_s8(va4, vb4));
            vst1q_s8(&this_data[i + 80],  vqaddq_s8(va5, vb5));
            vst1q_s8(&this_data[i + 96],  vqaddq_s8(va6, vb6));
            vst1q_s8(&this_data[i + 112], vqaddq_s8(va7, vb7));
        }
        // Tail: 64 int8s (4 registers × 16 int8s)
        if (len64 > len128) {
            const std::size_t i = len128;
            const int8x16_t va0 = vld1q_s8(&this_data[i + 0]);
            const int8x16_t va1 = vld1q_s8(&this_data[i + 16]);
            const int8x16_t va2 = vld1q_s8(&this_data[i + 32]);
            const int8x16_t va3 = vld1q_s8(&this_data[i + 48]);
            const int8x16_t vb0 = vld1q_s8(&other_data[i + 0]);
            const int8x16_t vb1 = vld1q_s8(&other_data[i + 16]);
            const int8x16_t vb2 = vld1q_s8(&other_data[i + 32]);
            const int8x16_t vb3 = vld1q_s8(&other_data[i + 48]);
            vst1q_s8(&this_data[i + 0],  vqaddq_s8(va0, vb0));
            vst1q_s8(&this_data[i + 16], vqaddq_s8(va1, vb1));
            vst1q_s8(&this_data[i + 32], vqaddq_s8(va2, vb2));
            vst1q_s8(&this_data[i + 48], vqaddq_s8(va3, vb3));
        }
        // Tail: 32 int8s (2 registers × 16 int8s)
        if (len32 > len64) {
            const std::size_t i = len64;
            const int8x16_t va0 = vld1q_s8(&this_data[i + 0]);
            const int8x16_t va1 = vld1q_s8(&this_data[i + 16]);
            const int8x16_t vb0 = vld1q_s8(&other_data[i + 0]);
            const int8x16_t vb1 = vld1q_s8(&other_data[i + 16]);
            vst1q_s8(&this_data[i + 0],  vqaddq_s8(va0, vb0));
            vst1q_s8(&this_data[i + 16], vqaddq_s8(va1, vb1));
        }
        // Tail: 16 int8s (1 register)
        if (len16 > len32) {
            const std::size_t i = len32;
            const int8x16_t va = vld1q_s8(&this_data[i]);
            const int8x16_t vb = vld1q_s8(&other_data[i]);
            vst1q_s8(&this_data[i], vqaddq_s8(va, vb));
        }
        // Tail: 8 int8s (half register)
        if (len8 > len16) {
            const std::size_t i = len16;
            const int8x8_t va = vld1_s8(&this_data[i]);
            const int8x8_t vb = vld1_s8(&other_data[i]);
            vst1_s8(&this_data[i], vqadd_s8(va, vb));
        }
        // Scalar tail
        for (std::size_t i = len8; i < size_; ++i) {
            int16_t sum = static_cast<int16_t>(this_data[i]) + static_cast<int16_t>(other_data[i]);
            this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(sum))));
        }
        return;
    }

    if (op_type == OperationType::MUL) {
        // Main loop: 128 int8s (8 registers × 16 int8s)
        for (std::size_t i = 0; i < len128; i += 128) {
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
            }
            const int8x16_t va0 = vld1q_s8(&this_data[i + 0]);
            const int8x16_t va1 = vld1q_s8(&this_data[i + 16]);
            const int8x16_t va2 = vld1q_s8(&this_data[i + 32]);
            const int8x16_t va3 = vld1q_s8(&this_data[i + 48]);
            const int8x16_t va4 = vld1q_s8(&this_data[i + 64]);
            const int8x16_t va5 = vld1q_s8(&this_data[i + 80]);
            const int8x16_t va6 = vld1q_s8(&this_data[i + 96]);
            const int8x16_t va7 = vld1q_s8(&this_data[i + 112]);
            
            const int8x16_t vb0 = vld1q_s8(&other_data[i + 0]);
            const int8x16_t vb1 = vld1q_s8(&other_data[i + 16]);
            const int8x16_t vb2 = vld1q_s8(&other_data[i + 32]);
            const int8x16_t vb3 = vld1q_s8(&other_data[i + 48]);
            const int8x16_t vb4 = vld1q_s8(&other_data[i + 64]);
            const int8x16_t vb5 = vld1q_s8(&other_data[i + 80]);
            const int8x16_t vb6 = vld1q_s8(&other_data[i + 96]);
            const int8x16_t vb7 = vld1q_s8(&other_data[i + 112]);
            
            // For multiply, we need to widen to int16, multiply, then narrow back
            // Process each 16-element vector separately
            auto mul_int8x16 = [](const int8x16_t& a, const int8x16_t& b) -> int8x16_t {
                const int16x8_t a_low = vmovl_s8(vget_low_s8(a));
                const int16x8_t a_high = vmovl_s8(vget_high_s8(a));
                const int16x8_t b_low = vmovl_s8(vget_low_s8(b));
                const int16x8_t b_high = vmovl_s8(vget_high_s8(b));
                const int16x8_t prod_low = vmulq_s16(a_low, b_low);
                const int16x8_t prod_high = vmulq_s16(a_high, b_high);
                return vcombine_s8(vqmovn_s16(prod_low), vqmovn_s16(prod_high));
            };
            
            vst1q_s8(&this_data[i + 0],   mul_int8x16(va0, vb0));
            vst1q_s8(&this_data[i + 16],  mul_int8x16(va1, vb1));
            vst1q_s8(&this_data[i + 32],  mul_int8x16(va2, vb2));
            vst1q_s8(&this_data[i + 48],  mul_int8x16(va3, vb3));
            vst1q_s8(&this_data[i + 64],  mul_int8x16(va4, vb4));
            vst1q_s8(&this_data[i + 80],  mul_int8x16(va5, vb5));
            vst1q_s8(&this_data[i + 96],  mul_int8x16(va6, vb6));
            vst1q_s8(&this_data[i + 112], mul_int8x16(va7, vb7));
        }
        // Tail: 64 int8s
        if (len64 > len128) {
            const std::size_t i = len128;
            auto mul_int8x16 = [](const int8x16_t& a, const int8x16_t& b) -> int8x16_t {
                const int16x8_t a_low = vmovl_s8(vget_low_s8(a));
                const int16x8_t a_high = vmovl_s8(vget_high_s8(a));
                const int16x8_t b_low = vmovl_s8(vget_low_s8(b));
                const int16x8_t b_high = vmovl_s8(vget_high_s8(b));
                const int16x8_t prod_low = vmulq_s16(a_low, b_low);
                const int16x8_t prod_high = vmulq_s16(a_high, b_high);
                return vcombine_s8(vqmovn_s16(prod_low), vqmovn_s16(prod_high));
            };
            const int8x16_t va0 = vld1q_s8(&this_data[i + 0]);
            const int8x16_t va1 = vld1q_s8(&this_data[i + 16]);
            const int8x16_t va2 = vld1q_s8(&this_data[i + 32]);
            const int8x16_t va3 = vld1q_s8(&this_data[i + 48]);
            const int8x16_t vb0 = vld1q_s8(&other_data[i + 0]);
            const int8x16_t vb1 = vld1q_s8(&other_data[i + 16]);
            const int8x16_t vb2 = vld1q_s8(&other_data[i + 32]);
            const int8x16_t vb3 = vld1q_s8(&other_data[i + 48]);
            vst1q_s8(&this_data[i + 0],  mul_int8x16(va0, vb0));
            vst1q_s8(&this_data[i + 16], mul_int8x16(va1, vb1));
            vst1q_s8(&this_data[i + 32], mul_int8x16(va2, vb2));
            vst1q_s8(&this_data[i + 48], mul_int8x16(va3, vb3));
        }
        // Tail: 32 int8s
        if (len32 > len64) {
            const std::size_t i = len64;
            auto mul_int8x16 = [](const int8x16_t& a, const int8x16_t& b) -> int8x16_t {
                const int16x8_t a_low = vmovl_s8(vget_low_s8(a));
                const int16x8_t a_high = vmovl_s8(vget_high_s8(a));
                const int16x8_t b_low = vmovl_s8(vget_low_s8(b));
                const int16x8_t b_high = vmovl_s8(vget_high_s8(b));
                const int16x8_t prod_low = vmulq_s16(a_low, b_low);
                const int16x8_t prod_high = vmulq_s16(a_high, b_high);
                return vcombine_s8(vqmovn_s16(prod_low), vqmovn_s16(prod_high));
            };
            const int8x16_t va0 = vld1q_s8(&this_data[i + 0]);
            const int8x16_t va1 = vld1q_s8(&this_data[i + 16]);
            const int8x16_t vb0 = vld1q_s8(&other_data[i + 0]);
            const int8x16_t vb1 = vld1q_s8(&other_data[i + 16]);
            vst1q_s8(&this_data[i + 0],  mul_int8x16(va0, vb0));
            vst1q_s8(&this_data[i + 16], mul_int8x16(va1, vb1));
        }
        // Tail: 16 int8s
        if (len16 > len32) {
            const std::size_t i = len32;
            const int8x16_t va = vld1q_s8(&this_data[i]);
            const int8x16_t vb = vld1q_s8(&other_data[i]);
            const int16x8_t va_low = vmovl_s8(vget_low_s8(va));
            const int16x8_t va_high = vmovl_s8(vget_high_s8(va));
            const int16x8_t vb_low = vmovl_s8(vget_low_s8(vb));
            const int16x8_t vb_high = vmovl_s8(vget_high_s8(vb));
            const int16x8_t prod_low = vmulq_s16(va_low, vb_low);
            const int16x8_t prod_high = vmulq_s16(va_high, vb_high);
            vst1q_s8(&this_data[i], vcombine_s8(vqmovn_s16(prod_low), vqmovn_s16(prod_high)));
        }
        // Tail: 8 int8s
        if (len8 > len16) {
            const std::size_t i = len16;
            const int8x8_t va = vld1_s8(&this_data[i]);
            const int8x8_t vb = vld1_s8(&other_data[i]);
            const int16x8_t va16 = vmovl_s8(va);
            const int16x8_t vb16 = vmovl_s8(vb);
            const int16x8_t prod = vmulq_s16(va16, vb16);
            vst1_s8(&this_data[i], vqmovn_s16(prod));
        }
        // Scalar tail
        for (std::size_t i = len8; i < size_; ++i) {
            int16_t prod = static_cast<int16_t>(this_data[i]) * static_cast<int16_t>(other_data[i]);
            this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(prod))));
        }
        return;
    }

    // DIV path for INT8 - optimized using float conversion with zero masking
    // NEON doesn't have int8 division, so we convert to float32, divide, then convert back
    if (op_type == OperationType::DIV) {
        const std::size_t len8 = (size_ / 8) * 8;

        // Process 8 int8s at a time
        for (std::size_t i = 0; i < len8; i += 8) {
            // Load 8 int8 values
            const int8x8_t va = vld1_s8(&this_data[i]);
            const int8x8_t vb = vld1_s8(&other_data[i]);

            // Mask for zeros in divisor
            const uint8x8_t zero_mask = vceq_s8(vb, vdup_n_s8(0));

            // Widen int8 → int16
            const int16x8_t va16 = vmovl_s8(va);
            const int16x8_t vb16 = vmovl_s8(vb);

            // Widen int16 → int32 (process low and high halves)
            const int32x4_t va32_low = vmovl_s16(vget_low_s16(va16));
            const int32x4_t va32_high = vmovl_s16(vget_high_s16(va16));
            const int32x4_t vb32_low = vmovl_s16(vget_low_s16(vb16));
            const int32x4_t vb32_high = vmovl_s16(vget_high_s16(vb16));

            // Convert to float32
            const float32x4_t vaf_low = vcvtq_f32_s32(va32_low);
            const float32x4_t vaf_high = vcvtq_f32_s32(va32_high);
            const float32x4_t vbf_low = vcvtq_f32_s32(vb32_low);
            const float32x4_t vbf_high = vcvtq_f32_s32(vb32_high);

            // Perform division in float32
            const float32x4_t vr_low = vdivq_f32(vaf_low, vbf_low);
            const float32x4_t vr_high = vdivq_f32(vaf_high, vbf_high);

            // Convert back to int32
            const int32x4_t vri32_low = vcvtq_s32_f32(vr_low);
            const int32x4_t vri32_high = vcvtq_s32_f32(vr_high);

            // Narrow int32 → int16 with saturation
            const int16x8_t vri16 = vcombine_s16(vqmovn_s32(vri32_low), vqmovn_s32(vri32_high));

            // Narrow int16 → int8 with saturation
            const int8x8_t result = vqmovn_s16(vri16);

            // Zero-out lanes where divisor was zero
            const int8x8_t safe_result = vbsl_s8(zero_mask, vdup_n_s8(0), result);

            vst1_s8(&this_data[i], safe_result);
        }

        // Scalar tail for remaining elements
        for (std::size_t i = len8; i < size_; ++i) {
            if (other_data[i] == 0) {
                this_data[i] = 0;  // Division by zero yields 0
            } else {
                this_data[i] = static_cast<int8_t>(this_data[i] / other_data[i]);
            }
        }
        return;
    }

    // SUB path (default)
    // Main loop: 128 int8s (8 registers × 16 int8s)
    for (std::size_t i = 0; i < len128; i += 128) {
        if (i + PREFETCH_DISTANCE < size_) {
            __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
            __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
        }
        const int8x16_t va0 = vld1q_s8(&this_data[i + 0]);
        const int8x16_t va1 = vld1q_s8(&this_data[i + 16]);
        const int8x16_t va2 = vld1q_s8(&this_data[i + 32]);
        const int8x16_t va3 = vld1q_s8(&this_data[i + 48]);
        const int8x16_t va4 = vld1q_s8(&this_data[i + 64]);
        const int8x16_t va5 = vld1q_s8(&this_data[i + 80]);
        const int8x16_t va6 = vld1q_s8(&this_data[i + 96]);
        const int8x16_t va7 = vld1q_s8(&this_data[i + 112]);
        
        const int8x16_t vb0 = vld1q_s8(&other_data[i + 0]);
        const int8x16_t vb1 = vld1q_s8(&other_data[i + 16]);
        const int8x16_t vb2 = vld1q_s8(&other_data[i + 32]);
        const int8x16_t vb3 = vld1q_s8(&other_data[i + 48]);
        const int8x16_t vb4 = vld1q_s8(&other_data[i + 64]);
        const int8x16_t vb5 = vld1q_s8(&other_data[i + 80]);
        const int8x16_t vb6 = vld1q_s8(&other_data[i + 96]);
        const int8x16_t vb7 = vld1q_s8(&other_data[i + 112]);
        
        vst1q_s8(&this_data[i + 0],   vsubq_s8(va0, vb0));
        vst1q_s8(&this_data[i + 16],  vsubq_s8(va1, vb1));
        vst1q_s8(&this_data[i + 32],  vsubq_s8(va2, vb2));
        vst1q_s8(&this_data[i + 48],  vsubq_s8(va3, vb3));
        vst1q_s8(&this_data[i + 64],  vsubq_s8(va4, vb4));
        vst1q_s8(&this_data[i + 80],  vsubq_s8(va5, vb5));
        vst1q_s8(&this_data[i + 96],  vsubq_s8(va6, vb6));
        vst1q_s8(&this_data[i + 112], vsubq_s8(va7, vb7));
    }
    // Tail: 64 int8s
    if (len64 > len128) {
        const std::size_t i = len128;
        const int8x16_t va0 = vld1q_s8(&this_data[i + 0]);
        const int8x16_t va1 = vld1q_s8(&this_data[i + 16]);
        const int8x16_t va2 = vld1q_s8(&this_data[i + 32]);
        const int8x16_t va3 = vld1q_s8(&this_data[i + 48]);
        const int8x16_t vb0 = vld1q_s8(&other_data[i + 0]);
        const int8x16_t vb1 = vld1q_s8(&other_data[i + 16]);
        const int8x16_t vb2 = vld1q_s8(&other_data[i + 32]);
        const int8x16_t vb3 = vld1q_s8(&other_data[i + 48]);
        vst1q_s8(&this_data[i + 0],  vsubq_s8(va0, vb0));
        vst1q_s8(&this_data[i + 16], vsubq_s8(va1, vb1));
        vst1q_s8(&this_data[i + 32], vsubq_s8(va2, vb2));
        vst1q_s8(&this_data[i + 48], vsubq_s8(va3, vb3));
    }
    // Tail: 32 int8s
    if (len32 > len64) {
        const std::size_t i = len64;
        const int8x16_t va0 = vld1q_s8(&this_data[i + 0]);
        const int8x16_t va1 = vld1q_s8(&this_data[i + 16]);
        const int8x16_t vb0 = vld1q_s8(&other_data[i + 0]);
        const int8x16_t vb1 = vld1q_s8(&other_data[i + 16]);
        vst1q_s8(&this_data[i + 0],  vsubq_s8(va0, vb0));
        vst1q_s8(&this_data[i + 16], vsubq_s8(va1, vb1));
    }
    // Tail: 16 int8s
    if (len16 > len32) {
        const std::size_t i = len32;
        const int8x16_t va = vld1q_s8(&this_data[i]);
        const int8x16_t vb = vld1q_s8(&other_data[i]);
        vst1q_s8(&this_data[i], vsubq_s8(va, vb));
    }
    // Tail: 8 int8s
    if (len8 > len16) {
        const std::size_t i = len16;
        const int8x8_t va = vld1_s8(&this_data[i]);
        const int8x8_t vb = vld1_s8(&other_data[i]);
        vst1_s8(&this_data[i], vsub_s8(va, vb));
    }
    // Scalar tail
    for (std::size_t i = len8; i < size_; ++i) {
        int16_t diff = static_cast<int16_t>(this_data[i]) - static_cast<int16_t>(other_data[i]);
        this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(diff))));
    }
}

// =======================
// INT16 kernels (NEON optimized)
// =======================

void NEONBackend::operation_int16(const TensorBackend& other, OperationType op_type) {
    const int16_t* other_data = other.data_int16();
    int16_t* this_data = data_int16();

    // Prefetch distance for INT16: 128 int16s (~256 bytes)
    constexpr std::size_t PREFETCH_DISTANCE = 128;

    // Hierarchical processing: 64 → 32 → 16 → 8 → 4 → scalar
    const std::size_t len64 = (size_ / 64) * 64;
    const std::size_t len32 = len64 + ((size_ - len64) / 32) * 32;
    const std::size_t len16 = len32 + ((size_ - len32) / 16) * 16;
    const std::size_t len8  = len16 + ((size_ - len16) / 8) * 8;
    const std::size_t len4  = len8  + ((size_ - len8) / 4) * 4;

    if (op_type == OperationType::ADD) {
        // Main loop: 64 int16s (8 registers × 8 int16s)
        for (std::size_t i = 0; i < len64; i += 64) {
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
            }
            const int16x8_t va0 = vld1q_s16(&this_data[i + 0]);
            const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
            const int16x8_t va2 = vld1q_s16(&this_data[i + 16]);
            const int16x8_t va3 = vld1q_s16(&this_data[i + 24]);
            const int16x8_t va4 = vld1q_s16(&this_data[i + 32]);
            const int16x8_t va5 = vld1q_s16(&this_data[i + 40]);
            const int16x8_t va6 = vld1q_s16(&this_data[i + 48]);
            const int16x8_t va7 = vld1q_s16(&this_data[i + 56]);

            const int16x8_t vb0 = vld1q_s16(&other_data[i + 0]);
            const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);
            const int16x8_t vb2 = vld1q_s16(&other_data[i + 16]);
            const int16x8_t vb3 = vld1q_s16(&other_data[i + 24]);
            const int16x8_t vb4 = vld1q_s16(&other_data[i + 32]);
            const int16x8_t vb5 = vld1q_s16(&other_data[i + 40]);
            const int16x8_t vb6 = vld1q_s16(&other_data[i + 48]);
            const int16x8_t vb7 = vld1q_s16(&other_data[i + 56]);

            vst1q_s16(&this_data[i + 0],  vqaddq_s16(va0, vb0));
            vst1q_s16(&this_data[i + 8],  vqaddq_s16(va1, vb1));
            vst1q_s16(&this_data[i + 16], vqaddq_s16(va2, vb2));
            vst1q_s16(&this_data[i + 24], vqaddq_s16(va3, vb3));
            vst1q_s16(&this_data[i + 32], vqaddq_s16(va4, vb4));
            vst1q_s16(&this_data[i + 40], vqaddq_s16(va5, vb5));
            vst1q_s16(&this_data[i + 48], vqaddq_s16(va6, vb6));
            vst1q_s16(&this_data[i + 56], vqaddq_s16(va7, vb7));
        }
        // Tail: 32 int16s (4 registers × 8 int16s)
        if (len32 > len64) {
            const std::size_t i = len64;
            const int16x8_t va0 = vld1q_s16(&this_data[i + 0]);
            const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
            const int16x8_t va2 = vld1q_s16(&this_data[i + 16]);
            const int16x8_t va3 = vld1q_s16(&this_data[i + 24]);
            const int16x8_t vb0 = vld1q_s16(&other_data[i + 0]);
            const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);
            const int16x8_t vb2 = vld1q_s16(&other_data[i + 16]);
            const int16x8_t vb3 = vld1q_s16(&other_data[i + 24]);
            vst1q_s16(&this_data[i + 0],  vqaddq_s16(va0, vb0));
            vst1q_s16(&this_data[i + 8],  vqaddq_s16(va1, vb1));
            vst1q_s16(&this_data[i + 16], vqaddq_s16(va2, vb2));
            vst1q_s16(&this_data[i + 24], vqaddq_s16(va3, vb3));
        }
        // Tail: 16 int16s (2 registers × 8 int16s)
        if (len16 > len32) {
            const std::size_t i = len32;
            const int16x8_t va0 = vld1q_s16(&this_data[i + 0]);
            const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
            const int16x8_t vb0 = vld1q_s16(&other_data[i + 0]);
            const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);
            vst1q_s16(&this_data[i + 0], vqaddq_s16(va0, vb0));
            vst1q_s16(&this_data[i + 8], vqaddq_s16(va1, vb1));
        }
        // Tail: 8 int16s (1 register)
        if (len8 > len16) {
            const std::size_t i = len16;
            const int16x8_t va = vld1q_s16(&this_data[i]);
            const int16x8_t vb = vld1q_s16(&other_data[i]);
            vst1q_s16(&this_data[i], vqaddq_s16(va, vb));
        }
        // Tail: 4 int16s (half register)
        if (len4 > len8) {
            const std::size_t i = len8;
            const int16x4_t va = vld1_s16(&this_data[i]);
            const int16x4_t vb = vld1_s16(&other_data[i]);
            vst1_s16(&this_data[i], vqadd_s16(va, vb));
        }
        // Scalar tail
        for (std::size_t i = len4; i < size_; ++i) {
            int32_t sum = static_cast<int32_t>(this_data[i]) + static_cast<int32_t>(other_data[i]);
            this_data[i] = static_cast<int16_t>(std::max(-32768, std::min(32767, sum)));
        }
        return;
    }

    if (op_type == OperationType::MUL) {
        // Helper lambda for int16 multiplication with saturation
        auto mul_int16x8_sat = [](const int16x8_t& a, const int16x8_t& b) -> int16x8_t {
            // Split into low and high halves
            const int16x4_t a_low = vget_low_s16(a);
            const int16x4_t a_high = vget_high_s16(a);
            const int16x4_t b_low = vget_low_s16(b);
            const int16x4_t b_high = vget_high_s16(b);
            
            // Widen to int32 and multiply
            const int32x4_t prod_low = vmull_s16(a_low, b_low);
            const int32x4_t prod_high = vmull_s16(a_high, b_high);
            
            // Narrow back to int16 with saturation
            return vcombine_s16(vqmovn_s32(prod_low), vqmovn_s32(prod_high));
        };

        // Main loop: 64 int16s (8 registers × 8 int16s)
        for (std::size_t i = 0; i < len64; i += 64) {
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
            }
            const int16x8_t va0 = vld1q_s16(&this_data[i + 0]);
            const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
            const int16x8_t va2 = vld1q_s16(&this_data[i + 16]);
            const int16x8_t va3 = vld1q_s16(&this_data[i + 24]);
            const int16x8_t va4 = vld1q_s16(&this_data[i + 32]);
            const int16x8_t va5 = vld1q_s16(&this_data[i + 40]);
            const int16x8_t va6 = vld1q_s16(&this_data[i + 48]);
            const int16x8_t va7 = vld1q_s16(&this_data[i + 56]);

            const int16x8_t vb0 = vld1q_s16(&other_data[i + 0]);
            const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);
            const int16x8_t vb2 = vld1q_s16(&other_data[i + 16]);
            const int16x8_t vb3 = vld1q_s16(&other_data[i + 24]);
            const int16x8_t vb4 = vld1q_s16(&other_data[i + 32]);
            const int16x8_t vb5 = vld1q_s16(&other_data[i + 40]);
            const int16x8_t vb6 = vld1q_s16(&other_data[i + 48]);
            const int16x8_t vb7 = vld1q_s16(&other_data[i + 56]);

            vst1q_s16(&this_data[i + 0],  mul_int16x8_sat(va0, vb0));
            vst1q_s16(&this_data[i + 8],  mul_int16x8_sat(va1, vb1));
            vst1q_s16(&this_data[i + 16], mul_int16x8_sat(va2, vb2));
            vst1q_s16(&this_data[i + 24], mul_int16x8_sat(va3, vb3));
            vst1q_s16(&this_data[i + 32], mul_int16x8_sat(va4, vb4));
            vst1q_s16(&this_data[i + 40], mul_int16x8_sat(va5, vb5));
            vst1q_s16(&this_data[i + 48], mul_int16x8_sat(va6, vb6));
            vst1q_s16(&this_data[i + 56], mul_int16x8_sat(va7, vb7));
        }
        // Tail: 32 int16s
        if (len32 > len64) {
            const std::size_t i = len64;
            const int16x8_t va0 = vld1q_s16(&this_data[i + 0]);
            const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
            const int16x8_t va2 = vld1q_s16(&this_data[i + 16]);
            const int16x8_t va3 = vld1q_s16(&this_data[i + 24]);
            const int16x8_t vb0 = vld1q_s16(&other_data[i + 0]);
            const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);
            const int16x8_t vb2 = vld1q_s16(&other_data[i + 16]);
            const int16x8_t vb3 = vld1q_s16(&other_data[i + 24]);
            vst1q_s16(&this_data[i + 0],  mul_int16x8_sat(va0, vb0));
            vst1q_s16(&this_data[i + 8],  mul_int16x8_sat(va1, vb1));
            vst1q_s16(&this_data[i + 16], mul_int16x8_sat(va2, vb2));
            vst1q_s16(&this_data[i + 24], mul_int16x8_sat(va3, vb3));
        }
        // Tail: 16 int16s
        if (len16 > len32) {
            const std::size_t i = len32;
            const int16x8_t va0 = vld1q_s16(&this_data[i + 0]);
            const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
            const int16x8_t vb0 = vld1q_s16(&other_data[i + 0]);
            const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);
            vst1q_s16(&this_data[i + 0], mul_int16x8_sat(va0, vb0));
            vst1q_s16(&this_data[i + 8], mul_int16x8_sat(va1, vb1));
        }
        // Tail: 8 int16s
        if (len8 > len16) {
            const std::size_t i = len16;
            const int16x8_t va = vld1q_s16(&this_data[i]);
            const int16x8_t vb = vld1q_s16(&other_data[i]);
            vst1q_s16(&this_data[i], mul_int16x8_sat(va, vb));
        }
        // Tail: 4 int16s
        if (len4 > len8) {
            const std::size_t i = len8;
            const int16x4_t va = vld1_s16(&this_data[i]);
            const int16x4_t vb = vld1_s16(&other_data[i]);
            const int32x4_t prod = vmull_s16(va, vb);
            vst1_s16(&this_data[i], vqmovn_s32(prod));
        }
        // Scalar tail
        for (std::size_t i = len4; i < size_; ++i) {
            int32_t product = static_cast<int32_t>(this_data[i]) * static_cast<int32_t>(other_data[i]);
            this_data[i] = static_cast<int16_t>(std::max(-32768, std::min(32767, product)));
        }
        return;
    }

    // DIV path for INT16 - optimized using float conversion with zero masking
    // NEON doesn't have int16 division, so we convert to float32, divide, then convert back
    if (op_type == OperationType::DIV) {
        const std::size_t len16 = (size_ / 16) * 16;
        const std::size_t len8 = (size_ / 8) * 8;
        const std::size_t len4 = (size_ / 4) * 4;

        // Process 16 int16s at a time (two sets of 8)
        for (std::size_t i = 0; i < len16; i += 16) {
            // First 8 int16s
            const int16x8_t va0 = vld1q_s16(&this_data[i]);
            const int16x8_t vb0 = vld1q_s16(&other_data[i]);

            const uint16x8_t zero_mask0 = vceqq_s16(vb0, vdupq_n_s16(0));

            const int32x4_t va32_0_low = vmovl_s16(vget_low_s16(va0));
            const int32x4_t va32_0_high = vmovl_s16(vget_high_s16(va0));
            const int32x4_t vb32_0_low = vmovl_s16(vget_low_s16(vb0));
            const int32x4_t vb32_0_high = vmovl_s16(vget_high_s16(vb0));

            const float32x4_t vr0_low = vdivq_f32(vcvtq_f32_s32(va32_0_low), vcvtq_f32_s32(vb32_0_low));
            const float32x4_t vr0_high = vdivq_f32(vcvtq_f32_s32(va32_0_high), vcvtq_f32_s32(vb32_0_high));

            const int16x8_t result0 = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(vr0_low)), vqmovn_s32(vcvtq_s32_f32(vr0_high)));
            const int16x8_t safe0 = vbslq_s16(zero_mask0, vdupq_n_s16(0), result0);
            vst1q_s16(&this_data[i], safe0);

            // Second 8 int16s
            const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
            const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);

            const uint16x8_t zero_mask1 = vceqq_s16(vb1, vdupq_n_s16(0));

            const int32x4_t va32_1_low = vmovl_s16(vget_low_s16(va1));
            const int32x4_t va32_1_high = vmovl_s16(vget_high_s16(va1));
            const int32x4_t vb32_1_low = vmovl_s16(vget_low_s16(vb1));
            const int32x4_t vb32_1_high = vmovl_s16(vget_high_s16(vb1));

            const float32x4_t vr1_low = vdivq_f32(vcvtq_f32_s32(va32_1_low), vcvtq_f32_s32(vb32_1_low));
            const float32x4_t vr1_high = vdivq_f32(vcvtq_f32_s32(va32_1_high), vcvtq_f32_s32(vb32_1_high));

            const int16x8_t result1 = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(vr1_low)), vqmovn_s32(vcvtq_s32_f32(vr1_high)));
            const int16x8_t safe1 = vbslq_s16(zero_mask1, vdupq_n_s16(0), result1);
            vst1q_s16(&this_data[i + 8], safe1);
        }

        // Tail: 8 int16s
        if (len8 > len16) {
            const std::size_t i = len16;
            const int16x8_t va = vld1q_s16(&this_data[i]);
            const int16x8_t vb = vld1q_s16(&other_data[i]);

            const uint16x8_t zero_mask = vceqq_s16(vb, vdupq_n_s16(0));

            const int32x4_t va32_low = vmovl_s16(vget_low_s16(va));
            const int32x4_t va32_high = vmovl_s16(vget_high_s16(va));
            const int32x4_t vb32_low = vmovl_s16(vget_low_s16(vb));
            const int32x4_t vb32_high = vmovl_s16(vget_high_s16(vb));

            const float32x4_t vr_low = vdivq_f32(vcvtq_f32_s32(va32_low), vcvtq_f32_s32(vb32_low));
            const float32x4_t vr_high = vdivq_f32(vcvtq_f32_s32(va32_high), vcvtq_f32_s32(vb32_high));

            const int16x8_t result = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(vr_low)), vqmovn_s32(vcvtq_s32_f32(vr_high)));
            const int16x8_t safe = vbslq_s16(zero_mask, vdupq_n_s16(0), result);
            vst1q_s16(&this_data[i], safe);
        }

        // Tail: 4 int16s
        if (len4 > len8) {
            const std::size_t i = len8;
            const int16x4_t va = vld1_s16(&this_data[i]);
            const int16x4_t vb = vld1_s16(&other_data[i]);

            const uint16x4_t zero_mask = vceq_s16(vb, vdup_n_s16(0));

            const int32x4_t va32 = vmovl_s16(va);
            const int32x4_t vb32 = vmovl_s16(vb);

            const float32x4_t vr = vdivq_f32(vcvtq_f32_s32(va32), vcvtq_f32_s32(vb32));
            const int16x4_t result = vqmovn_s32(vcvtq_s32_f32(vr));
            const int16x4_t safe = vbsl_s16(zero_mask, vdup_n_s16(0), result);

            vst1_s16(&this_data[i], safe);
        }

        // Scalar tail for remaining elements
        for (std::size_t i = len4; i < size_; ++i) {
            if (other_data[i] == 0) {
                this_data[i] = 0;  // Division by zero yields 0
            } else {
                this_data[i] = static_cast<int16_t>(this_data[i] / other_data[i]);
            }
        }
        return;
    }

    // SUB path (default)
    for (std::size_t i = 0; i < len64; i += 64) {
        if (i + PREFETCH_DISTANCE < size_) {
            __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
            __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
        }
        const int16x8_t va0 = vld1q_s16(&this_data[i + 0]);
        const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
        const int16x8_t va2 = vld1q_s16(&this_data[i + 16]);
        const int16x8_t va3 = vld1q_s16(&this_data[i + 24]);
        const int16x8_t va4 = vld1q_s16(&this_data[i + 32]);
        const int16x8_t va5 = vld1q_s16(&this_data[i + 40]);
        const int16x8_t va6 = vld1q_s16(&this_data[i + 48]);
        const int16x8_t va7 = vld1q_s16(&this_data[i + 56]);

        const int16x8_t vb0 = vld1q_s16(&other_data[i + 0]);
        const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);
        const int16x8_t vb2 = vld1q_s16(&other_data[i + 16]);
        const int16x8_t vb3 = vld1q_s16(&other_data[i + 24]);
        const int16x8_t vb4 = vld1q_s16(&other_data[i + 32]);
        const int16x8_t vb5 = vld1q_s16(&other_data[i + 40]);
        const int16x8_t vb6 = vld1q_s16(&other_data[i + 48]);
        const int16x8_t vb7 = vld1q_s16(&other_data[i + 56]);

        vst1q_s16(&this_data[i + 0],  vqsubq_s16(va0, vb0));
        vst1q_s16(&this_data[i + 8],  vqsubq_s16(va1, vb1));
        vst1q_s16(&this_data[i + 16], vqsubq_s16(va2, vb2));
        vst1q_s16(&this_data[i + 24], vqsubq_s16(va3, vb3));
        vst1q_s16(&this_data[i + 32], vqsubq_s16(va4, vb4));
        vst1q_s16(&this_data[i + 40], vqsubq_s16(va5, vb5));
        vst1q_s16(&this_data[i + 48], vqsubq_s16(va6, vb6));
        vst1q_s16(&this_data[i + 56], vqsubq_s16(va7, vb7));
    }
    // Tail: 32 int16s
    if (len32 > len64) {
        const std::size_t i = len64;
        const int16x8_t va0 = vld1q_s16(&this_data[i + 0]);
        const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
        const int16x8_t va2 = vld1q_s16(&this_data[i + 16]);
        const int16x8_t va3 = vld1q_s16(&this_data[i + 24]);
        const int16x8_t vb0 = vld1q_s16(&other_data[i + 0]);
        const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);
        const int16x8_t vb2 = vld1q_s16(&other_data[i + 16]);
        const int16x8_t vb3 = vld1q_s16(&other_data[i + 24]);
        vst1q_s16(&this_data[i + 0],  vqsubq_s16(va0, vb0));
        vst1q_s16(&this_data[i + 8],  vqsubq_s16(va1, vb1));
        vst1q_s16(&this_data[i + 16], vqsubq_s16(va2, vb2));
        vst1q_s16(&this_data[i + 24], vqsubq_s16(va3, vb3));
    }
    // Tail: 16 int16s
    if (len16 > len32) {
        const std::size_t i = len32;
        const int16x8_t va0 = vld1q_s16(&this_data[i + 0]);
        const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
        const int16x8_t vb0 = vld1q_s16(&other_data[i + 0]);
        const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);
        vst1q_s16(&this_data[i + 0], vqsubq_s16(va0, vb0));
        vst1q_s16(&this_data[i + 8], vqsubq_s16(va1, vb1));
    }
    // Tail: 8 int16s
    if (len8 > len16) {
        const std::size_t i = len16;
        const int16x8_t va = vld1q_s16(&this_data[i]);
        const int16x8_t vb = vld1q_s16(&other_data[i]);
        vst1q_s16(&this_data[i], vqsubq_s16(va, vb));
    }
    // Tail: 4 int16s
    if (len4 > len8) {
        const std::size_t i = len8;
        const int16x4_t va = vld1_s16(&this_data[i]);
        const int16x4_t vb = vld1_s16(&other_data[i]);
        vst1_s16(&this_data[i], vqsub_s16(va, vb));
    }
    // Scalar tail
    for (std::size_t i = len4; i < size_; ++i) {
        int32_t diff = static_cast<int32_t>(this_data[i]) - static_cast<int32_t>(other_data[i]);
        this_data[i] = static_cast<int16_t>(std::max(-32768, std::min(32767, diff)));
    }
}

} // namespace labneura

#endif // LABNEURA_HAVE_M1