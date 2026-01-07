#if defined(__ARM_NEON)

#include "labneura/backends/neon.h"
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

    aligned_size_ = ((size + 3) / 4) * 4;

    if (mode == QuantizationMode::FP32) {
        float* p = labneura::util::allocate_aligned<float>(aligned_size_, labneura::util::NEON_ALIGN);
        data_fp32_ = std::unique_ptr<float, void(*)(float*)>(p, labneura::util::free_aligned_float);
        std::fill(data_fp32_.get() + size_, data_fp32_.get() + aligned_size_, 0.0f);
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

// =======================
// Clone
// =======================

std::unique_ptr<TensorBackend> NEONBackend::clone() const {
    auto backend = std::make_unique<NEONBackend>(size_, quantization_mode_);
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

void NEONBackend::operation_fp32(const TensorBackend& other, OperationType op_type) {
    const float* other_data = other.data_fp32();
    float* this_data = data_fp32();

    // Hierarchical processing: 32 → 16 → 8 → 4 → scalar
    const std::size_t len32 = (size_ / 32) * 32;
    const std::size_t len16 = len32 + ((size_ - len32) / 16) * 16;
    const std::size_t len8  = len16 + ((size_ - len16) / 8) * 8;
    const std::size_t len4  = len8  + ((size_ - len8) / 4) * 4;

    if (op_type == OperationType::ADD) {
        // Main loop: 32 floats (8 registers × 4 floats)
        for (std::size_t i = 0; i < len32; i += 32) {
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

    // SUB path (default)
    for (std::size_t i = 0; i < len32; i += 32) {
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
// INT8 kernels
// =======================

void NEONBackend::operation_int8(const TensorBackend& other, OperationType op_type) {
    const int8_t* other_data = other.data_int8();
    int8_t* this_data = data_int8();

    const std::size_t neon_len = (size_ / 16) * 16;
    const std::size_t len8     = ((size_ - neon_len) / 8) * 8 + neon_len;
    const std::size_t len4     = ((size_ - len8) / 4) * 4 + len8;

    if (op_type == OperationType::ADD) {
        for (std::size_t i = 0; i < neon_len; i += 16) {
            const int8x16_t va = vld1q_s8(&this_data[i]);
            const int8x16_t vb = vld1q_s8(&other_data[i]);
            const int8x16_t vr = vaddq_s8(va, vb);
            vst1q_s8(&this_data[i], vr);
        }
        for (std::size_t i = neon_len; i < len8; i += 8) {
            const int8x8_t va = vld1_s8(&this_data[i]);
            const int8x8_t vb = vld1_s8(&other_data[i]);
            const int8x8_t vr = vadd_s8(va, vb);
            vst1_s8(&this_data[i], vr);
        }
        for (std::size_t i = len8; i < len4; i += 4) {
            const int8x8_t va = vld1_s8(&this_data[i]);
            const int8x8_t vb = vld1_s8(&other_data[i]);
            const int8x8_t vr = vadd_s8(va, vb);
            vst1_s8(&this_data[i], vr);
        }
        for (std::size_t i = len4; i < size_; ++i) {
            int16_t sum = static_cast<int16_t>(this_data[i]) + static_cast<int16_t>(other_data[i]);
            this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(sum))));
        }
        return;
    }

    if (op_type == OperationType::MUL) {
        for (std::size_t i = 0; i < neon_len; i += 16) {
            const int8x16_t va = vld1q_s8(&this_data[i]);
            const int8x16_t vb = vld1q_s8(&other_data[i]);
            const int16x8_t va_low = vmovl_s8(vget_low_s8(va));
            const int16x8_t va_high = vmovl_s8(vget_high_s8(va));
            const int16x8_t vb_low = vmovl_s8(vget_low_s8(vb));
            const int16x8_t vb_high = vmovl_s8(vget_high_s8(vb));
            const int16x8_t prod_low = vmulq_s16(va_low, vb_low);
            const int16x8_t prod_high = vmulq_s16(va_high, vb_high);
            const int8x16_t vr = vcombine_s8(vqmovn_s16(prod_low), vqmovn_s16(prod_high));
            vst1q_s8(&this_data[i], vr);
        }
        for (std::size_t i = neon_len; i < len8; i += 8) {
            const int8x8_t va = vld1_s8(&this_data[i]);
            const int8x8_t vb = vld1_s8(&other_data[i]);
            const int16x8_t va16 = vmovl_s8(va);
            const int16x8_t vb16 = vmovl_s8(vb);
            const int16x8_t prod = vmulq_s16(va16, vb16);
            const int8x8_t vr = vqmovn_s16(prod);
            vst1_s8(&this_data[i], vr);
        }
        for (std::size_t i = len8; i < len4; i += 4) {
            const int8x8_t va = vld1_s8(&this_data[i]);
            const int8x8_t vb = vld1_s8(&other_data[i]);
            const int16x8_t va16 = vmovl_s8(va);
            const int16x8_t vb16 = vmovl_s8(vb);
            const int16x8_t prod = vmulq_s16(va16, vb16);
            const int8x8_t vr = vqmovn_s16(prod);
            vst1_s8(&this_data[i], vr);
        }
        for (std::size_t i = len4; i < size_; ++i) {
            int16_t prod = static_cast<int16_t>(this_data[i]) * static_cast<int16_t>(other_data[i]);
            this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(prod))));
        }
        return;
    }

    for (std::size_t i = 0; i < neon_len; i += 16) {
        const int8x16_t va = vld1q_s8(&this_data[i]);
        const int8x16_t vb = vld1q_s8(&other_data[i]);
        const int8x16_t vr = vsubq_s8(va, vb);
        vst1q_s8(&this_data[i], vr);
    }
    for (std::size_t i = neon_len; i < len8; i += 8) {
        const int8x8_t va = vld1_s8(&this_data[i]);
        const int8x8_t vb = vld1_s8(&other_data[i]);
        const int8x8_t vr = vsub_s8(va, vb);
        vst1_s8(&this_data[i], vr);
    }
    for (std::size_t i = len8; i < len4; i += 4) {
        const int8x8_t va = vld1_s8(&this_data[i]);
        const int8x8_t vb = vld1_s8(&other_data[i]);
        const int8x8_t vr = vsub_s8(va, vb);
        vst1_s8(&this_data[i], vr);
    }
    for (std::size_t i = len4; i < size_; ++i) {
        int16_t diff = static_cast<int16_t>(this_data[i]) - static_cast<int16_t>(other_data[i]);
        this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(diff))));
    }
}

} // namespace labneura

#endif // LABNEURA_HAVE_M1