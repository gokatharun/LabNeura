#if defined(__ARM_NEON)

#include "labneura/backends/neon.h"
#include <arm_neon.h>

namespace labneura {

// =======================
// Constructor
// =======================

NEONBackend::NEONBackend(std::size_t size, QuantizationMode mode)
    : size_(size), quantization_mode_(mode) {

    aligned_size_ = ((size + 3) / 4) * 4;

    if (mode == QuantizationMode::FP32) {
        data_fp32_ = std::make_unique<float[]>(aligned_size_);
        std::fill(data_fp32_.get() + size_, data_fp32_.get() + aligned_size_, 0.0f);
    } else {
        data_int8_ = std::make_unique<int8_t[]>(aligned_size_);
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

    std::size_t len16 = (size_ / 16) * 16;
    for (std::size_t i = 0; i < len16; i += 16) {
        for (std::size_t j = 0; j < 16; j += 4) {
            float32x4_t va = vld1q_f32(&this_data[i + j]);
            float32x4_t vb = vld1q_f32(&other_data[i + j]);
            float32x4_t result;
            switch (op_type) {
                case OperationType::ADD:
                    result = vaddq_f32(va, vb);
                    break;
                case OperationType::MUL:
                    result = vmulq_f32(va, vb);
                    break;
                case OperationType::SUB:
                    result = vsubq_f32(va, vb);
                    break;
            }
            vst1q_f32(&this_data[i + j], result);
        }
    }

    // Handle remaining elements...
    std::size_t len8 = ((size_ - len16) / 8) * 8 + len16;
    for (std::size_t i = len16; i < len8; i += 8) {
        for (std::size_t j = 0; j < 8; j += 4) {
            float32x4_t va = vld1q_f32(&this_data[i + j]);
            float32x4_t vb = vld1q_f32(&other_data[i + j]);
            float32x4_t result;
            switch (op_type) {
                case OperationType::ADD:
                    result = vaddq_f32(va, vb);     
                    break;
                case OperationType::MUL:
                    result = vmulq_f32(va, vb);
                    break;
                case OperationType::SUB:
                    result = vsubq_f32(va, vb);
                    break;
            }
            vst1q_f32(&this_data[i + j], result);
        }
    }

    std::size_t len4 = ((size_ - len8) / 4) * 4 + len8;
    for (std::size_t i = len8; i < len4; i += 4) {
        float32x4_t va = vld1q_f32(&this_data[i]);
        float32x4_t vb = vld1q_f32(&other_data[i]);
        float32x4_t result;
        switch (op_type) {
            case OperationType::ADD:
                result = vaddq_f32(va, vb);
                break;
            case OperationType::MUL:
                result = vmulq_f32(va, vb);
                break;
            case OperationType::SUB:
                result = vsubq_f32(va, vb);
                break;
        }
        vst1q_f32(&this_data[i], result);
    }

    for (std::size_t i = len4; i < size_; ++i) {
        switch (op_type) {
            case OperationType::ADD:
                this_data[i] += other_data[i];
                break;
            case OperationType::MUL:
                this_data[i] *= other_data[i];
                break;
            case OperationType::SUB:
                this_data[i] -= other_data[i];
                break;
        }
    }
}

// =======================
// INT8 kernels
// =======================

void NEONBackend::operation_int8(const TensorBackend& other, OperationType op_type) {
    const int8_t* other_data = other.data_int8();
    int8_t* this_data = data_int8();

    std::size_t neon_len = (size_ / 16) * 16;
    for (std::size_t i = 0; i < neon_len; i += 16) {
        int8x16_t va = vld1q_s8(&this_data[i]);
        int8x16_t vb = vld1q_s8(&other_data[i]);
        int8x16_t result;
        switch (op_type) {
            case OperationType::ADD:
                result = vaddq_s8(va, vb);
                break;
            case OperationType::MUL: {
                int16x8_t va_low = vmovl_s8(vget_low_s8(va));
                int16x8_t va_high = vmovl_s8(vget_high_s8(va));
                int16x8_t vb_low = vmovl_s8(vget_low_s8(vb));
                int16x8_t vb_high = vmovl_s8(vget_high_s8(vb));
                int16x8_t prod_low = vmulq_s16(va_low, vb_low);
                int16x8_t prod_high = vmulq_s16(va_high, vb_high);
                result = vcombine_s8(vqmovn_s16(prod_low), vqmovn_s16(prod_high));
                break;
            }
            case OperationType::SUB:
                result = vsubq_s8(va, vb);
                break;
        }
        vst1q_s8(&this_data[i], result);
    }

    std::size_t len8 = ((size_ - neon_len) / 8) * 8 + neon_len;
    for (std::size_t i = neon_len; i < len8; i += 8) {
        int8x8_t va = vld1_s8(&this_data[i]);
        int8x8_t vb = vld1_s8(&other_data[i]);
        int8x8_t result;
        switch (op_type) {
            case OperationType::ADD:
                result = vadd_s8(va, vb);
                break;
            case OperationType::MUL: {
                int16x8_t va16 = vmovl_s8(va);
                int16x8_t vb16 = vmovl_s8(vb);
                int16x8_t prod = vmulq_s16(va16, vb16);
                result = vqmovn_s16(prod);
                break;
            }
            case OperationType::SUB:
                result = vsub_s8(va, vb);
                break;
        }
        vst1_s8(&this_data[i], result);
    }

    std::size_t len4 = ((size_ - len8) / 4) * 4 + len8;
    for (std::size_t i = len8; i < len4; i += 4) {
        int8x8_t va = vld1_s8(&this_data[i]);
        int8x8_t vb = vld1_s8(&other_data[i]);
        int8x8_t result;
        switch (op_type) {
            case OperationType::ADD:
                result = vadd_s8(va, vb);
                break;
            case OperationType::MUL: {  
                int16x8_t va16 = vmovl_s8(va);
                int16x8_t vb16 = vmovl_s8(vb);
                int16x8_t prod = vmulq_s16(va16, vb16);
                result = vqmovn_s16(prod);
                break;
            }   
            case OperationType::SUB:
                result = vsub_s8(va, vb);
                break;  
        }
        vst1_s8(&this_data[i], result);
    }

    for (std::size_t i = neon_len; i < size_; ++i) {
        switch (op_type) {
            case OperationType::ADD: {
                int16_t sum = static_cast<int16_t>(this_data[i]) +
                              static_cast<int16_t>(other_data[i]);
                this_data[i] = static_cast<int8_t>(
                    std::max(-128, std::min(127, static_cast<int>(sum)))
                );
                break;
            }
            case OperationType::MUL: {
                int16_t prod = static_cast<int16_t>(this_data[i]) *
                               static_cast<int16_t>(other_data[i]);
                this_data[i] = static_cast<int8_t>(
                    std::max(-128, std::min(127, static_cast<int>(prod)))
                );
                break;
            }
            case OperationType::SUB: {
                int16_t diff = static_cast<int16_t>(this_data[i]) -
                               static_cast<int16_t>(other_data[i]);
                this_data[i] = static_cast<int8_t>(
                    std::max(-128, std::min(127, static_cast<int>(diff)))
                );
                break;
            }
        }
    }
}

} // namespace labneura

#endif // LABNEURA_HAVE_M1