#if defined(__AVX2__)

#include "labneura/backends/avx2.h"
#include <immintrin.h>
#include <algorithm>

namespace labneura {

// =======================
// Constructor
// =======================

AVX2Backend::AVX2Backend(std::size_t size, QuantizationMode mode)
    : size_(size), quantization_mode_(mode) {

    // AVX2: 256-bit registers → 8 FP32 elements
    aligned_size_ = ((size + 7) / 8) * 8;

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

    std::size_t vec_len = (size_ / 8) * 8;
    for (std::size_t i = 0; i < vec_len; i += 8) {
        __m256 va = _mm256_loadu_ps(this_data + i);
        __m256 vb = _mm256_loadu_ps(other_data + i);
        __m256 result;

        switch (op_type) {
            case OperationType::ADD:
                result = _mm256_add_ps(va, vb);
                break;
            case OperationType::MUL:
                result = _mm256_mul_ps(va, vb);
                break;
            case OperationType::SUB:
                result = _mm256_sub_ps(va, vb);
                break;
        }

        _mm256_storeu_ps(this_data + i, result);
    }

    for (std::size_t i = vec_len; i < size_; ++i) {
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

void AVX2Backend::operation_int8(const TensorBackend& other, OperationType op_type) {
    const int8_t* other_data = other.data_int8();
    int8_t* this_data = data_int8();

    std::size_t vec_len = (size_ / 32) * 32;  // AVX2 = 32 int8
    for (std::size_t i = 0; i < vec_len; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this_data + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_data + i));
        __m256i result;

        switch (op_type) {
            case OperationType::ADD:
                result = _mm256_adds_epi8(va, vb);  // saturated
                break;
            case OperationType::SUB:
                result = _mm256_subs_epi8(va, vb);  // saturated
                break;
            case OperationType::MUL: {
                // No direct int8 mul → widen to int16
                __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 0));
                __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
                __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 0));
                __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));

                __m256i prod_lo = _mm256_mullo_epi16(va_lo, vb_lo);
                __m256i prod_hi = _mm256_mullo_epi16(va_hi, vb_hi);

                result = _mm256_packs_epi16(prod_lo, prod_hi); // saturate to int8
                break;
            }
        }

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(this_data + i), result);
    }

    for (std::size_t i = vec_len; i < size_; ++i) {
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

// =======================
// Public dispatch
// =======================

} // namespace labneura

#endif // __AVX2__