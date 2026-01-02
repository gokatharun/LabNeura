#if defined(__SSE4_1__)

#include "labneura/backends/sse41.h"
#include <smmintrin.h>
#include <algorithm>

namespace labneura {

// =======================
// Constructor
// =======================

SSE41Backend::SSE41Backend(std::size_t size, QuantizationMode mode)
    : size_(size), quantization_mode_(mode) {

    // SSE = 128-bit → 4 FP32
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

    std::size_t vec_end = (size_ / 4) * 4;
    for (std::size_t i = 0; i < vec_end; i += 4) {
        __m128 a = _mm_loadu_ps(d + i);
        __m128 b = _mm_loadu_ps(o + i);
        __m128 r;

        switch (op) {
            case OperationType::ADD: r = _mm_add_ps(a, b); break;
            case OperationType::MUL: r = _mm_mul_ps(a, b); break;
            case OperationType::SUB: r = _mm_sub_ps(a, b); break;
        }

        _mm_storeu_ps(d + i, r);
    }

    for (std::size_t i = vec_end; i < size_; ++i) {
        switch (op) {
            case OperationType::ADD: d[i] += o[i]; break;
            case OperationType::MUL: d[i] *= o[i]; break;
            case OperationType::SUB: d[i] -= o[i]; break;
        }
    }
}

// =======================
// INT8 kernels
// =======================

void SSE41Backend::operation_int8(const TensorBackend& other, OperationType op) {
    const int8_t* o = other.data_int8();
    int8_t* d = data_int8();

    std::size_t vec_end = (size_ / 16) * 16;
    for (std::size_t i = 0; i < vec_end; i += 16) {
        __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(d + i));
        __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o + i));
        __m128i r;

        switch (op) {
            case OperationType::ADD:
                r = _mm_adds_epi8(a, b);
                break;
            case OperationType::SUB:
                r = _mm_subs_epi8(a, b);
                break;
            case OperationType::MUL: {
                __m128i a_lo = _mm_cvtepi8_epi16(a);
                __m128i b_lo = _mm_cvtepi8_epi16(b);
                __m128i prod = _mm_mullo_epi16(a_lo, b_lo);
                r = _mm_packs_epi16(prod, prod);
                break;
            }
        }

        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + i), r);
    }

    for (std::size_t i = vec_end; i < size_; ++i) {
        switch (op) {
            case OperationType::ADD: {
                int16_t s = d[i] + o[i];
                d[i] = std::max(-128, std::min(127, int(s)));
                break;
            }
            case OperationType::MUL: {
                int16_t p = d[i] * o[i];
                d[i] = std::max(-128, std::min(127, int(p)));
                break;
            }
            case OperationType::SUB: {
                int16_t diff = d[i] - o[i];
                d[i] = std::max(-128, std::min(127, int(diff)));
                break;
            }
        }
    }
}

// =======================
// Public dispatch
// =======================

} // namespace labneura

#endif // __SSE4_1__