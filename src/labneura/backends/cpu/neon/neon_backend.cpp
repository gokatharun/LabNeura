#if defined(__ARM_NEON)

#include "labneura/backends/cpu/neon.h"
#include "labneura/types.hpp"
#include <arm_neon.h>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <stdexcept>
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


} // namespace labneura

#endif // __ARM_NEON || __aarch64__
