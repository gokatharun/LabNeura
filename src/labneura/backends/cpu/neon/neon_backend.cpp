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

namespace {

std::shared_ptr<TensorStorage> make_neon_storage(
    std::size_t size, QuantizationMode mode) {
    return TensorStorage::allocate(
        size, util::checked_round_up(size, 8), mode, util::NEON_ALIGN);
}

}  // namespace

// =======================
// Aligned allocation helpers (use central utility)
// =======================

// =======================
// Constructor
// =======================

NEONBackend::NEONBackend(std::size_t size, QuantizationMode mode)
    : TensorBackend(make_neon_storage(size, mode)),
      size_(size),
      aligned_size_(storage_->allocation_capacity),
      quantization_mode_(mode) {
    if (mode == QuantizationMode::FP32) {
        data_fp32_ = static_cast<float*>(storage_->data());
        std::fill(data_fp32_ + size_, data_fp32_ + aligned_size_, 0.0f);
    } else if (mode == QuantizationMode::FP16) {
        data_fp16_ = static_cast<int16_t*>(storage_->data());
        std::fill(data_fp16_ + size_, data_fp16_ + aligned_size_, static_cast<int16_t>(0));
    } else if (mode == QuantizationMode::INT16) {
        data_int16_ = static_cast<int16_t*>(storage_->data());
        std::fill(data_int16_ + size_, data_int16_ + aligned_size_, static_cast<int16_t>(0));
    } else if (mode == QuantizationMode::BOOL) {
        data_bool_ = static_cast<std::uint8_t*>(storage_->data());
        std::fill(data_bool_, data_bool_ + aligned_size_, std::uint8_t{0});
    } else {
        data_int8_ = static_cast<int8_t*>(storage_->data());
        std::fill(data_int8_ + size_, data_int8_ + aligned_size_, 0);
    }
}

NEONBackend::NEONBackend(std::shared_ptr<TensorStorage> storage,
                         std::size_t logical_size,
                         std::size_t storage_offset)
    : TensorBackend(std::move(storage), logical_size, storage_offset),
      size_(logical_size),
      aligned_size_(storage_->allocation_capacity),
      quantization_mode_(storage_->dtype) {
    if (quantization_mode_ == QuantizationMode::FP32) {
        data_fp32_ = data_at_storage_offset<float>();
    } else if (quantization_mode_ == QuantizationMode::FP16) {
        data_fp16_ = data_at_storage_offset<int16_t>();
    } else if (quantization_mode_ == QuantizationMode::INT16) {
        data_int16_ = data_at_storage_offset<int16_t>();
    } else if (quantization_mode_ == QuantizationMode::BOOL) {
        data_bool_ = data_at_storage_offset<std::uint8_t>();
    } else {
        data_int8_ = data_at_storage_offset<int8_t>();
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
    return data_fp32_;
}

const float* NEONBackend::data_fp32() const {
    if (quantization_mode_ != QuantizationMode::FP32) {
        throw std::runtime_error("Tensor is not in FP32 mode");
    }
    return data_fp32_;
}

int8_t* NEONBackend::data_int8() {
    if (quantization_mode_ != QuantizationMode::INT8) {
        throw std::runtime_error("Tensor is not in INT8 mode");
    }
    return data_int8_;
}

const int8_t* NEONBackend::data_int8() const {
    if (quantization_mode_ != QuantizationMode::INT8) {
        throw std::runtime_error("Tensor is not in INT8 mode");
    }
    return data_int8_;
}

int16_t* NEONBackend::data_fp16() {
    if (quantization_mode_ != QuantizationMode::FP16) {
        throw std::runtime_error("Tensor is not in FP16 mode");
    }
    return data_fp16_;
}

const int16_t* NEONBackend::data_fp16() const {
    if (quantization_mode_ != QuantizationMode::FP16) {
        throw std::runtime_error("Tensor is not in FP16 mode");
    }
    return data_fp16_;
}

int16_t* NEONBackend::data_int16() {
    if (quantization_mode_ != QuantizationMode::INT16) {
        throw std::runtime_error("Tensor is not in INT16 mode");
    }
    return data_int16_;
}

const int16_t* NEONBackend::data_int16() const {
    if (quantization_mode_ != QuantizationMode::INT16) {
        throw std::runtime_error("Tensor is not in INT16 mode");
    }
    return data_int16_;
}

std::uint8_t* NEONBackend::data_bool() {
    if (quantization_mode_ != QuantizationMode::BOOL) {
        throw std::runtime_error("Tensor is not in BOOL mode");
    }
    return canonicalize_bool_data(data_bool_);
}

const std::uint8_t* NEONBackend::data_bool() const {
    if (quantization_mode_ != QuantizationMode::BOOL) {
        throw std::runtime_error("Tensor is not in BOOL mode");
    }
    return validate_bool_data(data_bool_);
}

// =======================
// Clone
// =======================

std::unique_ptr<TensorBackend> NEONBackend::clone() const {
    auto backend = std::make_unique<NEONBackend>(size_, quantization_mode_);
    if (quantization_mode_ == QuantizationMode::FP32) {
        std::copy(data_fp32_, data_fp32_ + size_, backend->data_fp32());
    } else if (quantization_mode_ == QuantizationMode::FP16) {
        std::copy(data_fp16_, data_fp16_ + size_, backend->data_fp16());
    } else if (quantization_mode_ == QuantizationMode::INT16) {
        std::copy(data_int16_, data_int16_ + size_, backend->data_int16());
    } else if (quantization_mode_ == QuantizationMode::BOOL) {
        std::copy(data_bool_, data_bool_ + size_, backend->data_bool());
    } else {
        std::copy(data_int8_, data_int8_ + size_, backend->data_int8());
    }
    return backend;
}

std::unique_ptr<TensorBackend> NEONBackend::create_storage_alias(
    std::size_t logical_size, std::size_t storage_offset) const {
    return std::unique_ptr<TensorBackend>(
        new NEONBackend(storage_, logical_size, storage_offset));
}

// =======================
// FP32 kernels
// =======================


} // namespace labneura

#endif // __ARM_NEON || __aarch64__
