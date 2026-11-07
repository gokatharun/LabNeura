#include "labneura/backends/cpu/generic.h"
#include <memory>
#include <algorithm>
#include <stdexcept>
#include "labneura/utils/alignment.hpp"

namespace labneura {

GenericBackend::GenericBackend(std::size_t size, QuantizationMode mode)
    : TensorBackend(TensorStorage::allocate(
          size, size, mode, util::DEFAULT_ALIGN)),
      size_(size),
      quantization_mode_(mode) {
    if (mode == QuantizationMode::FP32) {
        data_fp32_ = static_cast<float*>(storage_->data());
    } else if (mode == QuantizationMode::FP16) {
        data_fp16_ = static_cast<int16_t*>(storage_->data());
    } else if (mode == QuantizationMode::INT8) {
        data_int8_ = static_cast<int8_t*>(storage_->data());
    } else if (mode == QuantizationMode::INT16) {
        data_int16_ = static_cast<int16_t*>(storage_->data());
    }
}

GenericBackend::GenericBackend(std::shared_ptr<TensorStorage> storage,
                               std::size_t logical_size,
                               std::size_t storage_offset)
    : TensorBackend(std::move(storage), logical_size, storage_offset),
      size_(logical_size),
      quantization_mode_(storage_->dtype) {
    if (quantization_mode_ == QuantizationMode::FP32) {
        data_fp32_ = data_at_storage_offset<float>();
    } else if (quantization_mode_ == QuantizationMode::FP16) {
        data_fp16_ = data_at_storage_offset<int16_t>();
    } else if (quantization_mode_ == QuantizationMode::INT8) {
        data_int8_ = data_at_storage_offset<int8_t>();
    } else if (quantization_mode_ == QuantizationMode::INT16) {
        data_int16_ = data_at_storage_offset<int16_t>();
    }
}

std::size_t GenericBackend::size() const {
    return size_;
}

QuantizationMode GenericBackend::quantization_mode() const {
    return quantization_mode_;
}

float* GenericBackend::data_fp32() {
    if (quantization_mode_ != QuantizationMode::FP32) {
        throw std::runtime_error("Tensor is not in FP32 mode");
    }
    return data_fp32_;
}

const float* GenericBackend::data_fp32() const {
    if (quantization_mode_ != QuantizationMode::FP32) {
        throw std::runtime_error("Tensor is not in FP32 mode");
    }
    return data_fp32_;
}

int8_t* GenericBackend::data_int8() {
    if (quantization_mode_ != QuantizationMode::INT8) {
        throw std::runtime_error("Tensor is not in INT8 mode");
    }
    return data_int8_;
}

const int8_t* GenericBackend::data_int8() const {
    if (quantization_mode_ != QuantizationMode::INT8) {
        throw std::runtime_error("Tensor is not in INT8 mode");
    }
    return data_int8_;
}

int16_t* GenericBackend::data_fp16() {
    if (quantization_mode_ != QuantizationMode::FP16) {
        throw std::runtime_error("Tensor is not in FP16 mode");
    }
    return data_fp16_;
}

const int16_t* GenericBackend::data_fp16() const {
    if (quantization_mode_ != QuantizationMode::FP16) {
        throw std::runtime_error("Tensor is not in FP16 mode");
    }
    return data_fp16_;
}

int16_t* GenericBackend::data_int16() {
    if (quantization_mode_ != QuantizationMode::INT16) {
        throw std::runtime_error("Tensor is not in INT16 mode");
    }
    return data_int16_;
}

const int16_t* GenericBackend::data_int16() const {
    if (quantization_mode_ != QuantizationMode::INT16) {
        throw std::runtime_error("Tensor is not in INT16 mode");
    }
    return data_int16_;
}

std::unique_ptr<TensorBackend> GenericBackend::create_storage_alias(
    std::size_t logical_size, std::size_t storage_offset) const {
    return std::unique_ptr<TensorBackend>(
        new GenericBackend(storage_, logical_size, storage_offset));
}

std::unique_ptr<TensorBackend> GenericBackend::clone() const {
    auto backend = std::make_unique<GenericBackend>(size_, quantization_mode_);
    if (quantization_mode_ == QuantizationMode::FP32) {
        std::copy(data_fp32_, data_fp32_ + size_,
                 backend->data_fp32());
    } else if (quantization_mode_ == QuantizationMode::FP16) {
        std::copy(data_fp16_, data_fp16_ + size_,
                 backend->data_fp16());
    } else if (quantization_mode_ == QuantizationMode::INT8) {
        std::copy(data_int8_, data_int8_ + size_,
                 backend->data_int8());
    } else if (quantization_mode_ == QuantizationMode::INT16) {
        std::copy(data_int16_, data_int16_ + size_,
                 backend->data_int16());
    }
    return backend;
}


} // namespace labneura
