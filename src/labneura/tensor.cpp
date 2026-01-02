#include "labneura/tensor.h"
#include "labneura/backends/generic.h"
#include "labneura/backends/neon.h"
#include "labneura/backends/avx2.h"
#include "labneura/backends/backend_factory.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace labneura {

// Tensor implementation
Tensor::Tensor() : quantization_mode_(QuantizationMode::FP32) {
    init_backend(0);
}

Tensor::Tensor(const std::vector<float>& data, QuantizationMode mode)
    : quantization_mode_(mode) {
    init_backend(data.size());
    if (!data.empty()) {
        float* fp32_data = data_fp32();
        for (std::size_t i = 0; i < data.size(); ++i) {
            fp32_data[i] = data[i];
        }
    }
}

Tensor::Tensor(const std::vector<int>& data, QuantizationMode mode)
    : quantization_mode_(mode) {
    init_backend(data.size());
    if (mode == QuantizationMode::FP32) {
        float* fp32_data = data_fp32();
        for (std::size_t i = 0; i < data.size(); ++i) {
            fp32_data[i] = static_cast<float>(data[i]);
        }
    } else if (mode == QuantizationMode::INT8) {
        int8_t* int8_data = data_int8();
        for (std::size_t i = 0; i < data.size(); ++i) {
            int8_data[i] = static_cast<int8_t>(
                std::max(-128, std::min(127, data[i]))
            );
        }
    }
}

Tensor::Tensor(float scalar, QuantizationMode mode)
    : quantization_mode_(mode) {
    init_backend(1);
    data_fp32()[0] = scalar;
}

Tensor::Tensor(int scalar, QuantizationMode mode)
    : quantization_mode_(mode) {
    init_backend(1);
    if (mode == QuantizationMode::FP32) {
        data_fp32()[0] = static_cast<float>(scalar);
    } else if (mode == QuantizationMode::INT8) {
        data_int8()[0] = static_cast<int8_t>(
            std::max(-128, std::min(127, scalar))
        );
    }
}

Tensor::~Tensor() = default;

Tensor::Tensor(const Tensor& other)
    : quantization_mode_(other.quantization_mode_),
      backend_(other.backend_ ? other.backend_->clone() : nullptr) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        quantization_mode_ = other.quantization_mode_;
        backend_ = other.backend_ ? other.backend_->clone() : nullptr;
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : quantization_mode_(other.quantization_mode_),
      backend_(std::move(other.backend_)) {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        quantization_mode_ = other.quantization_mode_;
        backend_ = std::move(other.backend_);
    }
    return *this;
}

std::size_t Tensor::size() const {
    return backend_ ? backend_->size() : 0;
}

float* Tensor::data_fp32() {
    return backend_ ? backend_->data_fp32() : nullptr;
}

const float* Tensor::data_fp32() const {
    return backend_ ? backend_->data_fp32() : nullptr;
}

int8_t* Tensor::data_int8() {
    return backend_ ? backend_->data_int8() : nullptr;
}

const int8_t* Tensor::data_int8() const {
    return backend_ ? backend_->data_int8() : nullptr;
}

Tensor Tensor::add(const Tensor& other) const {
    Tensor result(*this);
    result.add_inplace(other);
    return result;
}

void Tensor::add_inplace(const Tensor& other) {
    if (!backend_) {
        throw std::runtime_error("Tensor backend not initialized");
    }
    backend_->add_inplace(*other.backend_);
}

void Tensor::mul_inplace(const Tensor& other) {
    if (!backend_) {
        throw std::runtime_error("Tensor backend not initialized");
    }
    backend_->mul_inplace(*other.backend_);
}   

void Tensor::sub_inplace(const Tensor& other) {
    if (!backend_) {
        throw std::runtime_error("Tensor backend not initialized");
    }
    backend_->sub_inplace(*other.backend_);
}

void Tensor::init_backend(std::size_t size) {
    backend_ = create_best_backend(size, quantization_mode_);
}

} // namespace labneura
