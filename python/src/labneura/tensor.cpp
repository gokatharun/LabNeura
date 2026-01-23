#include "labneura/tensor.h"
#include "labneura/backends/cpu/generic.h"
#include "labneura/backends/cpu/neon.h"
#include "labneura/backends/cpu/avx2.h"
#include "labneura/backends/common/backend_factory.h"
#include "labneura/types.hpp"
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
        if (mode == QuantizationMode::FP32) {
            float* fp32_data = data_fp32();
            for (std::size_t i = 0; i < data.size(); ++i) {
                fp32_data[i] = data[i];
            }
        } else if (mode == QuantizationMode::FP16) {
            int16_t* fp16_data = data_fp16();
            for (std::size_t i = 0; i < data.size(); ++i) {
                // Proper FP16 encoding (stored in int16_t for NEON compatibility)
                uint16_t encoded = quantize_fp16(data[i]);
                std::memcpy(&fp16_data[i], &encoded, sizeof(uint16_t));
            }
        } else if (mode == QuantizationMode::INT8) {
            int8_t* int8_data = data_int8();
            for (std::size_t i = 0; i < data.size(); ++i) {
                int v = static_cast<int>(std::lrintf(data[i]));
                int8_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, v)));
            }
        } else {
            // Other modes: default to FP32
            float* fp32_data = data_fp32();
            for (std::size_t i = 0; i < data.size(); ++i) {
                fp32_data[i] = data[i];
            }
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
    } else if (mode == QuantizationMode::FP16) {
        int16_t* fp16_data = data_fp16();
        for (std::size_t i = 0; i < data.size(); ++i) {
            // Proper FP16 encoding from int (convert to float first, then quantize)
            uint16_t encoded = quantize_fp16(static_cast<float>(data[i]));
            std::memcpy(&fp16_data[i], &encoded, sizeof(uint16_t));
        }
    } else if (mode == QuantizationMode::INT8) {
        int8_t* int8_data = data_int8();
        for (std::size_t i = 0; i < data.size(); ++i) {
            int8_data[i] = static_cast<int8_t>(
                std::max(-128, std::min(127, data[i]))
            );
        }
    } else if (mode == QuantizationMode::INT16) {
        int16_t* int16_data = data_int16();
        for (std::size_t i = 0; i < data.size(); ++i) {
            int16_data[i] = static_cast<int16_t>(
                std::max(-32768, std::min(32767, data[i]))
            );
        }
    }
}

Tensor::Tensor(float scalar, QuantizationMode mode)
    : quantization_mode_(mode) {
    init_backend(1);
    if (mode == QuantizationMode::FP32) {
        data_fp32()[0] = scalar;
    } else if (mode == QuantizationMode::FP16) {
        uint16_t encoded = quantize_fp16(scalar);
        std::memcpy(&data_fp16()[0], &encoded, sizeof(uint16_t));
    } else if (mode == QuantizationMode::INT8) {
        int v = static_cast<int>(std::lrintf(scalar));
        data_int8()[0] = static_cast<int8_t>(std::max(-128, std::min(127, v)));
    } else if (mode == QuantizationMode::INT16) {
        int v = static_cast<int>(std::lrintf(scalar));
        data_int16()[0] = static_cast<int16_t>(std::max(-32768, std::min(32767, v)));
    } else {
        data_fp32()[0] = scalar;
    }
}

Tensor::Tensor(int scalar, QuantizationMode mode)
    : quantization_mode_(mode) {
    init_backend(1);
    if (mode == QuantizationMode::FP32) {
        data_fp32()[0] = static_cast<float>(scalar);
    } else if (mode == QuantizationMode::FP16) {
        uint16_t encoded = quantize_fp16(static_cast<float>(scalar));
        std::memcpy(&data_fp16()[0], &encoded, sizeof(uint16_t));
    } else if (mode == QuantizationMode::INT8) {
        data_int8()[0] = static_cast<int8_t>(
            std::max(-128, std::min(127, scalar))
        );
    } else if (mode == QuantizationMode::INT16) {
        data_int16()[0] = static_cast<int16_t>(
            std::max(-32768, std::min(32767, scalar))
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

int16_t* Tensor::data_fp16() {
    return backend_ ? backend_->data_fp16() : nullptr;
}

const int16_t* Tensor::data_fp16() const {
    return backend_ ? backend_->data_fp16() : nullptr;
}

int16_t* Tensor::data_int16() {
    return backend_ ? backend_->data_int16() : nullptr;
}

const int16_t* Tensor::data_int16() const {
    return backend_ ? backend_->data_int16() : nullptr;
}

Tensor Tensor::add(const Tensor& other) const {
    Tensor result(*this);
    result.add_inplace(other);
    return result;
}

Tensor Tensor::mul(const Tensor& other) const {
    Tensor result(*this);
    result.mul_inplace(other);
    return result;
}

Tensor Tensor::sub(const Tensor& other) const {
    Tensor result(*this);
    result.sub_inplace(other);
    return result;
}

Tensor Tensor::div(const Tensor& other) const {
    Tensor result(*this);
    result.div_inplace(other);
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

void Tensor::div_inplace(const Tensor& other) {
    if (!backend_) {
        throw std::runtime_error("Tensor backend not initialized");
    }
    backend_->div_inplace(*other.backend_);
}

void Tensor::init_backend(std::size_t size) {
    backend_ = create_best_backend(size, quantization_mode_);
}

} // namespace labneura
