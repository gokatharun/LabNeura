#include "labneura/tensor.h"
#include "labneura/backends/cpu/generic.h"
#include "labneura/backends/cpu/neon.h"
#include "labneura/backends/cpu/avx2.h"
#include "labneura/backends/common/backend_factory.h"
#include "labneura/backends/common/dtype_validation.h"
#include "labneura/types.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <typeinfo>

namespace labneura {

namespace {
std::size_t dtype_size(QuantizationMode mode) {
    switch (mode) {
        case QuantizationMode::FP32:
            return sizeof(float);
        case QuantizationMode::FP16:
        case QuantizationMode::INT16:
            return sizeof(int16_t);
        case QuantizationMode::INT8:
            return sizeof(int8_t);
        case QuantizationMode::FP64:
            throw std::invalid_argument("Unsupported tensor dtype: FP64");
        case QuantizationMode::INT32:
            throw std::invalid_argument("Unsupported tensor dtype: INT32");
        default:
            throw std::invalid_argument("Unsupported tensor dtype: unknown");
    }
}

std::size_t checked_shape_numel(const std::vector<std::size_t>& shape) {
    if (std::find(shape.begin(), shape.end(), std::size_t{0}) != shape.end()) {
        return 0;
    }

    std::size_t element_count = 1;
    for (const std::size_t dimension : shape) {
        if (element_count >
            std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error(
                "Tensor shape element count overflows size_t");
        }
        element_count *= dimension;
    }
    return element_count;
}

void check_allocation_size(std::size_t size, QuantizationMode mode) {
    validate_tensor_dtype_supported(mode);
    const std::size_t element_bytes = dtype_size(mode);
    if (size > std::numeric_limits<std::size_t>::max() / element_bytes) {
        throw std::overflow_error("Tensor allocation byte size overflows size_t");
    }
}

std::vector<float> flatten_rows(const std::vector<std::vector<float>>& rows,
                                QuantizationMode mode) {
    validate_tensor_dtype_supported(mode);
    std::vector<float> flat;
    if (rows.empty()) return flat;
    const std::size_t cols = rows.front().size();
    const std::vector<std::size_t> shape{rows.size(), cols};
    const std::size_t element_count = checked_shape_numel(shape);
    check_allocation_size(element_count, mode);
    flat.reserve(element_count);
    for (const auto& row : rows) {
        if (row.size() != cols) {
            throw std::invalid_argument(
                "2-D Tensor rows must all have the same length");
        }
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}
}  // namespace

// Tensor implementation
Tensor::Tensor() : quantization_mode_(QuantizationMode::FP32), shape_{0} {
    init_backend(0);
}

Tensor::Tensor(QuantizationMode mode, std::unique_ptr<TensorBackend> backend, std::vector<std::size_t> shape)
    : quantization_mode_(mode), backend_(std::move(backend)), shape_(std::move(shape)) {}

Tensor::Tensor(const std::vector<std::vector<float>>& rows, QuantizationMode mode)
    : Tensor(flatten_rows(rows, mode), mode) {
    shape_ = {rows.size(), rows.empty() ? std::size_t{0} : rows.front().size()};
}

Tensor::Tensor(const std::vector<float>& data, QuantizationMode mode)
    : quantization_mode_(mode), shape_{data.size()} {
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
        } else if (mode == QuantizationMode::INT16) {
            int16_t* int16_data = data_int16();
            for (std::size_t i = 0; i < data.size(); ++i) {
                int v = static_cast<int>(std::lrintf(data[i]));
                int16_data[i] = static_cast<int16_t>(std::max(-32768, std::min(32767, v)));
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
    : quantization_mode_(mode), shape_{data.size()} {
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

Tensor Tensor::from_flat(const std::vector<float>& data,
                         const std::vector<std::size_t>& shape,
                         QuantizationMode mode) {
    const std::size_t element_count = checked_numel(shape);
    validate_allocation_size(element_count, mode);
    if (element_count != data.size()) {
        throw std::invalid_argument(
            "Tensor shape element count does not match data length");
    }

    Tensor tensor(data, mode);
    tensor.shape_ = shape;
    return tensor;
}

Tensor Tensor::from_flat(const std::vector<int>& data,
                         const std::vector<std::size_t>& shape,
                         QuantizationMode mode) {
    const std::size_t element_count = checked_numel(shape);
    validate_allocation_size(element_count, mode);
    if (element_count != data.size()) {
        throw std::invalid_argument(
            "Tensor shape element count does not match data length");
    }

    Tensor tensor(data, mode);
    tensor.shape_ = shape;
    return tensor;
}

Tensor::Tensor(float scalar, QuantizationMode mode)
    : quantization_mode_(mode), shape_{1} {
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
    : quantization_mode_(mode), shape_{1} {
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
      backend_(other.backend_ ? other.backend_->clone() : nullptr),
      shape_(other.shape_) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        quantization_mode_ = other.quantization_mode_;
        backend_ = other.backend_ ? other.backend_->clone() : nullptr;
        shape_ = other.shape_;
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : quantization_mode_(other.quantization_mode_),
      backend_(std::move(other.backend_)),
      shape_(std::move(other.shape_)) {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        quantization_mode_ = other.quantization_mode_;
        backend_ = std::move(other.backend_);
        shape_ = std::move(other.shape_);
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
    return binary_op(other, OperationType::ADD);
}

void Tensor::add_into(const Tensor& other, Tensor& out) const {
    if (quantization_mode_ != QuantizationMode::FP32 ||
        other.quantization_mode_ != QuantizationMode::FP32 ||
        out.quantization_mode_ != QuantizationMode::FP32) {
        throw std::invalid_argument("add_into requires FP32 tensors");
    }
    if (!backend_ || !other.backend_ || !out.backend_) {
        throw std::runtime_error("Tensor backend not initialized");
    }
    if (shape_ != other.shape_ || shape_ != out.shape_) {
        throw std::invalid_argument("add_into requires identical tensor shapes");
    }
    const TensorBackend& input_backend = *backend_;
    const TensorBackend& other_backend = *other.backend_;
    const TensorBackend& output_backend = *out.backend_;
    if (typeid(input_backend) != typeid(other_backend) ||
        typeid(input_backend) != typeid(output_backend)) {
        throw std::invalid_argument(
            "add_into requires all tensors to use the same backend");
    }
    if (!backend_->operation_into(*other.backend_, *out.backend_, OperationType::ADD)) {
        throw std::runtime_error(
            "FP32 ADD into an existing output is not supported by this backend");
    }
}

Tensor Tensor::mul(const Tensor& other) const {
    return binary_op(other, OperationType::MUL);
}

Tensor Tensor::sub(const Tensor& other) const {
    return binary_op(other, OperationType::SUB);
}

Tensor Tensor::div(const Tensor& other) const {
    return binary_op(other, OperationType::DIV);
}

Tensor Tensor::binary_op(const Tensor& other, OperationType op) const {
    if (backend_ && other.backend_) {
        check_same_shape(other);
        if (auto out = backend_->operation_new(*other.backend_, op)) {
            return Tensor(quantization_mode_, std::move(out), shape_);
        }
    }

    Tensor result(*this);
    switch (op) {
        case OperationType::ADD: result.add_inplace(other); break;
        case OperationType::SUB: result.sub_inplace(other); break;
        case OperationType::MUL: result.mul_inplace(other); break;
        default: result.div_inplace(other); break;
    }
    return result;
}

void Tensor::add_inplace(const Tensor& other) {
    if (!backend_) {
        throw std::runtime_error("Tensor backend not initialized");
    }
    check_same_shape(other);
    backend_->add_inplace(*other.backend_);
}

void Tensor::mul_inplace(const Tensor& other) {
    if (!backend_) {
        throw std::runtime_error("Tensor backend not initialized");
    }
    check_same_shape(other);
    backend_->mul_inplace(*other.backend_);
}   

void Tensor::sub_inplace(const Tensor& other) {
    if (!backend_) {
        throw std::runtime_error("Tensor backend not initialized");
    }
    check_same_shape(other);
    backend_->sub_inplace(*other.backend_);
}

void Tensor::div_inplace(const Tensor& other) {
    if (!backend_) {
        throw std::runtime_error("Tensor backend not initialized");
    }
    check_same_shape(other);
    backend_->div_inplace(*other.backend_);
}

void Tensor::check_same_shape(const Tensor& other) const {
    // Different element counts are reported by the backend as a size mismatch.
    if (size() == other.size() && shape_ != other.shape_) {
        throw std::runtime_error("Tensor shape mismatch");
    }
}

void Tensor::init_backend(std::size_t size) {
    validate_tensor_dtype_supported(quantization_mode_);
    validate_allocation_size(size, quantization_mode_);
    backend_ = create_best_backend(size, quantization_mode_);
}

std::size_t Tensor::checked_numel(const std::vector<std::size_t>& shape) {
    return checked_shape_numel(shape);
}

void Tensor::validate_allocation_size(std::size_t size,
                                      QuantizationMode mode) {
    check_allocation_size(size, mode);
}

} // namespace labneura
