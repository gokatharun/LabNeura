#include "labneura/backends/common/base.h"
#include <stdexcept>

namespace labneura {

void TensorBackend::add_inplace(const TensorBackend& other) {
    operation(other, OperationType::ADD);
}

void TensorBackend::sub_inplace(const TensorBackend& other) {
    operation(other, OperationType::SUB);
}

void TensorBackend::mul_inplace(const TensorBackend& other) {
    operation(other, OperationType::MUL);
}

void TensorBackend::div_inplace(const TensorBackend& other) {
    operation(other, OperationType::DIV);
}

void TensorBackend::operation(const TensorBackend& other, OperationType op) {
    if (size() != other.size()) {
        throw std::runtime_error("Tensor size mismatch");
    }

    if (quantization_mode() != other.quantization_mode()) {
        throw std::runtime_error("Quantization mode mismatch");
    }

    if (storage_.get() == other.storage_.get() &&
        storage_offset_ != other.storage_offset_) {
        const std::size_t lhs_offset = storage_offset_;
        const std::size_t rhs_offset = other.storage_offset_;
        const bool overlaps =
            lhs_offset < rhs_offset
                ? rhs_offset - lhs_offset < size()
                : lhs_offset - rhs_offset < size();
        if (overlaps) {
            throw std::invalid_argument(
                "In-place operation does not support partially overlapping tensor views");
        }
    }

    if (quantization_mode() == QuantizationMode::FP32) {
        operation_fp32(other, op);
    } else if (quantization_mode() == QuantizationMode::FP16) {
        operation_fp16(other, op);
    } else if (quantization_mode() == QuantizationMode::INT16) {
        operation_int16(other, op);
    } else {
        operation_int8(other, op);
    }
}

std::unique_ptr<TensorBackend> TensorBackend::operation_new(const TensorBackend& other, OperationType op) const {
    if (size() != other.size()) {
        throw std::runtime_error("Tensor size mismatch");
    }

    if (quantization_mode() != other.quantization_mode()) {
        throw std::runtime_error("Quantization mode mismatch");
    }

    auto out = create_output();
    if (!out) {
        return nullptr;
    }
    if (!operation_into(other, *out, op)) {
        return nullptr;
    }
    return out;
}

std::unique_ptr<TensorBackend> TensorBackend::create_output() const {
    return nullptr;
}

std::unique_ptr<TensorBackend> TensorBackend::create_storage_alias(
    std::size_t, std::size_t) const {
    throw std::runtime_error(
        "Storage sharing is not implemented for this backend");
}

bool TensorBackend::operation_into(const TensorBackend&, TensorBackend&, OperationType) const {
    return false;
}

int16_t* TensorBackend::data_fp16() {
    throw std::runtime_error("FP16 data access not implemented for this backend");
}

const int16_t* TensorBackend::data_fp16() const {
    throw std::runtime_error("FP16 data access not implemented for this backend");
}

int16_t* TensorBackend::data_int16() {
    throw std::runtime_error("INT16 data access not implemented for this backend");
}

const int16_t* TensorBackend::data_int16() const {
    throw std::runtime_error("INT16 data access not implemented for this backend");
}

void TensorBackend::operation_fp16(const TensorBackend&, OperationType) {
    throw std::runtime_error("FP16 operations not implemented for this backend");
}

void TensorBackend::operation_int16(const TensorBackend&, OperationType) {
    throw std::runtime_error("INT16 operations not implemented for this backend");
}

} // namespace labneura