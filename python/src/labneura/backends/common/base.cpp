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