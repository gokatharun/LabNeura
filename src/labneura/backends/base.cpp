#include "labneura/backends/base.h"
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

void TensorBackend::operation(const TensorBackend& other, OperationType op) {
    if (size() != other.size()) {
        throw std::runtime_error("Tensor size mismatch");
    }

    if (quantization_mode() != other.quantization_mode()) {
        throw std::runtime_error("Quantization mode mismatch");
    }

    if (quantization_mode() == QuantizationMode::FP32) {
        operation_fp32(other, op);
    } else {
        operation_int8(other, op);
    }
}

} // namespace labneura