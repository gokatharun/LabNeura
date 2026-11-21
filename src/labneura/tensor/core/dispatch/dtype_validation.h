#pragma once

#include "labneura/tensor.h"
#include <stdexcept>

namespace labneura {

inline void validate_tensor_dtype_supported(QuantizationMode mode) {
    switch (mode) {
        case QuantizationMode::FP32:
        case QuantizationMode::FP16:
        case QuantizationMode::INT16:
        case QuantizationMode::INT8:
        case QuantizationMode::BOOL:
            return;
        case QuantizationMode::FP64:
            throw std::invalid_argument("Unsupported tensor dtype: FP64");
        case QuantizationMode::INT32:
            throw std::invalid_argument("Unsupported tensor dtype: INT32");
        default:
            throw std::invalid_argument("Unsupported tensor dtype: unknown");
    }
}

}  // namespace labneura
