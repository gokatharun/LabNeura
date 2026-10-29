#if defined(__AVX2__)

#include "labneura/backends/cpu/avx2.h"
#include "labneura/types.hpp"
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace labneura {

void AVX2Backend::operation_fp16(const TensorBackend& other, OperationType op_type) {
    const int16_t* other_data = other.data_fp16();
    int16_t* this_data = data_fp16();

    for (std::size_t i = 0; i < size_; ++i) {
        uint16_t lhs_bits;
        uint16_t rhs_bits;
        std::memcpy(&lhs_bits, &this_data[i], sizeof(lhs_bits));
        std::memcpy(&rhs_bits, &other_data[i], sizeof(rhs_bits));

        const float lhs = dequantize_fp16(lhs_bits);
        const float rhs = dequantize_fp16(rhs_bits);
        float result;
        switch (op_type) {
            case OperationType::ADD:
                result = lhs + rhs;
                break;
            case OperationType::SUB:
                result = lhs - rhs;
                break;
            case OperationType::MUL:
                result = lhs * rhs;
                break;
            case OperationType::DIV:
                result = rhs == 0.0f ? 0.0f : lhs / rhs;
                break;
            default:
                throw std::invalid_argument("Unsupported FP16 operation");
        }

        const uint16_t result_bits = quantize_fp16(result);
        std::memcpy(&this_data[i], &result_bits, sizeof(result_bits));
    }
}

} // namespace labneura

#endif // __AVX2__
