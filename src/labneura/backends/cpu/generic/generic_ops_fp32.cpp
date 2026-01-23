#include "labneura/backends/cpu/generic.h"
#include <algorithm>

namespace labneura {

void GenericBackend::operation_fp32(const TensorBackend& other, OperationType op_type) {
    const float* other_data = other.data_fp32();
    float* this_data = data_fp32();

    if (op_type == OperationType::ADD) {
        for (std::size_t i = 0; i < size_; ++i) this_data[i] += other_data[i];
        return;
    }
    if (op_type == OperationType::MUL) {
        for (std::size_t i = 0; i < size_; ++i) this_data[i] *= other_data[i];
        return;
    }
    if (op_type == OperationType::DIV) {
        for (std::size_t i = 0; i < size_; ++i) this_data[i] /= other_data[i];
        return;
    }
    for (std::size_t i = 0; i < size_; ++i) this_data[i] -= other_data[i];
}

} // namespace labneura
