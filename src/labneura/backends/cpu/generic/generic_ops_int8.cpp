#include "labneura/backends/cpu/generic.h"
#include <algorithm>
#include <stdexcept>

namespace labneura {

void GenericBackend::operation_int8(const TensorBackend& other, OperationType op_type) {
    const int8_t* other_data = other.data_int8();
    int8_t* this_data = data_int8();

    if (op_type == OperationType::ADD) {
        for (std::size_t i = 0; i < size_; ++i) {
            int16_t sum = static_cast<int16_t>(this_data[i]) + static_cast<int16_t>(other_data[i]);
            this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(sum))));
        }
        return;
    }
    if (op_type == OperationType::MUL) {
        for (std::size_t i = 0; i < size_; ++i) {
            int16_t prod = static_cast<int16_t>(this_data[i]) * static_cast<int16_t>(other_data[i]);
            this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(prod))));
        }
        return;
    }
    if (op_type == OperationType::DIV) {
        for (std::size_t i = 0; i < size_; ++i) {
            if (other_data[i] == 0) {
                this_data[i] = 0;  // Division by zero yields 0
            } else {
                int16_t quot = static_cast<int16_t>(this_data[i]) / static_cast<int16_t>(other_data[i]);
                this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(quot))));
            }
        }
        return;
    }
    for (std::size_t i = 0; i < size_; ++i) {
        int16_t diff = static_cast<int16_t>(this_data[i]) - static_cast<int16_t>(other_data[i]);
        this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(diff))));
    }
}

}// namespace labneura


