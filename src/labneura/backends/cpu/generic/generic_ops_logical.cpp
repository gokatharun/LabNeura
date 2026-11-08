#include "labneura/backends/cpu/generic.h"

#include <stdexcept>

namespace labneura {

void GenericBackend::logical_not_into(GenericBackend& output) const {
    if (quantization_mode_ != QuantizationMode::BOOL ||
        output.quantization_mode_ != QuantizationMode::BOOL) {
        throw std::runtime_error(
            "Logical NOT requires BOOL input and output tensors");
    }
    if (size_ != output.size_) {
        throw std::runtime_error(
            "Logical NOT input and output sizes must match");
    }

    const std::uint8_t* input_values = data_bool();
    std::uint8_t* output_values = output.data_bool();
    for (std::size_t i = 0; i < size_; ++i) {
        output_values[i] =
            input_values[i] == 0 ? std::uint8_t{1} : std::uint8_t{0};
    }
}

}  // namespace labneura
