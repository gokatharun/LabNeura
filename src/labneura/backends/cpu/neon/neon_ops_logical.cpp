#if defined(__ARM_NEON) || defined(__aarch64__)

#include "labneura/backends/cpu/neon.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace labneura {

void NEONBackend::logical_not_into(NEONBackend& output) const {
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
    const uint8x16_t zero = vdupq_n_u8(0);
    const uint8x16_t one = vdupq_n_u8(1);
    std::size_t index = 0;
    for (; size_ - index >= 16; index += 16) {
        const uint8x16_t input = vld1q_u8(input_values + index);
        const uint8x16_t is_false = vceqq_u8(input, zero);
        vst1q_u8(output_values + index, vandq_u8(is_false, one));
    }
    for (; index < size_; ++index) {
        output_values[index] =
            input_values[index] == 0 ? std::uint8_t{1} : std::uint8_t{0};
    }
}

}  // namespace labneura

#endif
