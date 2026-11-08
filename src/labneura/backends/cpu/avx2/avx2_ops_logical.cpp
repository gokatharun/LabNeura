#if defined(__AVX2__)

#include "labneura/backends/cpu/avx2.h"

#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace labneura {

void AVX2Backend::logical_not_into(AVX2Backend& output) const {
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
    const __m256i zero = _mm256_setzero_si256();
    const __m256i one = _mm256_set1_epi8(1);
    std::size_t index = 0;
    for (; size_ - index >= 32; index += 32) {
        const __m256i input = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(input_values + index));
        const __m256i is_false = _mm256_cmpeq_epi8(input, zero);
        const __m256i result = _mm256_and_si256(is_false, one);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(output_values + index), result);
    }
    for (; index < size_; ++index) {
        output_values[index] =
            input_values[index] == 0 ? std::uint8_t{1} : std::uint8_t{0};
    }
}

}  // namespace labneura

#endif
