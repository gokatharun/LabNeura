#include "labneura/types.hpp"
#include <iostream>

int main() {
    float original_value = 1.23f;
    float scale = 0.1f;
    int32_t zero_point = 0;

    // Quantize the original float value
    int8_t quantized_value = labneura::quantize_int8(original_value, scale, zero_point);
    std::cout << "Quantized Value: " << static_cast<int>(quantized_value) << std::endl;

    // Dequantize back to float
    float dequantized_value = labneura::dequantize_int8(quantized_value, scale, zero_point);
    std::cout << "Dequantized Value: " << dequantized_value << std::endl;

    return 0;
}