// Simple, tiny header to expose a couple of numeric helpers.
#pragma once

#include <cstdint>
#include <cmath>

namespace labneura {

// Quantize a single float into signed 8-bit using scale and zero-point.
inline int8_t quantize_int8(float x, float scale, int32_t zero_point = 0) {
    if (scale == 0.0f) return static_cast<int8_t>(zero_point);
    int32_t q = static_cast<int32_t>(std::nearbyint(x / scale)) + zero_point;
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    return static_cast<int8_t>(q);
}

// Dequantize signed 8-bit back to float.
inline float dequantize_int8(int8_t q, float scale, int32_t zero_point = 0) {
    return scale * (static_cast<int32_t>(q) - zero_point);
}

} // namespace labneura
