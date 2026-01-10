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

// Quantize a single float into signed 16-bit using scale and zero-point.
inline int16_t quantize_int16(float x, float scale, int32_t zero_point = 0) {
    if (scale == 0.0f) return static_cast<int16_t>(zero_point);
    int32_t q = static_cast<int32_t>(std::nearbyint(x / scale)) + zero_point;
    if (q > 32767) q = 32767;
    if (q < -32768) q = -32768;
    return static_cast<int16_t>(q);
}

// Dequantize signed 16-bit back to float.
inline float dequantize_int16(int16_t q, float scale, int32_t zero_point = 0) {
    return scale * (static_cast<int32_t>(q) - zero_point);
}

// Convert float32 to float16 (IEEE half-precision)
inline uint16_t quantize_fp16(float x) {
    // Use bit manipulation for FP32 to FP16 conversion
    union {
        float f;
        uint32_t ui;
    } f32 = {x};
    
    uint32_t f = f32.ui;
    uint16_t result;
    
    // Extract components from FP32
    uint32_t sign = (f >> 31) & 0x1;
    uint32_t exponent = (f >> 23) & 0xFF;
    uint32_t mantissa = f & 0x7FFFFF;
    
    // Handle special cases
    if (exponent == 0xFF) {
        // Infinity or NaN in FP32
        if (mantissa == 0) {
            // Infinity: exponent=31, mantissa=0 in FP16
            result = (sign << 15) | (31 << 10);
        } else {
            // NaN: exponent=31, mantissa!=0 in FP16
            result = (sign << 15) | (31 << 10) | 0x200;
        }
    } else if (exponent == 0) {
        // Denormalized or zero in FP32
        result = (sign << 15);
    } else {
        // Normal number: bias adjustment
        // FP32: exponent - bias(127) = exponent - 127
        // FP16: new_exponent - bias(15) = result
        // So: new_exponent = exponent - 127 + 15
        int32_t unbiased_exp = static_cast<int32_t>(exponent) - 127;
        int32_t new_exponent = unbiased_exp + 15;
        
        if (new_exponent >= 31) {
            // Overflow to infinity
            result = (sign << 15) | (31 << 10);
        } else if (new_exponent <= 0) {
            // Underflow to zero
            result = (sign << 15);
        } else {
            // Normal: combine sign, exponent (5 bits), mantissa (10 bits)
            // Round to nearest, ties to even
            uint32_t round_bit = (mantissa >> 12) & 0x1;  // Bit that will be rounded off
            uint32_t sticky_bits = mantissa & 0xFFF;      // All lower bits
            uint16_t new_mantissa = (mantissa >> 13) & 0x3FF;
            
            // Round up if: round_bit is 1 AND (sticky_bits != 0 OR new_mantissa is odd)
            if (round_bit && (sticky_bits || (new_mantissa & 1))) {
                new_mantissa++;
                // Check for mantissa overflow
                if (new_mantissa > 0x3FF) {
                    new_mantissa = 0;
                    new_exponent++;
                    if (new_exponent >= 31) {
                        // Overflow to infinity
                        result = (sign << 15) | (31 << 10);
                        return result;
                    }
                }
            }
            
            result = (sign << 15) | ((new_exponent & 0x1F) << 10) | new_mantissa;
        }
    }
    
    return result;
}

// Convert float16 (IEEE half-precision) to float32
inline float dequantize_fp16(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    
    uint32_t result;
    
    if (exponent == 0x1F) {
        // Infinity or NaN
        if (mantissa == 0) {
            // Infinity
            result = (sign << 31) | 0x7F800000;
        } else {
            // NaN
            result = (sign << 31) | 0x7FC00000;
        }
    } else if (exponent == 0) {
        // Zero (denormalized numbers are treated as zero for simplicity)
        result = (sign << 31);
    } else {
        // Normal number: bias adjustment from 15 to 127
        uint32_t new_exponent = exponent + (127 - 15);
        uint32_t new_mantissa = mantissa << 13;
        result = (sign << 31) | ((new_exponent & 0xFF) << 23) | new_mantissa;
    }
    
    union {
        uint32_t ui;
        float f;
    } f32 = {result};
    
    return f32.f;
}

} // namespace labneura
