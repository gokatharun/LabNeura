// Quantization and dequantization helpers for all supported types.
#pragma once

#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace labneura {

// ====================================================================
// FP64 (Double Precision) - No quantization, direct conversion
// ====================================================================
inline double quantize_fp64(double x, double /*scale*/ = 1.0, int32_t /*zero_point*/ = 0) {
    // FP64 is the highest precision type - no quantization needed
    return x;
}

inline double dequantize_fp64(double q, double /*scale*/ = 1.0, int32_t /*zero_point*/ = 0) {
    return q;
}

// ====================================================================
// FP32 (Single Precision) - Conversion from/to double
// ====================================================================
inline float quantize_fp32(double x, float /*scale*/ = 1.0f, int32_t /*zero_point*/ = 0) {
    // Clamp to FP32 range
    double clamped = std::max(
        static_cast<double>(-std::numeric_limits<float>::max()),
        std::min(static_cast<double>(std::numeric_limits<float>::max()), x)
    );
    return static_cast<float>(clamped);
}

inline double dequantize_fp32(float q, float /*scale*/ = 1.0f, int32_t /*zero_point*/ = 0) {
    return static_cast<double>(q);
}

// ====================================================================
// FP16 (Half Precision) - Stored as IEEE 754 half bits in int16_t
// ====================================================================
// Note: FP16 is a 16-bit floating point format (IEEE 754-2008)
// Layout: 1 sign bit, 5 exponent bits, 10 mantissa bits
// Range: ~±65504, smallest normalized: 2^-14 ≈ 6.1e-5

// Convert float32 to IEEE 754 half-precision bits (round-to-nearest-even)
inline uint16_t float_to_half_bits(float value) {
#if defined(__F16C__)
    // Use hardware conversion when available for correctness and speed
    constexpr int rounding = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    return static_cast<uint16_t>(_cvtss_sh(value, rounding));
#else
    uint32_t fbits;
    std::memcpy(&fbits, &value, sizeof(fbits));

    uint32_t sign = (fbits >> 16) & 0x8000u;
    int32_t  exp  = static_cast<int32_t>((fbits >> 23) & 0xFF) - 127 + 15; // unbiased to half bias
    uint32_t mant = fbits & 0x7FFFFFu;

    // NaN or Inf handling from float32
    if (((fbits >> 23) & 0xFF) == 0xFF) {
        if (mant == 0) {
            // Infinity
            return static_cast<uint16_t>(sign | 0x7C00u);
        } else {
            // NaN: preserve payload as much as possible
            uint16_t payload = static_cast<uint16_t>((mant >> 13) & 0x3FFu);
            if (payload == 0) payload = 1; // ensure it's NaN, not Inf
            return static_cast<uint16_t>(sign | 0x7C00u | payload);
        }
    }

    if (exp <= 0) {
        // Denormals or underflow to zero
        if (exp < -10) {
            return static_cast<uint16_t>(sign); // signed zero
        }
        // Subnormal half, implicit leading 1 for float32 mantissa
        mant |= 0x800000u;
        uint32_t shift = static_cast<uint32_t>(1 - exp);
        // Round-to-nearest-even: add rounding bias before shifting
        uint32_t rnd = (shift > 13) ? 0 : (1u << (shift - 1));
        uint32_t half_mant = (mant + rnd) >> shift;
        return static_cast<uint16_t>(sign | (half_mant >> 13));
    }

    if (exp >= 31) {
        // Overflow: representable as Inf in half
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    // Normalized case: round mantissa to 10 bits
    // Add rounding bias at bit 13
    mant += 0x1000u;
    if (mant & 0x00800000u) {
        // Mantissa overflow, adjust exponent
        mant = 0;
        exp += 1;
        if (exp >= 31) {
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
    }
    uint16_t half_mant = static_cast<uint16_t>(mant >> 13);
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exp) << 10) | half_mant);
#endif
}

// Convert IEEE 754 half-precision bits back to float32
inline float half_bits_to_float(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp  = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
    uint32_t mant = static_cast<uint32_t>(h) & 0x03FFu;

    uint32_t out_bits;
    if (exp == 0) {
        if (mant == 0) {
            // Zero
            out_bits = sign;
        } else {
            // Subnormal: normalize
            int e = -1;
            do {
                e += 1;
                mant <<= 1;
            } while ((mant & 0x0400u) == 0);
            mant &= 0x03FFu; // strip leading 1
            uint32_t fexp = static_cast<uint32_t>(127 - 15 - e);
            uint32_t fman = mant << 13;
            out_bits = sign | (fexp << 23) | fman;
        }
    } else if (exp == 0x1F) {
        // Inf or NaN
        uint32_t fman = mant << 13;
        out_bits = sign | 0x7F800000u | fman;
    } else {
        // Normalized
        uint32_t fexp = exp + (127 - 15);
        uint32_t fman = mant << 13;
        out_bits = sign | (fexp << 23) | fman;
    }

    float result;
    std::memcpy(&result, &out_bits, sizeof(result));
    return result;
#endif
}

inline int16_t quantize_fp16(double x, float scale = 1.0f, int32_t /*zero_point*/ = 0) {
    // Store raw IEEE-754 half bits in int16_t; no numeric clamp to INT16 range
    float scaled = static_cast<float>(x / scale);
    constexpr float FP16_MAX = 65504.0f;
    if (std::isnan(scaled)) {
        // Map NaN to a quiet NaN in half
        return static_cast<int16_t>(0x7E00u);
    }
    scaled = std::max(-FP16_MAX, std::min(FP16_MAX, scaled));
    uint16_t bits = float_to_half_bits(scaled);
    return static_cast<int16_t>(bits);
}

inline double dequantize_fp16(int16_t q, float scale = 1.0f, int32_t /*zero_point*/ = 0) {
    // Zero-point is not applicable to FP16; treat raw bits
    uint16_t bits = static_cast<uint16_t>(q);
    float value = half_bits_to_float(bits);
    return scale * static_cast<double>(value);
}

// ====================================================================
// INT32 (32-bit Signed Integer)
// ====================================================================
inline int32_t quantize_int32(double x, double scale = 1.0, int32_t zero_point = 0) {
    if (scale == 0.0) return zero_point;
    
    // Quantize with scale and zero-point
    double q_double = std::nearbyint(x / scale) + zero_point;
    
    // Clamp to INT32 range
    q_double = std::max(
        static_cast<double>(std::numeric_limits<int32_t>::min()),
        std::min(static_cast<double>(std::numeric_limits<int32_t>::max()), q_double)
    );
    
    return static_cast<int32_t>(q_double);
}

inline double dequantize_int32(int32_t q, double scale = 1.0, int32_t zero_point = 0) {
    return scale * static_cast<double>(q - zero_point);
}

// ====================================================================
// INT16 (16-bit Signed Integer)
// ====================================================================
inline int16_t quantize_int16(double x, float scale = 1.0f, int32_t zero_point = 0) {
    if (scale == 0.0f) return static_cast<int16_t>(zero_point);
    
    // Quantize with scale and zero-point
    int32_t q = static_cast<int32_t>(std::nearbyint(x / scale)) + zero_point;
    
    // Clamp to INT16 range
    q = std::max(-32768, std::min(32767, q));
    return static_cast<int16_t>(q);
}

inline double dequantize_int16(int16_t q, float scale = 1.0f, int32_t zero_point = 0) {
    return scale * static_cast<double>(q - zero_point);
}

// ====================================================================
// INT8 (8-bit Signed Integer) - Original implementation
// ====================================================================
inline int8_t quantize_int8(double x, float scale = 1.0f, int32_t zero_point = 0) {
    if (scale == 0.0f) return static_cast<int8_t>(zero_point);
    
    // Quantize with scale and zero-point
    int32_t q = static_cast<int32_t>(std::nearbyint(x / scale)) + zero_point;
    
    // Clamp to INT8 range
    q = std::max(-128, std::min(127, q));
    return static_cast<int8_t>(q);
}

inline double dequantize_int8(int8_t q, float scale = 1.0f, int32_t zero_point = 0) {
    return scale * static_cast<double>(q - zero_point);
}

// ====================================================================
// Automatic min/max scale calculation for quantization
// ====================================================================
template<typename T>
inline float calculate_scale(double min_val, double max_val) {
    // Calculate scale to map [min_val, max_val] to type T's range
    double range = max_val - min_val;
    if (range == 0.0) return 1.0f;
    
    double type_range;
    if constexpr (std::is_same_v<T, int8_t>) {
        type_range = 255.0;  // -128 to 127
    } else if constexpr (std::is_same_v<T, int16_t>) {
        type_range = 65535.0;  // -32768 to 32767
    } else if constexpr (std::is_same_v<T, int32_t>) {
        type_range = static_cast<double>(std::numeric_limits<int32_t>::max()) * 2.0;
    } else {
        type_range = 1.0;
    }
    
    return static_cast<float>(range / type_range);
}

template<typename T>
inline int32_t calculate_zero_point(double min_val, double max_val, float scale) {
    // Calculate zero-point to center the quantization range
    if (scale == 0.0f) return 0;
    
    double midpoint = (max_val + min_val) / 2.0;
    int32_t zp = static_cast<int32_t>(std::nearbyint(-midpoint / scale));
    
    // Clamp to type range
    if constexpr (std::is_same_v<T, int8_t>) {
        zp = std::max(-128, std::min(127, zp));
    } else if constexpr (std::is_same_v<T, int16_t>) {
        zp = std::max(-32768, std::min(32767, zp));
    }
    
    return zp;
}

} // namespace labneura
