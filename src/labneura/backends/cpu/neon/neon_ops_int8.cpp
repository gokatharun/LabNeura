#if defined(__ARM_NEON) || defined(__aarch64__)

#include "labneura/backends/cpu/neon.h"
#include "neon_internal.h"
#include "labneura/utils/alignment.hpp"
#include <algorithm>
#include <stdexcept>
#include <arm_neon.h>

namespace labneura {

void NEONBackend::operation_int8(const TensorBackend& other, OperationType op_type) {
    const int8_t* other_data = other.data_int8();
    int8_t* this_data = data_int8();

    // Hierarchical processing: 128 → 64 → 32 → 16 → 8 → scalar
    const std::size_t len128 = (size_ / 128) * 128;
    const std::size_t len64  = len128 + ((size_ - len128) / 64) * 64;
    const std::size_t len32  = len64  + ((size_ - len64) / 32) * 32;
    const std::size_t len16  = len32  + ((size_ - len32) / 16) * 16;
    const std::size_t len8   = len16  + ((size_ - len16) / 8) * 8;

    // Prefetch distance for INT8: 256 bytes (256 int8s)
    constexpr std::size_t PREFETCH_DISTANCE = 256;

    if (op_type == OperationType::ADD) {
        // Main loop: 128 int8s (8 registers × 16 int8s)
        for (std::size_t i = 0; i < len128; i += 128) {
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
            }
            const int8x16_t va0 = vld1q_s8(&this_data[i + 0]);
            const int8x16_t va1 = vld1q_s8(&this_data[i + 16]);
            const int8x16_t va2 = vld1q_s8(&this_data[i + 32]);
            const int8x16_t va3 = vld1q_s8(&this_data[i + 48]);
            const int8x16_t va4 = vld1q_s8(&this_data[i + 64]);
            const int8x16_t va5 = vld1q_s8(&this_data[i + 80]);
            const int8x16_t va6 = vld1q_s8(&this_data[i + 96]);
            const int8x16_t va7 = vld1q_s8(&this_data[i + 112]);
            
            const int8x16_t vb0 = vld1q_s8(&other_data[i + 0]);
            const int8x16_t vb1 = vld1q_s8(&other_data[i + 16]);
            const int8x16_t vb2 = vld1q_s8(&other_data[i + 32]);
            const int8x16_t vb3 = vld1q_s8(&other_data[i + 48]);
            const int8x16_t vb4 = vld1q_s8(&other_data[i + 64]);
            const int8x16_t vb5 = vld1q_s8(&other_data[i + 80]);
            const int8x16_t vb6 = vld1q_s8(&other_data[i + 96]);
            const int8x16_t vb7 = vld1q_s8(&other_data[i + 112]);
            
            vst1q_s8(&this_data[i + 0],   vqaddq_s8(va0, vb0));
            vst1q_s8(&this_data[i + 16],  vqaddq_s8(va1, vb1));
            vst1q_s8(&this_data[i + 32],  vqaddq_s8(va2, vb2));
            vst1q_s8(&this_data[i + 48],  vqaddq_s8(va3, vb3));
            vst1q_s8(&this_data[i + 64],  vqaddq_s8(va4, vb4));
            vst1q_s8(&this_data[i + 80],  vqaddq_s8(va5, vb5));
            vst1q_s8(&this_data[i + 96],  vqaddq_s8(va6, vb6));
            vst1q_s8(&this_data[i + 112], vqaddq_s8(va7, vb7));
        }
        // Tail: 64 int8s (4 registers × 16 int8s)
        if (len64 > len128) {
            const std::size_t i = len128;
            const int8x16_t va0 = vld1q_s8(&this_data[i + 0]);
            const int8x16_t va1 = vld1q_s8(&this_data[i + 16]);
            const int8x16_t va2 = vld1q_s8(&this_data[i + 32]);
            const int8x16_t va3 = vld1q_s8(&this_data[i + 48]);
            const int8x16_t vb0 = vld1q_s8(&other_data[i + 0]);
            const int8x16_t vb1 = vld1q_s8(&other_data[i + 16]);
            const int8x16_t vb2 = vld1q_s8(&other_data[i + 32]);
            const int8x16_t vb3 = vld1q_s8(&other_data[i + 48]);
            vst1q_s8(&this_data[i + 0],  vqaddq_s8(va0, vb0));
            vst1q_s8(&this_data[i + 16], vqaddq_s8(va1, vb1));
            vst1q_s8(&this_data[i + 32], vqaddq_s8(va2, vb2));
            vst1q_s8(&this_data[i + 48], vqaddq_s8(va3, vb3));
        }
        // Tail: 32 int8s (2 registers × 16 int8s)
        if (len32 > len64) {
            const std::size_t i = len64;
            const int8x16_t va0 = vld1q_s8(&this_data[i + 0]);
            const int8x16_t va1 = vld1q_s8(&this_data[i + 16]);
            const int8x16_t vb0 = vld1q_s8(&other_data[i + 0]);
            const int8x16_t vb1 = vld1q_s8(&other_data[i + 16]);
            vst1q_s8(&this_data[i + 0],  vqaddq_s8(va0, vb0));
            vst1q_s8(&this_data[i + 16], vqaddq_s8(va1, vb1));
        }
        // Tail: 16 int8s (1 register)
        if (len16 > len32) {
            const std::size_t i = len32;
            const int8x16_t va = vld1q_s8(&this_data[i]);
            const int8x16_t vb = vld1q_s8(&other_data[i]);
            vst1q_s8(&this_data[i], vqaddq_s8(va, vb));
        }
        // Tail: 8 int8s (half register)
        if (len8 > len16) {
            const std::size_t i = len16;
            const int8x8_t va = vld1_s8(&this_data[i]);
            const int8x8_t vb = vld1_s8(&other_data[i]);
            vst1_s8(&this_data[i], vqadd_s8(va, vb));
        }
        // Scalar tail
        for (std::size_t i = len8; i < size_; ++i) {
            int16_t sum = static_cast<int16_t>(this_data[i]) + static_cast<int16_t>(other_data[i]);
            this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(sum))));
        }
        return;
    }

    if (op_type == OperationType::MUL) {
        // Main loop: 128 int8s (8 registers × 16 int8s)
        for (std::size_t i = 0; i < len128; i += 128) {
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
            }
            const int8x16_t va0 = vld1q_s8(&this_data[i + 0]);
            const int8x16_t va1 = vld1q_s8(&this_data[i + 16]);
            const int8x16_t va2 = vld1q_s8(&this_data[i + 32]);
            const int8x16_t va3 = vld1q_s8(&this_data[i + 48]);
            const int8x16_t va4 = vld1q_s8(&this_data[i + 64]);
            const int8x16_t va5 = vld1q_s8(&this_data[i + 80]);
            const int8x16_t va6 = vld1q_s8(&this_data[i + 96]);
            const int8x16_t va7 = vld1q_s8(&this_data[i + 112]);
            
            const int8x16_t vb0 = vld1q_s8(&other_data[i + 0]);
            const int8x16_t vb1 = vld1q_s8(&other_data[i + 16]);
            const int8x16_t vb2 = vld1q_s8(&other_data[i + 32]);
            const int8x16_t vb3 = vld1q_s8(&other_data[i + 48]);
            const int8x16_t vb4 = vld1q_s8(&other_data[i + 64]);
            const int8x16_t vb5 = vld1q_s8(&other_data[i + 80]);
            const int8x16_t vb6 = vld1q_s8(&other_data[i + 96]);
            const int8x16_t vb7 = vld1q_s8(&other_data[i + 112]);
            
            // For multiply, we need to widen to int16, multiply, then narrow back
            // Process each 16-element vector separately
            auto mul_int8x16 = [](const int8x16_t& a, const int8x16_t& b) -> int8x16_t {
                const int16x8_t a_low = vmovl_s8(vget_low_s8(a));
                const int16x8_t a_high = vmovl_s8(vget_high_s8(a));
                const int16x8_t b_low = vmovl_s8(vget_low_s8(b));
                const int16x8_t b_high = vmovl_s8(vget_high_s8(b));
                const int16x8_t prod_low = vmulq_s16(a_low, b_low);
                const int16x8_t prod_high = vmulq_s16(a_high, b_high);
                return vcombine_s8(vqmovn_s16(prod_low), vqmovn_s16(prod_high));
            };
            
            vst1q_s8(&this_data[i + 0],   mul_int8x16(va0, vb0));
            vst1q_s8(&this_data[i + 16],  mul_int8x16(va1, vb1));
            vst1q_s8(&this_data[i + 32],  mul_int8x16(va2, vb2));
            vst1q_s8(&this_data[i + 48],  mul_int8x16(va3, vb3));
            vst1q_s8(&this_data[i + 64],  mul_int8x16(va4, vb4));
            vst1q_s8(&this_data[i + 80],  mul_int8x16(va5, vb5));
            vst1q_s8(&this_data[i + 96],  mul_int8x16(va6, vb6));
            vst1q_s8(&this_data[i + 112], mul_int8x16(va7, vb7));
        }
        // Tail: 64 int8s
        if (len64 > len128) {
            const std::size_t i = len128;
            auto mul_int8x16 = [](const int8x16_t& a, const int8x16_t& b) -> int8x16_t {
                const int16x8_t a_low = vmovl_s8(vget_low_s8(a));
                const int16x8_t a_high = vmovl_s8(vget_high_s8(a));
                const int16x8_t b_low = vmovl_s8(vget_low_s8(b));
                const int16x8_t b_high = vmovl_s8(vget_high_s8(b));
                const int16x8_t prod_low = vmulq_s16(a_low, b_low);
                const int16x8_t prod_high = vmulq_s16(a_high, b_high);
                return vcombine_s8(vqmovn_s16(prod_low), vqmovn_s16(prod_high));
            };
            const int8x16_t va0 = vld1q_s8(&this_data[i + 0]);
            const int8x16_t va1 = vld1q_s8(&this_data[i + 16]);
            const int8x16_t va2 = vld1q_s8(&this_data[i + 32]);
            const int8x16_t va3 = vld1q_s8(&this_data[i + 48]);
            const int8x16_t vb0 = vld1q_s8(&other_data[i + 0]);
            const int8x16_t vb1 = vld1q_s8(&other_data[i + 16]);
            const int8x16_t vb2 = vld1q_s8(&other_data[i + 32]);
            const int8x16_t vb3 = vld1q_s8(&other_data[i + 48]);
            vst1q_s8(&this_data[i + 0],  mul_int8x16(va0, vb0));
            vst1q_s8(&this_data[i + 16], mul_int8x16(va1, vb1));
            vst1q_s8(&this_data[i + 32], mul_int8x16(va2, vb2));
            vst1q_s8(&this_data[i + 48], mul_int8x16(va3, vb3));
        }
        // Tail: 32 int8s
        if (len32 > len64) {
            const std::size_t i = len64;
            auto mul_int8x16 = [](const int8x16_t& a, const int8x16_t& b) -> int8x16_t {
                const int16x8_t a_low = vmovl_s8(vget_low_s8(a));
                const int16x8_t a_high = vmovl_s8(vget_high_s8(a));
                const int16x8_t b_low = vmovl_s8(vget_low_s8(b));
                const int16x8_t b_high = vmovl_s8(vget_high_s8(b));
                const int16x8_t prod_low = vmulq_s16(a_low, b_low);
                const int16x8_t prod_high = vmulq_s16(a_high, b_high);
                return vcombine_s8(vqmovn_s16(prod_low), vqmovn_s16(prod_high));
            };
            const int8x16_t va0 = vld1q_s8(&this_data[i + 0]);
            const int8x16_t va1 = vld1q_s8(&this_data[i + 16]);
            const int8x16_t vb0 = vld1q_s8(&other_data[i + 0]);
            const int8x16_t vb1 = vld1q_s8(&other_data[i + 16]);
            vst1q_s8(&this_data[i + 0],  mul_int8x16(va0, vb0));
            vst1q_s8(&this_data[i + 16], mul_int8x16(va1, vb1));
        }
        // Tail: 16 int8s
        if (len16 > len32) {
            const std::size_t i = len32;
            const int8x16_t va = vld1q_s8(&this_data[i]);
            const int8x16_t vb = vld1q_s8(&other_data[i]);
            const int16x8_t va_low = vmovl_s8(vget_low_s8(va));
            const int16x8_t va_high = vmovl_s8(vget_high_s8(va));
            const int16x8_t vb_low = vmovl_s8(vget_low_s8(vb));
            const int16x8_t vb_high = vmovl_s8(vget_high_s8(vb));
            const int16x8_t prod_low = vmulq_s16(va_low, vb_low);
            const int16x8_t prod_high = vmulq_s16(va_high, vb_high);
            vst1q_s8(&this_data[i], vcombine_s8(vqmovn_s16(prod_low), vqmovn_s16(prod_high)));
        }
        // Tail: 8 int8s
        if (len8 > len16) {
            const std::size_t i = len16;
            const int8x8_t va = vld1_s8(&this_data[i]);
            const int8x8_t vb = vld1_s8(&other_data[i]);
            const int16x8_t va16 = vmovl_s8(va);
            const int16x8_t vb16 = vmovl_s8(vb);
            const int16x8_t prod = vmulq_s16(va16, vb16);
            vst1_s8(&this_data[i], vqmovn_s16(prod));
        }
        // Scalar tail
        for (std::size_t i = len8; i < size_; ++i) {
            int16_t prod = static_cast<int16_t>(this_data[i]) * static_cast<int16_t>(other_data[i]);
            this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(prod))));
        }
        return;
    }

    // DIV path for INT8 - optimized using float conversion with zero masking
    // NEON doesn't have int8 division, so we convert to float32, divide, then convert back
    if (op_type == OperationType::DIV) {
        const std::size_t len8 = (size_ / 8) * 8;

        // Process 8 int8s at a time
        for (std::size_t i = 0; i < len8; i += 8) {
            // Load 8 int8 values
            const int8x8_t va = vld1_s8(&this_data[i]);
            const int8x8_t vb = vld1_s8(&other_data[i]);

            // Mask for zeros in divisor
            const uint8x8_t zero_mask = vceq_s8(vb, vdup_n_s8(0));

            // Widen int8 → int16
            const int16x8_t va16 = vmovl_s8(va);
            const int16x8_t vb16 = vmovl_s8(vb);

            // Widen int16 → int32 (process low and high halves)
            const int32x4_t va32_low = vmovl_s16(vget_low_s16(va16));
            const int32x4_t va32_high = vmovl_s16(vget_high_s16(va16));
            const int32x4_t vb32_low = vmovl_s16(vget_low_s16(vb16));
            const int32x4_t vb32_high = vmovl_s16(vget_high_s16(vb16));

            // Convert to float32
            const float32x4_t vaf_low = vcvtq_f32_s32(va32_low);
            const float32x4_t vaf_high = vcvtq_f32_s32(va32_high);
            const float32x4_t vbf_low = vcvtq_f32_s32(vb32_low);
            const float32x4_t vbf_high = vcvtq_f32_s32(vb32_high);

            // Perform division in float32
            const float32x4_t vr_low = vdivq_f32(vaf_low, vbf_low);
            const float32x4_t vr_high = vdivq_f32(vaf_high, vbf_high);

            // Convert back to int32
            const int32x4_t vri32_low = vcvtq_s32_f32(vr_low);
            const int32x4_t vri32_high = vcvtq_s32_f32(vr_high);

            // Narrow int32 → int16 with saturation
            const int16x8_t vri16 = vcombine_s16(vqmovn_s32(vri32_low), vqmovn_s32(vri32_high));

            // Narrow int16 → int8 with saturation
            const int8x8_t result = vqmovn_s16(vri16);

            // Zero-out lanes where divisor was zero
            const int8x8_t safe_result = vbsl_s8(zero_mask, vdup_n_s8(0), result);

            vst1_s8(&this_data[i], safe_result);
        }

        // Scalar tail for remaining elements
        for (std::size_t i = len8; i < size_; ++i) {
            if (other_data[i] == 0) {
                this_data[i] = 0;  // Division by zero yields 0
            } else {
                this_data[i] = static_cast<int8_t>(this_data[i] / other_data[i]);
            }
        }
        return;
    }

    // SUB path (default)
    // Main loop: 128 int8s (8 registers × 16 int8s)
    for (std::size_t i = 0; i < len128; i += 128) {
        if (i + PREFETCH_DISTANCE < size_) {
            __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
            __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
        }
        const int8x16_t va0 = vld1q_s8(&this_data[i + 0]);
        const int8x16_t va1 = vld1q_s8(&this_data[i + 16]);
        const int8x16_t va2 = vld1q_s8(&this_data[i + 32]);
        const int8x16_t va3 = vld1q_s8(&this_data[i + 48]);
        const int8x16_t va4 = vld1q_s8(&this_data[i + 64]);
        const int8x16_t va5 = vld1q_s8(&this_data[i + 80]);
        const int8x16_t va6 = vld1q_s8(&this_data[i + 96]);
        const int8x16_t va7 = vld1q_s8(&this_data[i + 112]);
        
        const int8x16_t vb0 = vld1q_s8(&other_data[i + 0]);
        const int8x16_t vb1 = vld1q_s8(&other_data[i + 16]);
        const int8x16_t vb2 = vld1q_s8(&other_data[i + 32]);
        const int8x16_t vb3 = vld1q_s8(&other_data[i + 48]);
        const int8x16_t vb4 = vld1q_s8(&other_data[i + 64]);
        const int8x16_t vb5 = vld1q_s8(&other_data[i + 80]);
        const int8x16_t vb6 = vld1q_s8(&other_data[i + 96]);
        const int8x16_t vb7 = vld1q_s8(&other_data[i + 112]);
        
        vst1q_s8(&this_data[i + 0],   vsubq_s8(va0, vb0));
        vst1q_s8(&this_data[i + 16],  vsubq_s8(va1, vb1));
        vst1q_s8(&this_data[i + 32],  vsubq_s8(va2, vb2));
        vst1q_s8(&this_data[i + 48],  vsubq_s8(va3, vb3));
        vst1q_s8(&this_data[i + 64],  vsubq_s8(va4, vb4));
        vst1q_s8(&this_data[i + 80],  vsubq_s8(va5, vb5));
        vst1q_s8(&this_data[i + 96],  vsubq_s8(va6, vb6));
        vst1q_s8(&this_data[i + 112], vsubq_s8(va7, vb7));
    }
    // Tail: 64 int8s
    if (len64 > len128) {
        const std::size_t i = len128;
        const int8x16_t va0 = vld1q_s8(&this_data[i + 0]);
        const int8x16_t va1 = vld1q_s8(&this_data[i + 16]);
        const int8x16_t va2 = vld1q_s8(&this_data[i + 32]);
        const int8x16_t va3 = vld1q_s8(&this_data[i + 48]);
        const int8x16_t vb0 = vld1q_s8(&other_data[i + 0]);
        const int8x16_t vb1 = vld1q_s8(&other_data[i + 16]);
        const int8x16_t vb2 = vld1q_s8(&other_data[i + 32]);
        const int8x16_t vb3 = vld1q_s8(&other_data[i + 48]);
        vst1q_s8(&this_data[i + 0],  vsubq_s8(va0, vb0));
        vst1q_s8(&this_data[i + 16], vsubq_s8(va1, vb1));
        vst1q_s8(&this_data[i + 32], vsubq_s8(va2, vb2));
        vst1q_s8(&this_data[i + 48], vsubq_s8(va3, vb3));
    }
    // Tail: 32 int8s
    if (len32 > len64) {
        const std::size_t i = len64;
        const int8x16_t va0 = vld1q_s8(&this_data[i + 0]);
        const int8x16_t va1 = vld1q_s8(&this_data[i + 16]);
        const int8x16_t vb0 = vld1q_s8(&other_data[i + 0]);
        const int8x16_t vb1 = vld1q_s8(&other_data[i + 16]);
        vst1q_s8(&this_data[i + 0],  vsubq_s8(va0, vb0));
        vst1q_s8(&this_data[i + 16], vsubq_s8(va1, vb1));
    }
    // Tail: 16 int8s
    if (len16 > len32) {
        const std::size_t i = len32;
        const int8x16_t va = vld1q_s8(&this_data[i]);
        const int8x16_t vb = vld1q_s8(&other_data[i]);
        vst1q_s8(&this_data[i], vsubq_s8(va, vb));
    }
    // Tail: 8 int8s
    if (len8 > len16) {
        const std::size_t i = len16;
        const int8x8_t va = vld1_s8(&this_data[i]);
        const int8x8_t vb = vld1_s8(&other_data[i]);
        vst1_s8(&this_data[i], vsub_s8(va, vb));
    }
    // Scalar tail
    for (std::size_t i = len8; i < size_; ++i) {
        int16_t diff = static_cast<int16_t>(this_data[i]) - static_cast<int16_t>(other_data[i]);
        this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(diff))));
    }
}

// =======================
// INT16 kernels (NEON optimized)
// =======================


} // namespace labneura

#endif // __ARM_NEON || __aarch64__
