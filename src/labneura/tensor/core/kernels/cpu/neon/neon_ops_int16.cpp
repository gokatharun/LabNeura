#if defined(__ARM_NEON) || defined(__aarch64__)

#include "labneura/backends/cpu/neon.h"
#include "neon_internal.h"
#include "labneura/utils/alignment.hpp"
#include <algorithm>
#include <stdexcept>
#include <arm_neon.h>

namespace labneura {

void NEONBackend::operation_int16(const TensorBackend& other, OperationType op_type) {
    const int16_t* other_data = other.data_int16();
    int16_t* this_data = data_int16();

    // Prefetch distance for INT16: 128 int16s (~256 bytes)
    constexpr std::size_t PREFETCH_DISTANCE = 128;

    // Hierarchical processing: 64 → 32 → 16 → 8 → 4 → scalar
    const std::size_t len64 = (size_ / 64) * 64;
    const std::size_t len32 = len64 + ((size_ - len64) / 32) * 32;
    const std::size_t len16 = len32 + ((size_ - len32) / 16) * 16;
    const std::size_t len8  = len16 + ((size_ - len16) / 8) * 8;
    const std::size_t len4  = len8  + ((size_ - len8) / 4) * 4;

    if (op_type == OperationType::ADD) {
        // Main loop: 64 int16s (8 registers × 8 int16s)
        for (std::size_t i = 0; i < len64; i += 64) {
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
            }
            const int16x8_t va0 = vld1q_s16(&this_data[i + 0]);
            const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
            const int16x8_t va2 = vld1q_s16(&this_data[i + 16]);
            const int16x8_t va3 = vld1q_s16(&this_data[i + 24]);
            const int16x8_t va4 = vld1q_s16(&this_data[i + 32]);
            const int16x8_t va5 = vld1q_s16(&this_data[i + 40]);
            const int16x8_t va6 = vld1q_s16(&this_data[i + 48]);
            const int16x8_t va7 = vld1q_s16(&this_data[i + 56]);

            const int16x8_t vb0 = vld1q_s16(&other_data[i + 0]);
            const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);
            const int16x8_t vb2 = vld1q_s16(&other_data[i + 16]);
            const int16x8_t vb3 = vld1q_s16(&other_data[i + 24]);
            const int16x8_t vb4 = vld1q_s16(&other_data[i + 32]);
            const int16x8_t vb5 = vld1q_s16(&other_data[i + 40]);
            const int16x8_t vb6 = vld1q_s16(&other_data[i + 48]);
            const int16x8_t vb7 = vld1q_s16(&other_data[i + 56]);

            vst1q_s16(&this_data[i + 0],  vqaddq_s16(va0, vb0));
            vst1q_s16(&this_data[i + 8],  vqaddq_s16(va1, vb1));
            vst1q_s16(&this_data[i + 16], vqaddq_s16(va2, vb2));
            vst1q_s16(&this_data[i + 24], vqaddq_s16(va3, vb3));
            vst1q_s16(&this_data[i + 32], vqaddq_s16(va4, vb4));
            vst1q_s16(&this_data[i + 40], vqaddq_s16(va5, vb5));
            vst1q_s16(&this_data[i + 48], vqaddq_s16(va6, vb6));
            vst1q_s16(&this_data[i + 56], vqaddq_s16(va7, vb7));
        }
        // Tail: 32 int16s (4 registers × 8 int16s)
        if (len32 > len64) {
            const std::size_t i = len64;
            const int16x8_t va0 = vld1q_s16(&this_data[i + 0]);
            const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
            const int16x8_t va2 = vld1q_s16(&this_data[i + 16]);
            const int16x8_t va3 = vld1q_s16(&this_data[i + 24]);
            const int16x8_t vb0 = vld1q_s16(&other_data[i + 0]);
            const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);
            const int16x8_t vb2 = vld1q_s16(&other_data[i + 16]);
            const int16x8_t vb3 = vld1q_s16(&other_data[i + 24]);
            vst1q_s16(&this_data[i + 0],  vqaddq_s16(va0, vb0));
            vst1q_s16(&this_data[i + 8],  vqaddq_s16(va1, vb1));
            vst1q_s16(&this_data[i + 16], vqaddq_s16(va2, vb2));
            vst1q_s16(&this_data[i + 24], vqaddq_s16(va3, vb3));
        }
        // Tail: 16 int16s (2 registers × 8 int16s)
        if (len16 > len32) {
            const std::size_t i = len32;
            const int16x8_t va0 = vld1q_s16(&this_data[i + 0]);
            const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
            const int16x8_t vb0 = vld1q_s16(&other_data[i + 0]);
            const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);
            vst1q_s16(&this_data[i + 0], vqaddq_s16(va0, vb0));
            vst1q_s16(&this_data[i + 8], vqaddq_s16(va1, vb1));
        }
        // Tail: 8 int16s (1 register)
        if (len8 > len16) {
            const std::size_t i = len16;
            const int16x8_t va = vld1q_s16(&this_data[i]);
            const int16x8_t vb = vld1q_s16(&other_data[i]);
            vst1q_s16(&this_data[i], vqaddq_s16(va, vb));
        }
        // Tail: 4 int16s (half register)
        if (len4 > len8) {
            const std::size_t i = len8;
            const int16x4_t va = vld1_s16(&this_data[i]);
            const int16x4_t vb = vld1_s16(&other_data[i]);
            vst1_s16(&this_data[i], vqadd_s16(va, vb));
        }
        // Scalar tail
        for (std::size_t i = len4; i < size_; ++i) {
            int32_t sum = static_cast<int32_t>(this_data[i]) + static_cast<int32_t>(other_data[i]);
            this_data[i] = static_cast<int16_t>(std::max(-32768, std::min(32767, sum)));
        }
        return;
    }

    if (op_type == OperationType::MUL) {
        // Helper lambda for int16 multiplication with saturation
        auto mul_int16x8_sat = [](const int16x8_t& a, const int16x8_t& b) -> int16x8_t {
            // Split into low and high halves
            const int16x4_t a_low = vget_low_s16(a);
            const int16x4_t a_high = vget_high_s16(a);
            const int16x4_t b_low = vget_low_s16(b);
            const int16x4_t b_high = vget_high_s16(b);
            
            // Widen to int32 and multiply
            const int32x4_t prod_low = vmull_s16(a_low, b_low);
            const int32x4_t prod_high = vmull_s16(a_high, b_high);
            
            // Narrow back to int16 with saturation
            return vcombine_s16(vqmovn_s32(prod_low), vqmovn_s32(prod_high));
        };

        // Main loop: 64 int16s (8 registers × 8 int16s)
        for (std::size_t i = 0; i < len64; i += 64) {
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
            }
            const int16x8_t va0 = vld1q_s16(&this_data[i + 0]);
            const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
            const int16x8_t va2 = vld1q_s16(&this_data[i + 16]);
            const int16x8_t va3 = vld1q_s16(&this_data[i + 24]);
            const int16x8_t va4 = vld1q_s16(&this_data[i + 32]);
            const int16x8_t va5 = vld1q_s16(&this_data[i + 40]);
            const int16x8_t va6 = vld1q_s16(&this_data[i + 48]);
            const int16x8_t va7 = vld1q_s16(&this_data[i + 56]);

            const int16x8_t vb0 = vld1q_s16(&other_data[i + 0]);
            const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);
            const int16x8_t vb2 = vld1q_s16(&other_data[i + 16]);
            const int16x8_t vb3 = vld1q_s16(&other_data[i + 24]);
            const int16x8_t vb4 = vld1q_s16(&other_data[i + 32]);
            const int16x8_t vb5 = vld1q_s16(&other_data[i + 40]);
            const int16x8_t vb6 = vld1q_s16(&other_data[i + 48]);
            const int16x8_t vb7 = vld1q_s16(&other_data[i + 56]);

            vst1q_s16(&this_data[i + 0],  mul_int16x8_sat(va0, vb0));
            vst1q_s16(&this_data[i + 8],  mul_int16x8_sat(va1, vb1));
            vst1q_s16(&this_data[i + 16], mul_int16x8_sat(va2, vb2));
            vst1q_s16(&this_data[i + 24], mul_int16x8_sat(va3, vb3));
            vst1q_s16(&this_data[i + 32], mul_int16x8_sat(va4, vb4));
            vst1q_s16(&this_data[i + 40], mul_int16x8_sat(va5, vb5));
            vst1q_s16(&this_data[i + 48], mul_int16x8_sat(va6, vb6));
            vst1q_s16(&this_data[i + 56], mul_int16x8_sat(va7, vb7));
        }
        // Tail: 32 int16s
        if (len32 > len64) {
            const std::size_t i = len64;
            const int16x8_t va0 = vld1q_s16(&this_data[i + 0]);
            const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
            const int16x8_t va2 = vld1q_s16(&this_data[i + 16]);
            const int16x8_t va3 = vld1q_s16(&this_data[i + 24]);
            const int16x8_t vb0 = vld1q_s16(&other_data[i + 0]);
            const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);
            const int16x8_t vb2 = vld1q_s16(&other_data[i + 16]);
            const int16x8_t vb3 = vld1q_s16(&other_data[i + 24]);
            vst1q_s16(&this_data[i + 0],  mul_int16x8_sat(va0, vb0));
            vst1q_s16(&this_data[i + 8],  mul_int16x8_sat(va1, vb1));
            vst1q_s16(&this_data[i + 16], mul_int16x8_sat(va2, vb2));
            vst1q_s16(&this_data[i + 24], mul_int16x8_sat(va3, vb3));
        }
        // Tail: 16 int16s
        if (len16 > len32) {
            const std::size_t i = len32;
            const int16x8_t va0 = vld1q_s16(&this_data[i + 0]);
            const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
            const int16x8_t vb0 = vld1q_s16(&other_data[i + 0]);
            const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);
            vst1q_s16(&this_data[i + 0], mul_int16x8_sat(va0, vb0));
            vst1q_s16(&this_data[i + 8], mul_int16x8_sat(va1, vb1));
        }
        // Tail: 8 int16s
        if (len8 > len16) {
            const std::size_t i = len16;
            const int16x8_t va = vld1q_s16(&this_data[i]);
            const int16x8_t vb = vld1q_s16(&other_data[i]);
            vst1q_s16(&this_data[i], mul_int16x8_sat(va, vb));
        }
        // Tail: 4 int16s
        if (len4 > len8) {
            const std::size_t i = len8;
            const int16x4_t va = vld1_s16(&this_data[i]);
            const int16x4_t vb = vld1_s16(&other_data[i]);
            const int32x4_t prod = vmull_s16(va, vb);
            vst1_s16(&this_data[i], vqmovn_s32(prod));
        }
        // Scalar tail
        for (std::size_t i = len4; i < size_; ++i) {
            int32_t product = static_cast<int32_t>(this_data[i]) * static_cast<int32_t>(other_data[i]);
            this_data[i] = static_cast<int16_t>(std::max(-32768, std::min(32767, product)));
        }
        return;
    }

    // DIV path for INT16 - optimized using float conversion with zero masking
    // NEON doesn't have int16 division, so we convert to float32, divide, then convert back
    if (op_type == OperationType::DIV) {
        const std::size_t len16 = (size_ / 16) * 16;
        const std::size_t len8 = (size_ / 8) * 8;
        const std::size_t len4 = (size_ / 4) * 4;

        // Process 16 int16s at a time (two sets of 8)
        for (std::size_t i = 0; i < len16; i += 16) {
            // First 8 int16s
            const int16x8_t va0 = vld1q_s16(&this_data[i]);
            const int16x8_t vb0 = vld1q_s16(&other_data[i]);

            const uint16x8_t zero_mask0 = vceqq_s16(vb0, vdupq_n_s16(0));

            const int32x4_t va32_0_low = vmovl_s16(vget_low_s16(va0));
            const int32x4_t va32_0_high = vmovl_s16(vget_high_s16(va0));
            const int32x4_t vb32_0_low = vmovl_s16(vget_low_s16(vb0));
            const int32x4_t vb32_0_high = vmovl_s16(vget_high_s16(vb0));

            const float32x4_t vr0_low = vdivq_f32(vcvtq_f32_s32(va32_0_low), vcvtq_f32_s32(vb32_0_low));
            const float32x4_t vr0_high = vdivq_f32(vcvtq_f32_s32(va32_0_high), vcvtq_f32_s32(vb32_0_high));

            const int16x8_t result0 = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(vr0_low)), vqmovn_s32(vcvtq_s32_f32(vr0_high)));
            const int16x8_t safe0 = vbslq_s16(zero_mask0, vdupq_n_s16(0), result0);
            vst1q_s16(&this_data[i], safe0);

            // Second 8 int16s
            const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
            const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);

            const uint16x8_t zero_mask1 = vceqq_s16(vb1, vdupq_n_s16(0));

            const int32x4_t va32_1_low = vmovl_s16(vget_low_s16(va1));
            const int32x4_t va32_1_high = vmovl_s16(vget_high_s16(va1));
            const int32x4_t vb32_1_low = vmovl_s16(vget_low_s16(vb1));
            const int32x4_t vb32_1_high = vmovl_s16(vget_high_s16(vb1));

            const float32x4_t vr1_low = vdivq_f32(vcvtq_f32_s32(va32_1_low), vcvtq_f32_s32(vb32_1_low));
            const float32x4_t vr1_high = vdivq_f32(vcvtq_f32_s32(va32_1_high), vcvtq_f32_s32(vb32_1_high));

            const int16x8_t result1 = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(vr1_low)), vqmovn_s32(vcvtq_s32_f32(vr1_high)));
            const int16x8_t safe1 = vbslq_s16(zero_mask1, vdupq_n_s16(0), result1);
            vst1q_s16(&this_data[i + 8], safe1);
        }

        // Tail: 8 int16s
        if (len8 > len16) {
            const std::size_t i = len16;
            const int16x8_t va = vld1q_s16(&this_data[i]);
            const int16x8_t vb = vld1q_s16(&other_data[i]);

            const uint16x8_t zero_mask = vceqq_s16(vb, vdupq_n_s16(0));

            const int32x4_t va32_low = vmovl_s16(vget_low_s16(va));
            const int32x4_t va32_high = vmovl_s16(vget_high_s16(va));
            const int32x4_t vb32_low = vmovl_s16(vget_low_s16(vb));
            const int32x4_t vb32_high = vmovl_s16(vget_high_s16(vb));

            const float32x4_t vr_low = vdivq_f32(vcvtq_f32_s32(va32_low), vcvtq_f32_s32(vb32_low));
            const float32x4_t vr_high = vdivq_f32(vcvtq_f32_s32(va32_high), vcvtq_f32_s32(vb32_high));

            const int16x8_t result = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(vr_low)), vqmovn_s32(vcvtq_s32_f32(vr_high)));
            const int16x8_t safe = vbslq_s16(zero_mask, vdupq_n_s16(0), result);
            vst1q_s16(&this_data[i], safe);
        }

        // Tail: 4 int16s
        if (len4 > len8) {
            const std::size_t i = len8;
            const int16x4_t va = vld1_s16(&this_data[i]);
            const int16x4_t vb = vld1_s16(&other_data[i]);

            const uint16x4_t zero_mask = vceq_s16(vb, vdup_n_s16(0));

            const int32x4_t va32 = vmovl_s16(va);
            const int32x4_t vb32 = vmovl_s16(vb);

            const float32x4_t vr = vdivq_f32(vcvtq_f32_s32(va32), vcvtq_f32_s32(vb32));
            const int16x4_t result = vqmovn_s32(vcvtq_s32_f32(vr));
            const int16x4_t safe = vbsl_s16(zero_mask, vdup_n_s16(0), result);

            vst1_s16(&this_data[i], safe);
        }

        // Scalar tail for remaining elements
        for (std::size_t i = len4; i < size_; ++i) {
            if (other_data[i] == 0) {
                this_data[i] = 0;  // Division by zero yields 0
            } else {
                this_data[i] = static_cast<int16_t>(this_data[i] / other_data[i]);
            }
        }
        return;
    }

    // SUB path (default)
    for (std::size_t i = 0; i < len64; i += 64) {
        if (i + PREFETCH_DISTANCE < size_) {
            __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
            __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
        }
        const int16x8_t va0 = vld1q_s16(&this_data[i + 0]);
        const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
        const int16x8_t va2 = vld1q_s16(&this_data[i + 16]);
        const int16x8_t va3 = vld1q_s16(&this_data[i + 24]);
        const int16x8_t va4 = vld1q_s16(&this_data[i + 32]);
        const int16x8_t va5 = vld1q_s16(&this_data[i + 40]);
        const int16x8_t va6 = vld1q_s16(&this_data[i + 48]);
        const int16x8_t va7 = vld1q_s16(&this_data[i + 56]);

        const int16x8_t vb0 = vld1q_s16(&other_data[i + 0]);
        const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);
        const int16x8_t vb2 = vld1q_s16(&other_data[i + 16]);
        const int16x8_t vb3 = vld1q_s16(&other_data[i + 24]);
        const int16x8_t vb4 = vld1q_s16(&other_data[i + 32]);
        const int16x8_t vb5 = vld1q_s16(&other_data[i + 40]);
        const int16x8_t vb6 = vld1q_s16(&other_data[i + 48]);
        const int16x8_t vb7 = vld1q_s16(&other_data[i + 56]);

        vst1q_s16(&this_data[i + 0],  vqsubq_s16(va0, vb0));
        vst1q_s16(&this_data[i + 8],  vqsubq_s16(va1, vb1));
        vst1q_s16(&this_data[i + 16], vqsubq_s16(va2, vb2));
        vst1q_s16(&this_data[i + 24], vqsubq_s16(va3, vb3));
        vst1q_s16(&this_data[i + 32], vqsubq_s16(va4, vb4));
        vst1q_s16(&this_data[i + 40], vqsubq_s16(va5, vb5));
        vst1q_s16(&this_data[i + 48], vqsubq_s16(va6, vb6));
        vst1q_s16(&this_data[i + 56], vqsubq_s16(va7, vb7));
    }
    // Tail: 32 int16s
    if (len32 > len64) {
        const std::size_t i = len64;
        const int16x8_t va0 = vld1q_s16(&this_data[i + 0]);
        const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
        const int16x8_t va2 = vld1q_s16(&this_data[i + 16]);
        const int16x8_t va3 = vld1q_s16(&this_data[i + 24]);
        const int16x8_t vb0 = vld1q_s16(&other_data[i + 0]);
        const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);
        const int16x8_t vb2 = vld1q_s16(&other_data[i + 16]);
        const int16x8_t vb3 = vld1q_s16(&other_data[i + 24]);
        vst1q_s16(&this_data[i + 0],  vqsubq_s16(va0, vb0));
        vst1q_s16(&this_data[i + 8],  vqsubq_s16(va1, vb1));
        vst1q_s16(&this_data[i + 16], vqsubq_s16(va2, vb2));
        vst1q_s16(&this_data[i + 24], vqsubq_s16(va3, vb3));
    }
    // Tail: 16 int16s
    if (len16 > len32) {
        const std::size_t i = len32;
        const int16x8_t va0 = vld1q_s16(&this_data[i + 0]);
        const int16x8_t va1 = vld1q_s16(&this_data[i + 8]);
        const int16x8_t vb0 = vld1q_s16(&other_data[i + 0]);
        const int16x8_t vb1 = vld1q_s16(&other_data[i + 8]);
        vst1q_s16(&this_data[i + 0], vqsubq_s16(va0, vb0));
        vst1q_s16(&this_data[i + 8], vqsubq_s16(va1, vb1));
    }
    // Tail: 8 int16s
    if (len8 > len16) {
        const std::size_t i = len16;
        const int16x8_t va = vld1q_s16(&this_data[i]);
        const int16x8_t vb = vld1q_s16(&other_data[i]);
        vst1q_s16(&this_data[i], vqsubq_s16(va, vb));
    }
    // Tail: 4 int16s
    if (len4 > len8) {
        const std::size_t i = len8;
        const int16x4_t va = vld1_s16(&this_data[i]);
        const int16x4_t vb = vld1_s16(&other_data[i]);
        vst1_s16(&this_data[i], vqsub_s16(va, vb));
    }
    // Scalar tail
    for (std::size_t i = len4; i < size_; ++i) {
        int32_t diff = static_cast<int32_t>(this_data[i]) - static_cast<int32_t>(other_data[i]);
        this_data[i] = static_cast<int16_t>(std::max(-32768, std::min(32767, diff)));
    }
}

} // namespace labneura

#endif // LABNEURA_HAVE_M1