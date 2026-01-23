#if defined(__ARM_NEON) || defined(__aarch64__)

#include "labneura/backends/cpu/neon.h"
#include "neon_internal.h"
#include "labneura/utils/alignment.hpp"
#include "labneura/types.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <stdexcept>

namespace labneura {

void NEONBackend::operation_fp32(const TensorBackend& other, OperationType op_type) {
    const float* other_data = other.data_fp32();
    float* this_data = data_fp32();

    // Prefetch distance for ARM NEON: 256 bytes (64 floats) ahead
    // M1 has larger caches but benefits from prefetch at 100K-1M sizes
    constexpr std::size_t PREFETCH_DISTANCE = 64;  // 64 floats = 256 bytes

    // Hierarchical processing: 32 → 16 → 8 → 4 → scalar
    const std::size_t len32 = (size_ / 32) * 32;
    const std::size_t len16 = len32 + ((size_ - len32) / 16) * 16;
    const std::size_t len8  = len16 + ((size_ - len16) / 8) * 8;
    const std::size_t len4  = len8  + ((size_ - len8) / 4) * 4;

    if (op_type == OperationType::ADD) {
        // Main loop: 32 floats (8 registers × 4 floats)
        for (std::size_t i = 0; i < len32; i += 32) {
            // Prefetch next iteration's data (ARM intrinsic)
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);  // write, temporal
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1); // read, temporal
            }
            const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
            const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            const float32x4_t va2 = vld1q_f32(&this_data[i + 8]);
            const float32x4_t va3 = vld1q_f32(&this_data[i + 12]);
            const float32x4_t va4 = vld1q_f32(&this_data[i + 16]);
            const float32x4_t va5 = vld1q_f32(&this_data[i + 20]);
            const float32x4_t va6 = vld1q_f32(&this_data[i + 24]);
            const float32x4_t va7 = vld1q_f32(&this_data[i + 28]);
            
            const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
            const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            const float32x4_t vb2 = vld1q_f32(&other_data[i + 8]);
            const float32x4_t vb3 = vld1q_f32(&other_data[i + 12]);
            const float32x4_t vb4 = vld1q_f32(&other_data[i + 16]);
            const float32x4_t vb5 = vld1q_f32(&other_data[i + 20]);
            const float32x4_t vb6 = vld1q_f32(&other_data[i + 24]);
            const float32x4_t vb7 = vld1q_f32(&other_data[i + 28]);
            
            vst1q_f32(&this_data[i + 0],  vaddq_f32(va0, vb0));
            vst1q_f32(&this_data[i + 4],  vaddq_f32(va1, vb1));
            vst1q_f32(&this_data[i + 8],  vaddq_f32(va2, vb2));
            vst1q_f32(&this_data[i + 12], vaddq_f32(va3, vb3));
            vst1q_f32(&this_data[i + 16], vaddq_f32(va4, vb4));
            vst1q_f32(&this_data[i + 20], vaddq_f32(va5, vb5));
            vst1q_f32(&this_data[i + 24], vaddq_f32(va6, vb6));
            vst1q_f32(&this_data[i + 28], vaddq_f32(va7, vb7));
        }
        // Tail: 16 floats (4 registers × 4 floats)
        if (len16 > len32) {
            const std::size_t i = len32;
            const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
            const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            const float32x4_t va2 = vld1q_f32(&this_data[i + 8]);
            const float32x4_t va3 = vld1q_f32(&this_data[i + 12]);
            const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
            const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            const float32x4_t vb2 = vld1q_f32(&other_data[i + 8]);
            const float32x4_t vb3 = vld1q_f32(&other_data[i + 12]);
            vst1q_f32(&this_data[i + 0],  vaddq_f32(va0, vb0));
            vst1q_f32(&this_data[i + 4],  vaddq_f32(va1, vb1));
            vst1q_f32(&this_data[i + 8],  vaddq_f32(va2, vb2));
            vst1q_f32(&this_data[i + 12], vaddq_f32(va3, vb3));
        }
        // Tail: 8 floats (2 registers × 4 floats)
        if (len8 > len16) {
            const std::size_t i = len16;
            const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
            const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
            const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            vst1q_f32(&this_data[i + 0], vaddq_f32(va0, vb0));
            vst1q_f32(&this_data[i + 4], vaddq_f32(va1, vb1));
        }
        // Tail: 4 floats (1 register)
        if (len4 > len8) {
            const std::size_t i = len8;
            const float32x4_t va = vld1q_f32(&this_data[i]);
            const float32x4_t vb = vld1q_f32(&other_data[i]);
            vst1q_f32(&this_data[i], vaddq_f32(va, vb));
        }
        // Scalar tail: remaining 1-3 elements
        for (std::size_t i = len4; i < size_; ++i) {
            this_data[i] += other_data[i];
        }
        return;
    }

    if (op_type == OperationType::MUL) {
        // Main loop: 32 floats
        for (std::size_t i = 0; i < len32; i += 32) {
            // Prefetch for multiply operation
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
            }
            const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
            const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            const float32x4_t va2 = vld1q_f32(&this_data[i + 8]);
            const float32x4_t va3 = vld1q_f32(&this_data[i + 12]);
            const float32x4_t va4 = vld1q_f32(&this_data[i + 16]);
            const float32x4_t va5 = vld1q_f32(&this_data[i + 20]);
            const float32x4_t va6 = vld1q_f32(&this_data[i + 24]);
            const float32x4_t va7 = vld1q_f32(&this_data[i + 28]);
            
            const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
            const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            const float32x4_t vb2 = vld1q_f32(&other_data[i + 8]);
            const float32x4_t vb3 = vld1q_f32(&other_data[i + 12]);
            const float32x4_t vb4 = vld1q_f32(&other_data[i + 16]);
            const float32x4_t vb5 = vld1q_f32(&other_data[i + 20]);
            const float32x4_t vb6 = vld1q_f32(&other_data[i + 24]);
            const float32x4_t vb7 = vld1q_f32(&other_data[i + 28]);
            
            vst1q_f32(&this_data[i + 0],  vmulq_f32(va0, vb0));
            vst1q_f32(&this_data[i + 4],  vmulq_f32(va1, vb1));
            vst1q_f32(&this_data[i + 8],  vmulq_f32(va2, vb2));
            vst1q_f32(&this_data[i + 12], vmulq_f32(va3, vb3));
            vst1q_f32(&this_data[i + 16], vmulq_f32(va4, vb4));
            vst1q_f32(&this_data[i + 20], vmulq_f32(va5, vb5));
            vst1q_f32(&this_data[i + 24], vmulq_f32(va6, vb6));
            vst1q_f32(&this_data[i + 28], vmulq_f32(va7, vb7));
        }
        // Tail: 16 floats
        if (len16 > len32) {
            const std::size_t i = len32;
            const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
            const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            const float32x4_t va2 = vld1q_f32(&this_data[i + 8]);
            const float32x4_t va3 = vld1q_f32(&this_data[i + 12]);
            const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
            const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            const float32x4_t vb2 = vld1q_f32(&other_data[i + 8]);
            const float32x4_t vb3 = vld1q_f32(&other_data[i + 12]);
            vst1q_f32(&this_data[i + 0],  vmulq_f32(va0, vb0));
            vst1q_f32(&this_data[i + 4],  vmulq_f32(va1, vb1));
            vst1q_f32(&this_data[i + 8],  vmulq_f32(va2, vb2));
            vst1q_f32(&this_data[i + 12], vmulq_f32(va3, vb3));
        }
        // Tail: 8 floats
        if (len8 > len16) {
            const std::size_t i = len16;
            const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
            const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
            const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            vst1q_f32(&this_data[i + 0], vmulq_f32(va0, vb0));
            vst1q_f32(&this_data[i + 4], vmulq_f32(va1, vb1));
        }
        // Tail: 4 floats
        if (len4 > len8) {
            const std::size_t i = len8;
            const float32x4_t va = vld1q_f32(&this_data[i]);
            const float32x4_t vb = vld1q_f32(&other_data[i]);
            vst1q_f32(&this_data[i], vmulq_f32(va, vb));
        }
        // Scalar tail
        for (std::size_t i = len4; i < size_; ++i) {
            this_data[i] *= other_data[i];
        }
        return;
    }

    // DIV path for FP32
    if (op_type == OperationType::DIV) {
        // Main loop: 32 floats (8 registers × 4 floats)
        for (std::size_t i = 0; i < len32; i += 32) {
            // Prefetch for division operation
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
            }
            
            const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
            const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            const float32x4_t va2 = vld1q_f32(&this_data[i + 8]);
            const float32x4_t va3 = vld1q_f32(&this_data[i + 12]);
            const float32x4_t va4 = vld1q_f32(&this_data[i + 16]);
            const float32x4_t va5 = vld1q_f32(&this_data[i + 20]);
            const float32x4_t va6 = vld1q_f32(&this_data[i + 24]);
            const float32x4_t va7 = vld1q_f32(&this_data[i + 28]);
            
            const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
            const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            const float32x4_t vb2 = vld1q_f32(&other_data[i + 8]);
            const float32x4_t vb3 = vld1q_f32(&other_data[i + 12]);
            const float32x4_t vb4 = vld1q_f32(&other_data[i + 16]);
            const float32x4_t vb5 = vld1q_f32(&other_data[i + 20]);
            const float32x4_t vb6 = vld1q_f32(&other_data[i + 24]);
            const float32x4_t vb7 = vld1q_f32(&other_data[i + 28]);
            
            // NEON division using vdivq_f32 (ARMv8 and later)
            #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__)
                vst1q_f32(&this_data[i + 0],  vdivq_f32(va0, vb0));
                vst1q_f32(&this_data[i + 4],  vdivq_f32(va1, vb1));
                vst1q_f32(&this_data[i + 8],  vdivq_f32(va2, vb2));
                vst1q_f32(&this_data[i + 12], vdivq_f32(va3, vb3));
                vst1q_f32(&this_data[i + 16], vdivq_f32(va4, vb4));
                vst1q_f32(&this_data[i + 20], vdivq_f32(va5, vb5));
                vst1q_f32(&this_data[i + 24], vdivq_f32(va6, vb6));
                vst1q_f32(&this_data[i + 28], vdivq_f32(va7, vb7));
            #else
                // Fallback using reciprocal estimate + Newton-Raphson (ARMv7 NEON)
                float32x4_t r0 = vrecpeq_f32(vb0);
                float32x4_t r1 = vrecpeq_f32(vb1);
                float32x4_t r2 = vrecpeq_f32(vb2);
                float32x4_t r3 = vrecpeq_f32(vb3);
                float32x4_t r4 = vrecpeq_f32(vb4);
                float32x4_t r5 = vrecpeq_f32(vb5);
                float32x4_t r6 = vrecpeq_f32(vb6);
                float32x4_t r7 = vrecpeq_f32(vb7);
                
                // Newton-Raphson iteration: r = r * (2 - b * r)
                r0 = vmulq_f32(r0, vrecpsq_f32(vb0, r0));
                r1 = vmulq_f32(r1, vrecpsq_f32(vb1, r1));
                r2 = vmulq_f32(r2, vrecpsq_f32(vb2, r2));
                r3 = vmulq_f32(r3, vrecpsq_f32(vb3, r3));
                r4 = vmulq_f32(r4, vrecpsq_f32(vb4, r4));
                r5 = vmulq_f32(r5, vrecpsq_f32(vb5, r5));
                r6 = vmulq_f32(r6, vrecpsq_f32(vb6, r6));
                r7 = vmulq_f32(r7, vrecpsq_f32(vb7, r7));
                
                // Second iteration for better precision
                r0 = vmulq_f32(r0, vrecpsq_f32(vb0, r0));
                r1 = vmulq_f32(r1, vrecpsq_f32(vb1, r1));
                r2 = vmulq_f32(r2, vrecpsq_f32(vb2, r2));
                r3 = vmulq_f32(r3, vrecpsq_f32(vb3, r3));
                r4 = vmulq_f32(r4, vrecpsq_f32(vb4, r4));
                r5 = vmulq_f32(r5, vrecpsq_f32(vb5, r5));
                r6 = vmulq_f32(r6, vrecpsq_f32(vb6, r6));
                r7 = vmulq_f32(r7, vrecpsq_f32(vb7, r7));
                
                vst1q_f32(&this_data[i + 0],  vmulq_f32(va0, r0));
                vst1q_f32(&this_data[i + 4],  vmulq_f32(va1, r1));
                vst1q_f32(&this_data[i + 8],  vmulq_f32(va2, r2));
                vst1q_f32(&this_data[i + 12], vmulq_f32(va3, r3));
                vst1q_f32(&this_data[i + 16], vmulq_f32(va4, r4));
                vst1q_f32(&this_data[i + 20], vmulq_f32(va5, r5));
                vst1q_f32(&this_data[i + 24], vmulq_f32(va6, r6));
                vst1q_f32(&this_data[i + 28], vmulq_f32(va7, r7));
            #endif
        }
        
        // Tail: 16 floats (4 registers × 4 floats)
        if (len16 > len32) {
            const std::size_t i = len32;
            const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
            const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            const float32x4_t va2 = vld1q_f32(&this_data[i + 8]);
            const float32x4_t va3 = vld1q_f32(&this_data[i + 12]);
            const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
            const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            const float32x4_t vb2 = vld1q_f32(&other_data[i + 8]);
            const float32x4_t vb3 = vld1q_f32(&other_data[i + 12]);
            
            #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__)
                vst1q_f32(&this_data[i + 0],  vdivq_f32(va0, vb0));
                vst1q_f32(&this_data[i + 4],  vdivq_f32(va1, vb1));
                vst1q_f32(&this_data[i + 8],  vdivq_f32(va2, vb2));
                vst1q_f32(&this_data[i + 12], vdivq_f32(va3, vb3));
            #else
                float32x4_t r0 = vrecpeq_f32(vb0);
                float32x4_t r1 = vrecpeq_f32(vb1);
                float32x4_t r2 = vrecpeq_f32(vb2);
                float32x4_t r3 = vrecpeq_f32(vb3);
                r0 = vmulq_f32(r0, vrecpsq_f32(vb0, r0));
                r1 = vmulq_f32(r1, vrecpsq_f32(vb1, r1));
                r2 = vmulq_f32(r2, vrecpsq_f32(vb2, r2));
                r3 = vmulq_f32(r3, vrecpsq_f32(vb3, r3));
                r0 = vmulq_f32(r0, vrecpsq_f32(vb0, r0));
                r1 = vmulq_f32(r1, vrecpsq_f32(vb1, r1));
                r2 = vmulq_f32(r2, vrecpsq_f32(vb2, r2));
                r3 = vmulq_f32(r3, vrecpsq_f32(vb3, r3));
                vst1q_f32(&this_data[i + 0],  vmulq_f32(va0, r0));
                vst1q_f32(&this_data[i + 4],  vmulq_f32(va1, r1));
                vst1q_f32(&this_data[i + 8],  vmulq_f32(va2, r2));
                vst1q_f32(&this_data[i + 12], vmulq_f32(va3, r3));
            #endif
        }
        
        // Tail: 8 floats (2 registers × 4 floats)
        if (len8 > len16) {
            const std::size_t i = len16;
            const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
            const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
            const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            
            #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__)
                vst1q_f32(&this_data[i + 0], vdivq_f32(va0, vb0));
                vst1q_f32(&this_data[i + 4], vdivq_f32(va1, vb1));
            #else
                float32x4_t r0 = vrecpeq_f32(vb0);
                float32x4_t r1 = vrecpeq_f32(vb1);
                r0 = vmulq_f32(r0, vrecpsq_f32(vb0, r0));
                r1 = vmulq_f32(r1, vrecpsq_f32(vb1, r1));
                r0 = vmulq_f32(r0, vrecpsq_f32(vb0, r0));
                r1 = vmulq_f32(r1, vrecpsq_f32(vb1, r1));
                vst1q_f32(&this_data[i + 0], vmulq_f32(va0, r0));
                vst1q_f32(&this_data[i + 4], vmulq_f32(va1, r1));
            #endif
        }
        
        // Tail: 4 floats (1 register)
        if (len4 > len8) {
            const std::size_t i = len8;
            const float32x4_t va = vld1q_f32(&this_data[i]);
            const float32x4_t vb = vld1q_f32(&other_data[i]);
            
            #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__)
                vst1q_f32(&this_data[i], vdivq_f32(va, vb));
            #else
                float32x4_t r = vrecpeq_f32(vb);
                r = vmulq_f32(r, vrecpsq_f32(vb, r));
                r = vmulq_f32(r, vrecpsq_f32(vb, r));
                vst1q_f32(&this_data[i], vmulq_f32(va, r));
            #endif
        }
        
        // Scalar tail: remaining 1-3 elements
        for (std::size_t i = len4; i < size_; ++i) {
            this_data[i] /= other_data[i];
        }
        return;
    }

    // SUB path (default)
    for (std::size_t i = 0; i < len32; i += 32) {
        // Prefetch for subtract operation
        if (i + PREFETCH_DISTANCE < size_) {
            __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
            __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
        }
        
        const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
        const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
        const float32x4_t va2 = vld1q_f32(&this_data[i + 8]);
        const float32x4_t va3 = vld1q_f32(&this_data[i + 12]);
        const float32x4_t va4 = vld1q_f32(&this_data[i + 16]);
        const float32x4_t va5 = vld1q_f32(&this_data[i + 20]);
        const float32x4_t va6 = vld1q_f32(&this_data[i + 24]);
        const float32x4_t va7 = vld1q_f32(&this_data[i + 28]);
        
        const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
        const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
        const float32x4_t vb2 = vld1q_f32(&other_data[i + 8]);
        const float32x4_t vb3 = vld1q_f32(&other_data[i + 12]);
        const float32x4_t vb4 = vld1q_f32(&other_data[i + 16]);
        const float32x4_t vb5 = vld1q_f32(&other_data[i + 20]);
        const float32x4_t vb6 = vld1q_f32(&other_data[i + 24]);
        const float32x4_t vb7 = vld1q_f32(&other_data[i + 28]);
        
        vst1q_f32(&this_data[i + 0],  vsubq_f32(va0, vb0));
        vst1q_f32(&this_data[i + 4],  vsubq_f32(va1, vb1));
        vst1q_f32(&this_data[i + 8],  vsubq_f32(va2, vb2));
        vst1q_f32(&this_data[i + 12], vsubq_f32(va3, vb3));
        vst1q_f32(&this_data[i + 16], vsubq_f32(va4, vb4));
        vst1q_f32(&this_data[i + 20], vsubq_f32(va5, vb5));
        vst1q_f32(&this_data[i + 24], vsubq_f32(va6, vb6));
        vst1q_f32(&this_data[i + 28], vsubq_f32(va7, vb7));
    }
    // Tail: 16 floats
    if (len16 > len32) {
        const std::size_t i = len32;
        const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
        const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
        const float32x4_t va2 = vld1q_f32(&this_data[i + 8]);
        const float32x4_t va3 = vld1q_f32(&this_data[i + 12]);
        const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
        const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
        const float32x4_t vb2 = vld1q_f32(&other_data[i + 8]);
        const float32x4_t vb3 = vld1q_f32(&other_data[i + 12]);
        vst1q_f32(&this_data[i + 0],  vsubq_f32(va0, vb0));
        vst1q_f32(&this_data[i + 4],  vsubq_f32(va1, vb1));
        vst1q_f32(&this_data[i + 8],  vsubq_f32(va2, vb2));
        vst1q_f32(&this_data[i + 12], vsubq_f32(va3, vb3));
    }
    // Tail: 8 floats
    if (len8 > len16) {
        const std::size_t i = len16;
        const float32x4_t va0 = vld1q_f32(&this_data[i + 0]);
        const float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
        const float32x4_t vb0 = vld1q_f32(&other_data[i + 0]);
        const float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
        vst1q_f32(&this_data[i + 0], vsubq_f32(va0, vb0));
        vst1q_f32(&this_data[i + 4], vsubq_f32(va1, vb1));
    }
    // Tail: 4 floats
    if (len4 > len8) {
        const std::size_t i = len8;
        const float32x4_t va = vld1q_f32(&this_data[i]);
        const float32x4_t vb = vld1q_f32(&other_data[i]);
        vst1q_f32(&this_data[i], vsubq_f32(va, vb));
    }
    // Scalar tail
    for (std::size_t i = len4; i < size_; ++i) {
        this_data[i] -= other_data[i];
    }
}

// =======================
// FP16 kernels
// =======================

void NEONBackend::operation_fp16(const TensorBackend& other, OperationType op_type) {
    const float16_t* other_data = reinterpret_cast<const float16_t*>(other.data_fp16());
    float16_t* this_data = reinterpret_cast<float16_t*>(data_fp16());

    // Prefetch distance: 128 float16s (~256 bytes)
    constexpr std::size_t PREFETCH_DISTANCE = 128;

    // Hierarchical processing: 64 → 32 → 16 → 8 → 4 → scalar
    const std::size_t len64 = (size_ / 64) * 64;
    const std::size_t len32 = len64 + ((size_ - len64) / 32) * 32;
    const std::size_t len16 = len32 + ((size_ - len32) / 16) * 16;
    const std::size_t len8  = len16 + ((size_ - len16) / 8) * 8;
    const std::size_t len4  = len8  + ((size_ - len8) / 4) * 4;

    if (op_type == OperationType::ADD) {
        // Main loop: 64 values (8 registers × 8 float16s)
        for (std::size_t i = 0; i < len64; i += 64) {
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
            }
            const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
            const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
            const float16x8_t va2 = vld1q_f16(&this_data[i + 16]);
            const float16x8_t va3 = vld1q_f16(&this_data[i + 24]);
            const float16x8_t va4 = vld1q_f16(&this_data[i + 32]);
            const float16x8_t va5 = vld1q_f16(&this_data[i + 40]);
            const float16x8_t va6 = vld1q_f16(&this_data[i + 48]);
            const float16x8_t va7 = vld1q_f16(&this_data[i + 56]);

            const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
            const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
            const float16x8_t vb2 = vld1q_f16(&other_data[i + 16]);
            const float16x8_t vb3 = vld1q_f16(&other_data[i + 24]);
            const float16x8_t vb4 = vld1q_f16(&other_data[i + 32]);
            const float16x8_t vb5 = vld1q_f16(&other_data[i + 40]);
            const float16x8_t vb6 = vld1q_f16(&other_data[i + 48]);
            const float16x8_t vb7 = vld1q_f16(&other_data[i + 56]);

            vst1q_f16(&this_data[i + 0],  vaddq_f16(va0, vb0));
            vst1q_f16(&this_data[i + 8],  vaddq_f16(va1, vb1));
            vst1q_f16(&this_data[i + 16], vaddq_f16(va2, vb2));
            vst1q_f16(&this_data[i + 24], vaddq_f16(va3, vb3));
            vst1q_f16(&this_data[i + 32], vaddq_f16(va4, vb4));
            vst1q_f16(&this_data[i + 40], vaddq_f16(va5, vb5));
            vst1q_f16(&this_data[i + 48], vaddq_f16(va6, vb6));
            vst1q_f16(&this_data[i + 56], vaddq_f16(va7, vb7));
        }
        // Tail: 32 values (4 registers × 8 float16s)
        if (len32 > len64) {
            const std::size_t i = len64;
            const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
            const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
            const float16x8_t va2 = vld1q_f16(&this_data[i + 16]);
            const float16x8_t va3 = vld1q_f16(&this_data[i + 24]);
            const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
            const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
            const float16x8_t vb2 = vld1q_f16(&other_data[i + 16]);
            const float16x8_t vb3 = vld1q_f16(&other_data[i + 24]);
            vst1q_f16(&this_data[i + 0],  vaddq_f16(va0, vb0));
            vst1q_f16(&this_data[i + 8],  vaddq_f16(va1, vb1));
            vst1q_f16(&this_data[i + 16], vaddq_f16(va2, vb2));
            vst1q_f16(&this_data[i + 24], vaddq_f16(va3, vb3));
        }
        // Tail: 16 values (2 registers × 8 float16s)
        if (len16 > len32) {
            const std::size_t i = len32;
            const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
            const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
            const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
            const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
            vst1q_f16(&this_data[i + 0], vaddq_f16(va0, vb0));
            vst1q_f16(&this_data[i + 8], vaddq_f16(va1, vb1));
        }
        // Tail: 8 values (1 register)
        if (len8 > len16) {
            const std::size_t i = len16;
            const float16x8_t va = vld1q_f16(&this_data[i]);
            const float16x8_t vb = vld1q_f16(&other_data[i]);
            vst1q_f16(&this_data[i], vaddq_f16(va, vb));
        }
        // Tail: 4 values (half register)
        if (len4 > len8) {
            const std::size_t i = len8;
            const float16x4_t va = vld1_f16(&this_data[i]);
            const float16x4_t vb = vld1_f16(&other_data[i]);
            vst1_f16(&this_data[i], vadd_f16(va, vb));
        }
        // Scalar tail
        for (std::size_t i = len4; i < size_; ++i) {
            this_data[i] = this_data[i] + other_data[i];
        }
        return;
    }

    if (op_type == OperationType::MUL) {
        // Main loop: 64 values
        for (std::size_t i = 0; i < len64; i += 64) {
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
            }
            const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
            const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
            const float16x8_t va2 = vld1q_f16(&this_data[i + 16]);
            const float16x8_t va3 = vld1q_f16(&this_data[i + 24]);
            const float16x8_t va4 = vld1q_f16(&this_data[i + 32]);
            const float16x8_t va5 = vld1q_f16(&this_data[i + 40]);
            const float16x8_t va6 = vld1q_f16(&this_data[i + 48]);
            const float16x8_t va7 = vld1q_f16(&this_data[i + 56]);

            const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
            const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
            const float16x8_t vb2 = vld1q_f16(&other_data[i + 16]);
            const float16x8_t vb3 = vld1q_f16(&other_data[i + 24]);
            const float16x8_t vb4 = vld1q_f16(&other_data[i + 32]);
            const float16x8_t vb5 = vld1q_f16(&other_data[i + 40]);
            const float16x8_t vb6 = vld1q_f16(&other_data[i + 48]);
            const float16x8_t vb7 = vld1q_f16(&other_data[i + 56]);

            vst1q_f16(&this_data[i + 0],  vmulq_f16(va0, vb0));
            vst1q_f16(&this_data[i + 8],  vmulq_f16(va1, vb1));
            vst1q_f16(&this_data[i + 16], vmulq_f16(va2, vb2));
            vst1q_f16(&this_data[i + 24], vmulq_f16(va3, vb3));
            vst1q_f16(&this_data[i + 32], vmulq_f16(va4, vb4));
            vst1q_f16(&this_data[i + 40], vmulq_f16(va5, vb5));
            vst1q_f16(&this_data[i + 48], vmulq_f16(va6, vb6));
            vst1q_f16(&this_data[i + 56], vmulq_f16(va7, vb7));
        }
        // Tail: 32 values
        if (len32 > len64) {
            const std::size_t i = len64;
            const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
            const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
            const float16x8_t va2 = vld1q_f16(&this_data[i + 16]);
            const float16x8_t va3 = vld1q_f16(&this_data[i + 24]);
            const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
            const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
            const float16x8_t vb2 = vld1q_f16(&other_data[i + 16]);
            const float16x8_t vb3 = vld1q_f16(&other_data[i + 24]);
            vst1q_f16(&this_data[i + 0],  vmulq_f16(va0, vb0));
            vst1q_f16(&this_data[i + 8],  vmulq_f16(va1, vb1));
            vst1q_f16(&this_data[i + 16], vmulq_f16(va2, vb2));
            vst1q_f16(&this_data[i + 24], vmulq_f16(va3, vb3));
        }
        // Tail: 16 values
        if (len16 > len32) {
            const std::size_t i = len32;
            const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
            const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
            const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
            const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
            vst1q_f16(&this_data[i + 0], vmulq_f16(va0, vb0));
            vst1q_f16(&this_data[i + 8], vmulq_f16(va1, vb1));
        }
        // Tail: 8 values
        if (len8 > len16) {
            const std::size_t i = len16;
            const float16x8_t va = vld1q_f16(&this_data[i]);
            const float16x8_t vb = vld1q_f16(&other_data[i]);
            vst1q_f16(&this_data[i], vmulq_f16(va, vb));
        }
        // Tail: 4 values
        if (len4 > len8) {
            const std::size_t i = len8;
            const float16x4_t va = vld1_f16(&this_data[i]);
            const float16x4_t vb = vld1_f16(&other_data[i]);
            vst1_f16(&this_data[i], vmul_f16(va, vb));
        }
        // Scalar tail
        for (std::size_t i = len4; i < size_; ++i) {
            this_data[i] = this_data[i] * other_data[i];
        }
        return;
    }

    // DIV path for FP16
    if (op_type == OperationType::DIV) {
        // Main loop: 64 values (8 registers × 8 float16s)
        for (std::size_t i = 0; i < len64; i += 64) {
            if (i + PREFETCH_DISTANCE < size_) {
                __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
                __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
            }
            const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
            const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
            const float16x8_t va2 = vld1q_f16(&this_data[i + 16]);
            const float16x8_t va3 = vld1q_f16(&this_data[i + 24]);
            const float16x8_t va4 = vld1q_f16(&this_data[i + 32]);
            const float16x8_t va5 = vld1q_f16(&this_data[i + 40]);
            const float16x8_t va6 = vld1q_f16(&this_data[i + 48]);
            const float16x8_t va7 = vld1q_f16(&this_data[i + 56]);

            const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
            const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
            const float16x8_t vb2 = vld1q_f16(&other_data[i + 16]);
            const float16x8_t vb3 = vld1q_f16(&other_data[i + 24]);
            const float16x8_t vb4 = vld1q_f16(&other_data[i + 32]);
            const float16x8_t vb5 = vld1q_f16(&other_data[i + 40]);
            const float16x8_t vb6 = vld1q_f16(&other_data[i + 48]);
            const float16x8_t vb7 = vld1q_f16(&other_data[i + 56]);

            vst1q_f16(&this_data[i + 0],  vdivq_f16(va0, vb0));
            vst1q_f16(&this_data[i + 8],  vdivq_f16(va1, vb1));
            vst1q_f16(&this_data[i + 16], vdivq_f16(va2, vb2));
            vst1q_f16(&this_data[i + 24], vdivq_f16(va3, vb3));
            vst1q_f16(&this_data[i + 32], vdivq_f16(va4, vb4));
            vst1q_f16(&this_data[i + 40], vdivq_f16(va5, vb5));
            vst1q_f16(&this_data[i + 48], vdivq_f16(va6, vb6));
            vst1q_f16(&this_data[i + 56], vdivq_f16(va7, vb7));
        }
        // Tail: 32 values
        if (len32 > len64) {
            const std::size_t i = len64;
            const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
            const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
            const float16x8_t va2 = vld1q_f16(&this_data[i + 16]);
            const float16x8_t va3 = vld1q_f16(&this_data[i + 24]);
            const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
            const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
            const float16x8_t vb2 = vld1q_f16(&other_data[i + 16]);
            const float16x8_t vb3 = vld1q_f16(&other_data[i + 24]);
            vst1q_f16(&this_data[i + 0],  vdivq_f16(va0, vb0));
            vst1q_f16(&this_data[i + 8],  vdivq_f16(va1, vb1));
            vst1q_f16(&this_data[i + 16], vdivq_f16(va2, vb2));
            vst1q_f16(&this_data[i + 24], vdivq_f16(va3, vb3));
        }
        // Tail: 16 values
        if (len16 > len32) {
            const std::size_t i = len32;
            const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
            const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
            const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
            const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
            vst1q_f16(&this_data[i + 0], vdivq_f16(va0, vb0));
            vst1q_f16(&this_data[i + 8], vdivq_f16(va1, vb1));
        }
        // Tail: 8 values
        if (len8 > len16) {
            const std::size_t i = len16;
            const float16x8_t va = vld1q_f16(&this_data[i]);
            const float16x8_t vb = vld1q_f16(&other_data[i]);
            vst1q_f16(&this_data[i], vdivq_f16(va, vb));
        }
        // Tail: 4 values
        if (len4 > len8) {
            const std::size_t i = len8;
            const float16x4_t va = vld1_f16(&this_data[i]);
            const float16x4_t vb = vld1_f16(&other_data[i]);
            vst1_f16(&this_data[i], vdiv_f16(va, vb));
        }
        // Scalar tail
        for (std::size_t i = len4; i < size_; ++i) {
            // Cast int16_t to uint16_t for fp16 operations
            uint16_t a_bits, b_bits;
            std::memcpy(&a_bits, &this_data[i], sizeof(uint16_t));
            std::memcpy(&b_bits, &other_data[i], sizeof(uint16_t));
            float a = dequantize_fp16(a_bits);
            float b = dequantize_fp16(b_bits);
            uint16_t result = quantize_fp16(a / b);
            std::memcpy(&this_data[i], &result, sizeof(uint16_t));
        }
        return;
    }

    // SUB path (default)
    for (std::size_t i = 0; i < len64; i += 64) {
        if (i + PREFETCH_DISTANCE < size_) {
            __builtin_prefetch(&this_data[i + PREFETCH_DISTANCE], 1, 1);
            __builtin_prefetch(&other_data[i + PREFETCH_DISTANCE], 0, 1);
        }

        const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
        const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
        const float16x8_t va2 = vld1q_f16(&this_data[i + 16]);
        const float16x8_t va3 = vld1q_f16(&this_data[i + 24]);
        const float16x8_t va4 = vld1q_f16(&this_data[i + 32]);
        const float16x8_t va5 = vld1q_f16(&this_data[i + 40]);
        const float16x8_t va6 = vld1q_f16(&this_data[i + 48]);
        const float16x8_t va7 = vld1q_f16(&this_data[i + 56]);

        const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
        const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
        const float16x8_t vb2 = vld1q_f16(&other_data[i + 16]);
        const float16x8_t vb3 = vld1q_f16(&other_data[i + 24]);
        const float16x8_t vb4 = vld1q_f16(&other_data[i + 32]);
        const float16x8_t vb5 = vld1q_f16(&other_data[i + 40]);
        const float16x8_t vb6 = vld1q_f16(&other_data[i + 48]);
        const float16x8_t vb7 = vld1q_f16(&other_data[i + 56]);

        vst1q_f16(&this_data[i + 0],  vsubq_f16(va0, vb0));
        vst1q_f16(&this_data[i + 8],  vsubq_f16(va1, vb1));
        vst1q_f16(&this_data[i + 16], vsubq_f16(va2, vb2));
        vst1q_f16(&this_data[i + 24], vsubq_f16(va3, vb3));
        vst1q_f16(&this_data[i + 32], vsubq_f16(va4, vb4));
        vst1q_f16(&this_data[i + 40], vsubq_f16(va5, vb5));
        vst1q_f16(&this_data[i + 48], vsubq_f16(va6, vb6));
        vst1q_f16(&this_data[i + 56], vsubq_f16(va7, vb7));
    }
    // Tail: 32 values
    if (len32 > len64) {
        const std::size_t i = len64;
        const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
        const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
        const float16x8_t va2 = vld1q_f16(&this_data[i + 16]);
        const float16x8_t va3 = vld1q_f16(&this_data[i + 24]);
        const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
        const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
        const float16x8_t vb2 = vld1q_f16(&other_data[i + 16]);
        const float16x8_t vb3 = vld1q_f16(&other_data[i + 24]);
        vst1q_f16(&this_data[i + 0],  vsubq_f16(va0, vb0));
        vst1q_f16(&this_data[i + 8],  vsubq_f16(va1, vb1));
        vst1q_f16(&this_data[i + 16], vsubq_f16(va2, vb2));
        vst1q_f16(&this_data[i + 24], vsubq_f16(va3, vb3));
    }
    // Tail: 16 values
    if (len16 > len32) {
        const std::size_t i = len32;
        const float16x8_t va0 = vld1q_f16(&this_data[i + 0]);
        const float16x8_t va1 = vld1q_f16(&this_data[i + 8]);
        const float16x8_t vb0 = vld1q_f16(&other_data[i + 0]);
        const float16x8_t vb1 = vld1q_f16(&other_data[i + 8]);
        vst1q_f16(&this_data[i + 0], vsubq_f16(va0, vb0));
        vst1q_f16(&this_data[i + 8], vsubq_f16(va1, vb1));
    }
    // Tail: 8 values
    if (len8 > len16) {
        const std::size_t i = len16;
        const float16x8_t va = vld1q_f16(&this_data[i]);
        const float16x8_t vb = vld1q_f16(&other_data[i]);
        vst1q_f16(&this_data[i], vsubq_f16(va, vb));
    }
    // Tail: 4 values
    if (len4 > len8) {
        const std::size_t i = len8;
        const float16x4_t va = vld1_f16(&this_data[i]);
        const float16x4_t vb = vld1_f16(&other_data[i]);
        vst1_f16(&this_data[i], vsub_f16(va, vb));
    }
    // Scalar tail
    for (std::size_t i = len4; i < size_; ++i) {
        this_data[i] = this_data[i] - other_data[i];
    }
}

// =======================
// INT8 kernels
// =======================


} // namespace labneura

#endif // __ARM_NEON || __aarch64__
