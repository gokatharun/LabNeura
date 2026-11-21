#if defined(__ARM_NEON) || defined(__aarch64__)

#include "labneura/backends/cpu/neon.h"
#include "neon_internal.h"
#include "labneura/runtime.h"
#include "labneura/utils/alignment.hpp"
#include "labneura/types.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace labneura {

namespace {

constexpr std::size_t kParallelAddThreshold = 1000000;
constexpr std::size_t kParallelAddGrainSize = 16384;

// FP32 semantics: ADD/SUB/MUL/DIV are IEEE-754 single-precision on AArch64 (the vector
// and scalar-tail paths both use exactly-rounded hardware operations, so signed zeros,
// infinities, NaNs and denormals behave identically). Only 32-bit ARM (no vdivq_f32)
// uses a reciprocal-estimate + Newton-Raphson division, which is NOT correctly rounded.
struct Fp32Add {
    static float32x4_t v(float32x4_t a, float32x4_t b) { return vaddq_f32(a, b); }
    static float s(float a, float b) { return a + b; }
};
struct Fp32Sub {
    static float32x4_t v(float32x4_t a, float32x4_t b) { return vsubq_f32(a, b); }
    static float s(float a, float b) { return a - b; }
};
struct Fp32Mul {
    static float32x4_t v(float32x4_t a, float32x4_t b) { return vmulq_f32(a, b); }
    static float s(float a, float b) { return a * b; }
};
struct Fp32Div {
    static float32x4_t v(float32x4_t a, float32x4_t b) {
#if defined(__aarch64__)
        return vdivq_f32(a, b);
#else
        float32x4_t r = vrecpeq_f32(b);
        r = vmulq_f32(r, vrecpsq_f32(b, r));
        r = vmulq_f32(r, vrecpsq_f32(b, r));
        return vmulq_f32(a, r);
#endif
    }
    static float s(float a, float b) { return a / b; }
};

// dst[i] = a[i] op b[i]. dst may equal a or b exactly (each element is loaded before
// it is stored); partially overlapping buffers are not supported.
// ADD/SUB/MUL are bandwidth-bound and run faster without software prefetch (measured
// on Apple M1), so prefetching is enabled only where Prefetch is true (DIV).
template <typename Op, bool Prefetch = false>
void fp32_kernel(float* dst, const float* a, const float* b, std::size_t n) {
    // 256 bytes ahead of the current block
    constexpr std::size_t PREFETCH_DISTANCE = 64;
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        if constexpr (Prefetch) {
            if (i + PREFETCH_DISTANCE < n) {
                __builtin_prefetch(&dst[i + PREFETCH_DISTANCE], 1, 1);
                if (dst != a) __builtin_prefetch(&a[i + PREFETCH_DISTANCE], 0, 1);
                if (dst != b) __builtin_prefetch(&b[i + PREFETCH_DISTANCE], 0, 1);
            }
        }
        const float32x4_t a0 = vld1q_f32(&a[i + 0]);
        const float32x4_t a1 = vld1q_f32(&a[i + 4]);
        const float32x4_t a2 = vld1q_f32(&a[i + 8]);
        const float32x4_t a3 = vld1q_f32(&a[i + 12]);
        const float32x4_t a4 = vld1q_f32(&a[i + 16]);
        const float32x4_t a5 = vld1q_f32(&a[i + 20]);
        const float32x4_t a6 = vld1q_f32(&a[i + 24]);
        const float32x4_t a7 = vld1q_f32(&a[i + 28]);
        const float32x4_t b0 = vld1q_f32(&b[i + 0]);
        const float32x4_t b1 = vld1q_f32(&b[i + 4]);
        const float32x4_t b2 = vld1q_f32(&b[i + 8]);
        const float32x4_t b3 = vld1q_f32(&b[i + 12]);
        const float32x4_t b4 = vld1q_f32(&b[i + 16]);
        const float32x4_t b5 = vld1q_f32(&b[i + 20]);
        const float32x4_t b6 = vld1q_f32(&b[i + 24]);
        const float32x4_t b7 = vld1q_f32(&b[i + 28]);
        vst1q_f32(&dst[i + 0],  Op::v(a0, b0));
        vst1q_f32(&dst[i + 4],  Op::v(a1, b1));
        vst1q_f32(&dst[i + 8],  Op::v(a2, b2));
        vst1q_f32(&dst[i + 12], Op::v(a3, b3));
        vst1q_f32(&dst[i + 16], Op::v(a4, b4));
        vst1q_f32(&dst[i + 20], Op::v(a5, b5));
        vst1q_f32(&dst[i + 24], Op::v(a6, b6));
        vst1q_f32(&dst[i + 28], Op::v(a7, b7));
    }
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(&dst[i], Op::v(vld1q_f32(&a[i]), vld1q_f32(&b[i])));
    }
    for (; i < n; ++i) {
        dst[i] = Op::s(a[i], b[i]);
    }
}

void fp32_dispatch(float* dst, const float* a, const float* b, std::size_t n, OperationType op) {
    if (op == OperationType::ADD) {
        fp32_kernel<Fp32Add>(dst, a, b, n);
    } else if (op == OperationType::MUL) {
        fp32_kernel<Fp32Mul>(dst, a, b, n);
    } else if (op == OperationType::DIV) {
        fp32_kernel<Fp32Div, true>(dst, a, b, n);
    } else {
        fp32_kernel<Fp32Sub>(dst, a, b, n);
    }
}

void fp32_add_dispatch(float* dst, const float* a, const float* b,
                       std::size_t n) {
    if (n >= kParallelAddThreshold && execution_thread_count() > 1) {
        parallel_for(0, n, kParallelAddGrainSize,
                     [dst, a, b](std::size_t begin, std::size_t end) {
                         fp32_kernel<Fp32Add>(dst + begin, a + begin, b + begin,
                                              end - begin);
                     });
        return;
    }
    fp32_kernel<Fp32Add>(dst, a, b, n);
}

}  // namespace

void NEONBackend::operation_fp32(const TensorBackend& other, OperationType op_type) {
    float* this_data = data_fp32();
    if (op_type == OperationType::ADD) {
        fp32_add_dispatch(this_data, this_data, other.data_fp32(), size_);
        return;
    }
    fp32_dispatch(this_data, this_data, other.data_fp32(), size_, op_type);
}

std::unique_ptr<TensorBackend> NEONBackend::create_output() const {
    if (quantization_mode_ != QuantizationMode::FP32) {
        return nullptr;
    }
    return std::make_unique<NEONBackend>(size_, quantization_mode_);
}

bool NEONBackend::operation_into(const TensorBackend& other, TensorBackend& out, OperationType op_type) const {
    if (quantization_mode_ != QuantizationMode::FP32) {
        return false;
    }
    if (out.size() != size_ || out.quantization_mode() != quantization_mode_) {
        throw std::runtime_error("Output tensor does not match operand size or mode");
    }
    if (op_type == OperationType::ADD) {
        fp32_add_dispatch(out.data_fp32(), data_fp32(), other.data_fp32(), size_);
    } else {
        fp32_dispatch(out.data_fp32(), data_fp32(), other.data_fp32(), size_,
                      op_type);
    }
    return true;
}

// =======================
// FP16 kernels
// =======================

void NEONBackend::operation_fp16(const TensorBackend& other, OperationType op_type) {
    const float16_t* other_data = reinterpret_cast<const float16_t*>(other.data_fp16());
    float16_t* this_data = reinterpret_cast<float16_t*>(data_fp16());

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__)
    // Prefetch distance: 128 float16s (~256 bytes)
    constexpr std::size_t PREFETCH_DISTANCE = 128;

    // Hierarchical processing: 64 → 32 → 16 → 8 → 4 → scalar
    const std::size_t len64 = (size_ / 64) * 64;
    const std::size_t len32 = len64 + ((size_ - len64) / 32) * 32;
    const std::size_t len16 = len32 + ((size_ - len32) / 16) * 16;
    const std::size_t len8  = len16 + ((size_ - len16) / 8) * 8;
    const std::size_t len4  = len8  + ((size_ - len8) / 4) * 4;
#endif

    if (op_type == OperationType::ADD) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__)
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
#else
        for (std::size_t i = 0; i < size_; ++i) {
            this_data[i] = this_data[i] + other_data[i];
        }
#endif
        return;
    }

    if (op_type == OperationType::MUL) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__)
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
#else
        for (std::size_t i = 0; i < size_; ++i) {
            this_data[i] = this_data[i] * other_data[i];
        }
#endif
        return;
    }

    // DIV path for FP16
    if (op_type == OperationType::DIV) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__)
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
        // Scalar tail: float16_t <-> float conversions are IEEE round-to-nearest-even.
        for (std::size_t i = len4; i < size_; ++i) {
            const float a = static_cast<float>(this_data[i]);
            const float b = static_cast<float>(other_data[i]);
            this_data[i] = static_cast<float16_t>(a / b);
        }
#else
        // Without native FP16 arithmetic, divide in FP32 and round once to FP16.
        for (std::size_t i = 0; i < size_; ++i) {
            const float a = static_cast<float>(this_data[i]);
            const float b = static_cast<float>(other_data[i]);
            this_data[i] = static_cast<float16_t>(a / b);
        }
#endif
        return;
    }

    // SUB path (default)
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__)
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
#else
    for (std::size_t i = 0; i < size_; ++i) {
        this_data[i] = this_data[i] - other_data[i];
    }
#endif
}

// =======================
// INT8 kernels
// =======================


} // namespace labneura

#endif // __ARM_NEON || __aarch64__
