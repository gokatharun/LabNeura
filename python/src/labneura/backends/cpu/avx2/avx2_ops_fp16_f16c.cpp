#if defined(__AVX2__) && defined(__F16C__)

#include "avx2_fp16_f16c.h"
#include <immintrin.h>
#include <stdexcept>

namespace labneura {

namespace {

constexpr int kRoundNearestEven = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

struct AddOp {
    static __m256 vec(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
    static float scalar(float a, float b) { return a + b; }
};

struct SubOp {
    static __m256 vec(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
    static float scalar(float a, float b) { return a - b; }
};

struct MulOp {
    static __m256 vec(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
    static float scalar(float a, float b) { return a * b; }
};

struct DivOp {
    static __m256 vec(__m256 a, __m256 b) { return _mm256_div_ps(a, b); }
    static float scalar(float a, float b) { return a / b; }
};

// FP32 arithmetic on FP16 operands followed by one round-to-nearest-even
// conversion is correctly rounded for these four operations.
template <typename Op>
void run(int16_t* dst, const int16_t* src, std::size_t size) {
    std::size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        const __m256 a = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst + i)));
        const __m256 b = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                         _mm256_cvtps_ph(Op::vec(a, b), kRoundNearestEven));
    }
    for (; i < size; ++i) {
        const float a = _cvtsh_ss(static_cast<uint16_t>(dst[i]));
        const float b = _cvtsh_ss(static_cast<uint16_t>(src[i]));
        dst[i] = static_cast<int16_t>(
            _cvtss_sh(Op::scalar(a, b), kRoundNearestEven));
    }
}

}  // namespace

void avx2_fp16_f16c_kernel(int16_t* dst, const int16_t* src, std::size_t size,
                           OperationType op_type) {
    switch (op_type) {
        case ADD: run<AddOp>(dst, src, size); break;
        case SUB: run<SubOp>(dst, src, size); break;
        case MUL: run<MulOp>(dst, src, size); break;
        case DIV: run<DivOp>(dst, src, size); break;
        default: throw std::invalid_argument("Unsupported FP16 operation");
    }
}

}  // namespace labneura

#endif  // __AVX2__ && __F16C__
