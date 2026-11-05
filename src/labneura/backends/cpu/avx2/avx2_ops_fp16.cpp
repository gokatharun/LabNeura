#if defined(__AVX2__)

#include "labneura/backends/cpu/avx2.h"
#include "labneura/backends/common/cpu_features.h"
#include "labneura/backends/common/fp16_kernel_selection.h"
#include "labneura/types.hpp"
#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#if defined(LABNEURA_ENABLE_AVX2_F16C)
#include "avx2_fp16_f16c.h"
#endif

namespace labneura {

namespace {

// Software-conversion fallback for CPUs without F16C. IEEE semantics come from
// quantize_fp16/dequantize_fp16; this file is built without -mf16c.
template <typename Op>
void run_scalar(int16_t* dst, const int16_t* src, std::size_t size, Op op) {
    for (std::size_t i = 0; i < size; ++i) {
        uint16_t lhs_bits;
        uint16_t rhs_bits;
        std::memcpy(&lhs_bits, &dst[i], sizeof(lhs_bits));
        std::memcpy(&rhs_bits, &src[i], sizeof(rhs_bits));
        const uint16_t result_bits =
            quantize_fp16(op(dequantize_fp16(lhs_bits), dequantize_fp16(rhs_bits)));
        std::memcpy(&dst[i], &result_bits, sizeof(result_bits));
    }
}

#if defined(LABNEURA_ENABLE_AVX2_F16C)
constexpr bool kF16CCompiled = true;
#else
constexpr bool kF16CCompiled = false;
#endif

std::atomic<bool> g_force_scalar{false};

}  // namespace

namespace detail {

bool avx2_fp16_f16c_compiled() { return kF16CCompiled; }

AVX2FP16Kernel avx2_fp16_active_kernel() {
    static const bool f16c_supported = cpu_supports_f16c();
    if (g_force_scalar.load(std::memory_order_relaxed)) {
        return AVX2FP16Kernel::SCALAR;
    }
    return select_avx2_fp16_kernel(kF16CCompiled, f16c_supported);
}

void avx2_fp16_force_scalar_for_testing(bool force) {
    g_force_scalar.store(force, std::memory_order_relaxed);
}

}  // namespace detail

void AVX2Backend::operation_fp16(const TensorBackend& other, OperationType op_type) {
    const int16_t* other_data = other.data_fp16();
    int16_t* this_data = data_fp16();

    if (detail::avx2_fp16_active_kernel() == detail::AVX2FP16Kernel::F16C) {
#if defined(LABNEURA_ENABLE_AVX2_F16C)
        avx2_fp16_f16c_kernel(this_data, other_data, size_, op_type);
        return;
#endif
    }

    switch (op_type) {
        case ADD:
            run_scalar(this_data, other_data, size_, [](float a, float b) { return a + b; });
            break;
        case SUB:
            run_scalar(this_data, other_data, size_, [](float a, float b) { return a - b; });
            break;
        case MUL:
            run_scalar(this_data, other_data, size_, [](float a, float b) { return a * b; });
            break;
        case DIV:
            run_scalar(this_data, other_data, size_, [](float a, float b) { return a / b; });
            break;
        default:
            throw std::invalid_argument("Unsupported FP16 operation");
    }
}

} // namespace labneura

#endif // __AVX2__
