#pragma once

namespace labneura::detail {

enum class AVX2FP16Kernel {
    SCALAR,
    F16C,
};

// F16C is independent of AVX2: the F16C kernel is used only when it was
// compiled into this build and the CPU and OS both report support for it.
inline AVX2FP16Kernel select_avx2_fp16_kernel(bool f16c_compiled,
                                               bool f16c_supported) {
    return (f16c_compiled && f16c_supported) ? AVX2FP16Kernel::F16C
                                             : AVX2FP16Kernel::SCALAR;
}

// Internal test seam (not installed, not part of the public API). Defined in
// avx2_ops_fp16.cpp, only when the AVX2 backend is built.
bool avx2_fp16_f16c_compiled();
// The kernel AVX2Backend::operation_fp16 will use right now.
AVX2FP16Kernel avx2_fp16_active_kernel();
// Forces the scalar fallback (never forces F16C, so it cannot run F16C code
// on unsupported CPUs).
void avx2_fp16_force_scalar_for_testing(bool force);

}  // namespace labneura::detail
