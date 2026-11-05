#include "labneura/backends/common/backend_factory.h"
#include "labneura/backends/common/cpu_features.h"
#include "fp16_kernel_selection.h"
#include "labneura/tensor.h"
#include "backend_selection.h"
#include "test_environment.h"
#if defined(LABNEURA_TEST_AVX2)
#include "labneura/backends/cpu/avx2.h"
#endif
#if defined(LABNEURA_TEST_NEON)
#include "labneura/backends/cpu/neon.h"
#endif

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class SkipTest : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// ---- Independent IEEE binary16 reference (does not use labneura/types.hpp) ----

double half_to_double(uint16_t bits) {
    const int sign = (bits >> 15) & 1;
    const int exponent = (bits >> 10) & 0x1F;
    const int mantissa = bits & 0x3FF;
    double value;
    if (exponent == 0x1F) {
        value = mantissa == 0 ? std::numeric_limits<double>::infinity()
                              : std::numeric_limits<double>::quiet_NaN();
    } else if (exponent == 0) {
        value = std::ldexp(static_cast<double>(mantissa), -24);
    } else {
        value = std::ldexp(static_cast<double>(mantissa + 1024), exponent - 25);
    }
    return sign ? -value : value;
}

// Rounds with the current (default round-to-nearest-even) mode via nearbyint.
uint16_t double_to_half(double x) {
    if (std::isnan(x)) {
        return 0x7E00;
    }
    const uint16_t sign = std::signbit(x) ? 0x8000 : 0;
    const double a = std::fabs(x);
    if (a >= 65520.0) {
        return sign | 0x7C00;
    }
    if (a < std::ldexp(1.0, -14)) {
        return sign | static_cast<uint16_t>(std::nearbyint(std::ldexp(a, 24)));
    }
    int e = std::ilogb(a);
    long m = static_cast<long>(std::nearbyint(std::ldexp(a, 10 - e)));
    if (m == 2048) {
        m = 1024;
        ++e;
    }
    return sign | static_cast<uint16_t>(((e + 15) << 10) | (m - 1024));
}

bool is_nan_bits(uint16_t bits) {
    return (bits & 0x7C00) == 0x7C00 && (bits & 0x03FF) != 0;
}

uint16_t expected_bits(uint16_t lhs, uint16_t rhs, labneura::OperationType op) {
    const double a = half_to_double(lhs);
    const double b = half_to_double(rhs);
    switch (op) {
        case labneura::ADD: return double_to_half(a + b);
        case labneura::SUB: return double_to_half(a - b);
        case labneura::MUL: return double_to_half(a * b);
        case labneura::DIV: return double_to_half(a / b);
    }
    throw std::invalid_argument("bad op");
}

const char* op_name(labneura::OperationType op) {
    switch (op) {
        case labneura::ADD: return "ADD";
        case labneura::SUB: return "SUB";
        case labneura::MUL: return "MUL";
        case labneura::DIV: return "DIV";
    }
    return "?";
}

const labneura::OperationType kOps[] = {labneura::ADD, labneura::SUB,
                                         labneura::MUL, labneura::DIV};

// Each value probes a specific IEEE case; see the comments for the pairings
// that produce round-to-nearest-even ties.
const std::vector<uint16_t> kSpecials = {
    0x0000,  // +0
    0x8000,  // -0
    0x7C00,  // +inf
    0xFC00,  // -inf
    0x7E00,  // quiet NaN
    0x0001,  // smallest positive subnormal
    0x8001,  // smallest negative subnormal
    0x0003,  // 3 subnormal units (tie when halved)
    0x03FF,  // largest subnormal
    0x0400,  // smallest positive normal
    0x8400,  // smallest negative normal
    0x7BFF,  // max finite (65504)
    0xFBFF,  // -max finite
    0x3C00,  // 1.0
    0xBC00,  // -1.0
    0x4000,  // 2.0
    0x3800,  // 0.5
    0x4200,  // 3.0
    0x3C01,  // 1 + 2^-10 (odd mantissa)
    0x1000,  // 2^-11: 1.0 + 2^-11 is a tie that rounds to even
    0x4800,  // 8: 65504 + 8 rounds back to 65504
    0x4C00,  // 16: 65504 + 16 = 65520 overflows to infinity
};

struct Operands {
    std::vector<uint16_t> lhs;
    std::vector<uint16_t> rhs;
};

// Every special/special pair, rotated by `offset` so each size sees different
// pairs in different SIMD lanes.
Operands special_pairs(std::size_t size, std::size_t offset) {
    std::vector<std::pair<uint16_t, uint16_t>> all;
    for (const uint16_t a : kSpecials) {
        for (const uint16_t b : kSpecials) {
            all.emplace_back(a, b);
        }
    }
    Operands out;
    for (std::size_t i = 0; i < size; ++i) {
        const auto& pair = all[(i + offset) % all.size()];
        out.lhs.push_back(pair.first);
        out.rhs.push_back(pair.second);
    }
    return out;
}

// Distinct, finite, non-zero values in every lane.
Operands distinct_lanes(std::size_t size) {
    Operands out;
    for (std::size_t i = 0; i < size; ++i) {
        const double a = (static_cast<double>(i) * 3.0 + 1.0) * 0.5;
        const double b = static_cast<double>((i * 7) % 13 + 1) * 0.25;
        out.lhs.push_back(double_to_half(i % 5 == 4 ? -a : a));
        out.rhs.push_back(double_to_half(i % 3 == 2 ? -b : b));
    }
    return out;
}

void check_bits(const int16_t* actual, const Operands& in,
                labneura::OperationType op, const std::string& context) {
    for (std::size_t i = 0; i < in.lhs.size(); ++i) {
        const uint16_t want = expected_bits(in.lhs[i], in.rhs[i], op);
        const uint16_t got = static_cast<uint16_t>(actual[i]);
        const bool same = is_nan_bits(want) ? is_nan_bits(got) : got == want;
        if (!same) {
            throw std::runtime_error(
                context + " " + op_name(op) + ": element " + std::to_string(i) +
                " lhs=0x" + std::to_string(in.lhs[i]) +
                " rhs=0x" + std::to_string(in.rhs[i]) +
                " expected bits " + std::to_string(want) +
                ", got " + std::to_string(got) + " (decimal)");
        }
    }
}

labneura::Tensor make_tensor(const std::vector<uint16_t>& bits) {
    labneura::Tensor tensor(std::vector<float>(bits.size(), 0.0f),
                            labneura::QuantizationMode::FP16);
    for (std::size_t i = 0; i < bits.size(); ++i) {
        tensor.data_fp16()[i] = static_cast<int16_t>(bits[i]);
    }
    return tensor;
}

template <typename BackendType>
void run_backend_checks(const std::string& label) {
    const std::vector<std::size_t> sizes = {1, 7, 8, 9, 15, 16, 17, 31, 32,
                                            33, 63, 64, 65};
    for (const std::size_t size : sizes) {
        std::vector<Operands> cases = {distinct_lanes(size)};
        for (const std::size_t offset : {0u, 5u, 131u, 400u}) {
            cases.push_back(special_pairs(size, offset));
        }
        for (const auto& in : cases) {
            for (const auto op : kOps) {
                const std::string context =
                    label + " size " + std::to_string(size);

                auto lhs_backend = labneura::create_best_backend(
                    size, labneura::QuantizationMode::FP16);
                auto rhs_backend = labneura::create_best_backend(
                    size, labneura::QuantizationMode::FP16);
                require(dynamic_cast<BackendType*>(lhs_backend.get()) != nullptr &&
                            dynamic_cast<BackendType*>(rhs_backend.get()) != nullptr,
                        label + " override did not select the intended backend");
                for (std::size_t i = 0; i < size; ++i) {
                    lhs_backend->data_fp16()[i] = static_cast<int16_t>(in.lhs[i]);
                    rhs_backend->data_fp16()[i] = static_cast<int16_t>(in.rhs[i]);
                }
                lhs_backend->operation(*rhs_backend, op);
                check_bits(lhs_backend->data_fp16(), in, op,
                           context + " direct in-place");

                labneura::Tensor inplace = make_tensor(in.lhs);
                const labneura::Tensor rhs_tensor = make_tensor(in.rhs);
                switch (op) {
                    case labneura::ADD: inplace.add_inplace(rhs_tensor); break;
                    case labneura::SUB: inplace.sub_inplace(rhs_tensor); break;
                    case labneura::MUL: inplace.mul_inplace(rhs_tensor); break;
                    case labneura::DIV: inplace.div_inplace(rhs_tensor); break;
                }
                check_bits(inplace.data_fp16(), in, op, context + " Tensor in-place");

                const labneura::Tensor lhs_tensor = make_tensor(in.lhs);
                labneura::Tensor allocated;
                switch (op) {
                    case labneura::ADD: allocated = lhs_tensor.add(rhs_tensor); break;
                    case labneura::SUB: allocated = lhs_tensor.sub(rhs_tensor); break;
                    case labneura::MUL: allocated = lhs_tensor.mul(rhs_tensor); break;
                    case labneura::DIV: allocated = lhs_tensor.div(rhs_tensor); break;
                }
                check_bits(allocated.data_fp16(), in, op,
                           context + " Tensor allocating");
                // Operands of an allocating operation must be left untouched.
                for (std::size_t i = 0; i < size; ++i) {
                    require(static_cast<uint16_t>(lhs_tensor.data_fp16()[i]) ==
                                in.lhs[i],
                            context + " allocating operation modified its input");
                }
            }
        }
    }
}

void check_reference_self_consistency() {
    // Guards the independent reference itself against typos.
    require(double_to_half(1.0) == 0x3C00, "reference: 1.0");
    require(double_to_half(65504.0) == 0x7BFF, "reference: max finite");
    require(double_to_half(65520.0) == 0x7C00, "reference: overflow tie");
    require(double_to_half(std::ldexp(1.0, -24)) == 0x0001, "reference: min subnormal");
    require(double_to_half(std::ldexp(1.0, -25)) == 0x0000, "reference: underflow tie to even");
    require(double_to_half(1.0 + std::ldexp(1.0, -11)) == 0x3C00, "reference: tie to even down");
    require(double_to_half(1.0 + std::ldexp(3.0, -11)) == 0x3C02, "reference: tie to even up");
    require(half_to_double(0x0400) == std::ldexp(1.0, -14), "reference: min normal");
    require(expected_bits(0x3C00, 0x0000, labneura::DIV) == 0x7C00, "reference: 1/0");
    require(expected_bits(0xBC00, 0x0000, labneura::DIV) == 0xFC00, "reference: -1/0");
    require(is_nan_bits(expected_bits(0x0000, 0x0000, labneura::DIV)), "reference: 0/0");
    require(is_nan_bits(expected_bits(0x7C00, 0x7C00, labneura::DIV)), "reference: inf/inf");
}

void check_dispatch() {
    using labneura::detail::AVX2FP16Kernel;
    using labneura::detail::select_avx2_fp16_kernel;
    using labneura::detail::select_backend;
    using labneura::detail::BackendKind;

    // AVX2 + F16C available.
    require(select_backend("AVX2", true, true, false, false) == BackendKind::AVX2,
            "AVX2 selection must work when AVX2 is available");
    require(select_avx2_fp16_kernel(true, true) == AVX2FP16Kernel::F16C,
            "AVX2 + F16C must use the F16C kernel");
    // AVX2 available, F16C unavailable: AVX2 stays selected, kernel is scalar.
    require(select_backend("AVX2", true, true, false, false) == BackendKind::AVX2,
            "AVX2 selection must not depend on F16C");
    require(select_avx2_fp16_kernel(true, false) == AVX2FP16Kernel::SCALAR,
            "AVX2 without F16C must use the scalar kernel");
    require(select_avx2_fp16_kernel(false, true) == AVX2FP16Kernel::SCALAR,
            "F16C kernel not compiled in must use the scalar kernel");
    // AVX2 unavailable.
    bool rejected = false;
    try {
        select_backend("AVX2", true, false, false, false);
    } catch (const std::invalid_argument& error) {
        rejected = std::string(error.what()) ==
                   "Requested backend AVX2 is not supported by this CPU";
    }
    require(rejected, "Forced AVX2 must be rejected when AVX2 is unavailable");

    const auto features = labneura::detect_cpu_features();
    require(features.f16c == labneura::cpu_supports_f16c(),
            "detect_cpu_features().f16c must match cpu_supports_f16c()");
#if !defined(__x86_64__) && !defined(__i386__) && !defined(_M_X64) && !defined(_M_IX86)
    require(!features.f16c, "F16C must not be reported on non-x86 CPUs");
#endif
}

void check_avx2() {
#if !defined(LABNEURA_TEST_AVX2)
    throw SkipTest("AVX2 backend was not compiled for this target");
#else
    if (!labneura::cpu_supports_avx2()) {
        throw SkipTest("CPU/OS does not support AVX2 state");
    }
    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", "AVX2");
    using labneura::detail::AVX2FP16Kernel;
    const bool compiled = labneura::detail::avx2_fp16_f16c_compiled();
    const bool supported = labneura::cpu_supports_f16c();
    std::cout << "F16C compiled: " << compiled << ", CPU supports F16C: "
              << supported << '\n';

    if (compiled && supported) {
        require(labneura::detail::avx2_fp16_active_kernel() == AVX2FP16Kernel::F16C,
                "F16C is compiled and supported but the F16C kernel is not selected");
        std::cout << "Running AVX2 FP16 with the F16C vector kernel\n";
        run_backend_checks<labneura::AVX2Backend>("AVX2/F16C");
    } else {
        std::cout << "F16C kernel unavailable; scalar fallback is the active kernel\n";
    }

    // Scalar fallback, forced so it is covered on F16C-capable hosts too.
    labneura::detail::avx2_fp16_force_scalar_for_testing(true);
    try {
        require(labneura::detail::avx2_fp16_active_kernel() == AVX2FP16Kernel::SCALAR,
                "forcing the scalar fallback did not select it");
        std::cout << "Running AVX2 FP16 with the scalar fallback\n";
        run_backend_checks<labneura::AVX2Backend>("AVX2/scalar");
    } catch (...) {
        labneura::detail::avx2_fp16_force_scalar_for_testing(false);
        throw;
    }
    labneura::detail::avx2_fp16_force_scalar_for_testing(false);
#endif
}

void check_neon() {
#if !defined(LABNEURA_TEST_NEON)
    throw SkipTest("NEON backend was not compiled for this target");
#else
    if (!labneura::cpu_supports_neon()) {
        throw SkipTest("CPU does not support NEON");
    }
    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", "NEON");
    run_backend_checks<labneura::NEONBackend>("NEON");
#endif
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::invalid_argument("Expected DISPATCH, AVX2 or NEON");
        }
        const std::string mode = argv[1];
        if (mode == "DISPATCH") {
            check_reference_self_consistency();
            check_dispatch();
        } else if (mode == "AVX2") {
            check_reference_self_consistency();
            check_avx2();
        } else if (mode == "NEON") {
            check_reference_self_consistency();
            check_neon();
        } else {
            throw std::invalid_argument("Unknown mode: " + mode);
        }
    } catch (const SkipTest& skip) {
        std::cout << "SKIP: " << skip.what() << '\n';
        return 77;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
