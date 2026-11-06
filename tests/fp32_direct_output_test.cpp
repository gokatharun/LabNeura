#include "labneura/backends/common/backend_factory.h"
#include "labneura/backends/common/cpu_features.h"
#include "labneura/backends/cpu/generic.h"
#include "labneura/tensor.h"
#include "test_environment.h"
#if defined(LABNEURA_TEST_AVX2)
#include "labneura/backends/cpu/avx2.h"
#endif
#if defined(LABNEURA_TEST_NEON)
#include "labneura/backends/cpu/neon.h"
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
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

const float INF = std::numeric_limits<float>::infinity();
const float QNAN = std::numeric_limits<float>::quiet_NaN();
const float MIN_SUBNORMAL = std::numeric_limits<float>::denorm_min();
const float MIN_NORMAL = std::numeric_limits<float>::min();
const float MAX_FINITE = std::numeric_limits<float>::max();

const std::vector<float>& special_values() {
    static const std::vector<float> values = {
        0.0f, -0.0f, INF, -INF, QNAN, MIN_SUBNORMAL, -MIN_SUBNORMAL,
        MIN_NORMAL, -MIN_NORMAL, MAX_FINITE, -MAX_FINITE, 1.0f, -1.0f,
        3.0f, 0.1f, 1.0e-20f, 1.0e20f,
    };
    return values;
}

// Expected value computed in double. For float operands, rounding the double result to
// float equals the correctly rounded float result (double has more than 2p + 2 bits),
// so this reference does not share any code with the kernels under test.
float expected(float a, float b, labneura::OperationType op) {
    const double x = a;
    const double y = b;
    double r = 0.0;
    switch (op) {
        case labneura::ADD: r = x + y; break;
        case labneura::SUB: r = x - y; break;
        case labneura::MUL: r = x * y; break;
        default: r = x / y; break;
    }
    // Converting a finite double outside the float range is undefined behavior, so
    // overflow is handled explicitly. 0x1.ffffffp127 is the midpoint between the
    // largest finite float and 2^128; round-to-nearest-even sends it (and above) to
    // infinity.
    if (std::isfinite(r) && std::fabs(r) >= 0x1.ffffffp127) {
        return r < 0 ? -INF : INF;
    }
    return static_cast<float>(r);
}

bool same_float(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) {
        return std::isnan(a) && std::isnan(b);
    }
    uint32_t ba;
    uint32_t bb;
    std::memcpy(&ba, &a, sizeof(a));
    std::memcpy(&bb, &b, sizeof(b));
    return ba == bb;  // distinguishes +0 / -0
}

// Distinct finite values per lane, with special values interleaved at varying lanes.
void make_inputs(std::size_t n, std::vector<float>& a, std::vector<float>& b) {
    a.resize(n);
    b.resize(n);
    const auto& sp = special_values();
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = 1.5f + static_cast<float>(i) * 0.37f - (i % 3 == 0 ? 7.0f : 0.0f);
        b[i] = 2.25f + static_cast<float>(i) * 0.11f + (i % 4 == 0 ? -9.0f : 0.0f);
        if (n <= 4096 || i % 97 == 0) {
            if (i % 5 == 1) a[i] = sp[(i * 7) % sp.size()];
            if (i % 7 == 2) b[i] = sp[(i * 3 + 1) % sp.size()];
        }
    }
}

const char* op_name(labneura::OperationType op) {
    switch (op) {
        case labneura::ADD: return "ADD";
        case labneura::SUB: return "SUB";
        case labneura::MUL: return "MUL";
        default: return "DIV";
    }
}

labneura::Tensor apply(const labneura::Tensor& a, const labneura::Tensor& b,
                       labneura::OperationType op) {
    switch (op) {
        case labneura::ADD: return a.add(b);
        case labneura::SUB: return a.sub(b);
        case labneura::MUL: return a.mul(b);
        default: return a.div(b);
    }
}

void apply_inplace(labneura::Tensor& a, const labneura::Tensor& b, labneura::OperationType op) {
    switch (op) {
        case labneura::ADD: a.add_inplace(b); break;
        case labneura::SUB: a.sub_inplace(b); break;
        case labneura::MUL: a.mul_inplace(b); break;
        default: a.div_inplace(b); break;
    }
}

void check_values(const float* actual, const std::vector<float>& a, const std::vector<float>& b,
                  labneura::OperationType op, const std::string& what) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float want = expected(a[i], b[i], op);
        if (!same_float(actual[i], want)) {
            throw std::runtime_error(what + " " + op_name(op) + " mismatch at index " +
                                     std::to_string(i) + " of " + std::to_string(a.size()) +
                                     ": a=" + std::to_string(a[i]) + " b=" + std::to_string(b[i]) +
                                     " got=" + std::to_string(actual[i]) +
                                     " want=" + std::to_string(want));
        }
    }
}

void check_size(std::size_t n, const std::string& label) {
    std::vector<float> a_data;
    std::vector<float> b_data;
    make_inputs(n, a_data, b_data);
    const labneura::Tensor a(a_data);
    const labneura::Tensor b(b_data);

    for (auto op : {labneura::ADD, labneura::SUB, labneura::MUL, labneura::DIV}) {
        const std::string what = label + " n=" + std::to_string(n);

        const labneura::Tensor out = apply(a, b, op);
        require(out.size() == n, what + ": output size");
        require(out.shape() == a.shape(), what + ": output shape");
        require(out.data_fp32() != a.data_fp32() && out.data_fp32() != b.data_fp32(),
                what + ": output must not alias an input");
        check_values(out.data_fp32(), a_data, b_data, op, what + " allocating");

        // Inputs are unchanged, bit for bit.
        for (std::size_t i = 0; i < n; ++i) {
            require(same_float(a.data_fp32()[i], a_data[i]), what + ": input a modified");
            require(same_float(b.data_fp32()[i], b_data[i]), what + ": input b modified");
        }

        // The in-place kernel agrees with the same independent reference.
        labneura::Tensor inplace(a_data);
        apply_inplace(inplace, b, op);
        check_values(inplace.data_fp32(), a_data, b_data, op, what + " in-place");
        for (std::size_t i = 0; i < n; ++i) {
            require(same_float(out.data_fp32()[i], inplace.data_fp32()[i]),
                    what + ": allocating and in-place results differ");
        }
    }
}

const std::vector<std::size_t>& sizes() {
    static const std::vector<std::size_t> values = {
        1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1024, 4096, 16384, 65536, 1000000};
    return values;
}

// Aliasing at the backend boundary: allocating on self (a op a) must work and leave
// the operand intact.
void check_self_operand(const std::string& label) {
    std::vector<float> a_data;
    std::vector<float> unused;
    make_inputs(37, a_data, unused);
    const labneura::Tensor a(a_data);
    const labneura::Tensor out = a.add(a);
    check_values(out.data_fp32(), a_data, a_data, labneura::ADD, label + " self add");
    for (std::size_t i = 0; i < a_data.size(); ++i) {
        require(same_float(a.data_fp32()[i], a_data[i]), label + ": self operand modified");
    }
}

void check_shapes_and_errors(const std::string& label) {
    const labneura::Tensor m1(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}});
    const labneura::Tensor m2(std::vector<std::vector<float>>{{6, 5, 4}, {3, 2, 1}});
    const labneura::Tensor sum = m1.add(m2);
    require((sum.shape() == std::vector<std::size_t>{2, 3}), label + ": 2-D shape propagation");
    for (std::size_t i = 0; i < 6; ++i) {
        require(sum.data_fp32()[i] == 7.0f, label + ": 2-D values");
    }

    const labneura::Tensor transposed(std::vector<std::vector<float>>{{1, 2}, {3, 4}, {5, 6}});
    bool threw = false;
    try {
        m1.add(transposed);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, label + ": same-size different-shape must throw");

    threw = false;
    try {
        m1.add(labneura::Tensor(std::vector<float>{1, 2, 3}));
    } catch (const std::runtime_error& error) {
        threw = std::string(error.what()).find("size mismatch") != std::string::npos ||
                std::string(error.what()).find("shape mismatch") != std::string::npos;
    }
    require(threw, label + ": size mismatch must throw");

    // INT8 keeps using the unchanged clone + in-place path.
    const labneura::Tensor i1(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                              labneura::QuantizationMode::INT8);
    const labneura::Tensor i2(std::vector<float>{9, 8, 7, 6, 5, 4, 3, 2, 1},
                              labneura::QuantizationMode::INT8);
    const labneura::Tensor isum = i1.add(i2);
    for (std::size_t i = 0; i < 9; ++i) {
        require(isum.data_int8()[i] == 10, label + ": INT8 allocating add");
        require(i1.data_int8()[i] == static_cast<int8_t>(i + 1), label + ": INT8 input modified");
    }
}

void run_all(const std::string& label, const std::string& expected_backend) {
    require(labneura::detect_backend() == expected_backend,
            label + ": unexpected backend " + labneura::detect_backend());
    for (std::size_t n : sizes()) {
        check_size(n, label);
    }
    check_self_operand(label);
    check_shapes_and_errors(label);
    std::cout << label << ": all FP32 allocating/in-place checks passed\n";
}

// Direct-output support is reported through operation_new(): non-null means the kernel
// wrote straight into a freshly allocated backend, nullptr means the clone fallback.
template <typename Backend>
void check_direct_support(bool expect_fp32_direct, const std::string& label) {
    Backend a(33, labneura::QuantizationMode::FP32);
    Backend b(33, labneura::QuantizationMode::FP32);
    for (std::size_t i = 0; i < 33; ++i) {
        a.data_fp32()[i] = static_cast<float>(i);
        b.data_fp32()[i] = 2.0f;
    }
    const auto out = a.operation_new(b, labneura::MUL);
    require((out != nullptr) == expect_fp32_direct, label + ": FP32 direct-output availability");
    if (out) {
        require(out->size() == 33 && out->quantization_mode() == labneura::QuantizationMode::FP32,
                label + ": output metadata");
        for (std::size_t i = 0; i < 33; ++i) {
            require(out->data_fp32()[i] == 2.0f * static_cast<float>(i), label + ": direct values");
            require(a.data_fp32()[i] == static_cast<float>(i), label + ": lhs modified");
            require(b.data_fp32()[i] == 2.0f, label + ": rhs modified");
        }
    }

    // Other dtypes must keep using the existing path.
    Backend i8a(8, labneura::QuantizationMode::INT8);
    Backend i8b(8, labneura::QuantizationMode::INT8);
    require(i8a.operation_new(i8b, labneura::ADD) == nullptr,
            label + ": INT8 must not use direct output");

    // Validation errors are still reported.
    Backend wrong_size(7, labneura::QuantizationMode::FP32);
    bool threw = false;
    try {
        a.operation_new(wrong_size, labneura::ADD);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, label + ": size mismatch must throw");
}

void check_generic() {
    labneura::test::ScopedEnvironmentVariable backend_override("LABNEURA_BACKEND", "GENERIC");
    run_all("GENERIC", "GENERIC");
    // Generic has no direct-output kernel: it exercises the compatibility fallback.
    check_direct_support<labneura::GenericBackend>(false, "GENERIC");
}

void check_avx2() {
#if !defined(LABNEURA_TEST_AVX2)
    throw SkipTest("AVX2 backend was not compiled for this target");
#else
    if (!labneura::cpu_supports_avx2()) {
        throw SkipTest("CPU/OS does not support AVX2 state");
    }
    labneura::test::ScopedEnvironmentVariable backend_override("LABNEURA_BACKEND", "AVX2");
    run_all("AVX2", "AVX2");
    check_direct_support<labneura::AVX2Backend>(true, "AVX2");
#endif
}

void check_neon() {
#if !defined(LABNEURA_TEST_NEON)
    throw SkipTest("NEON backend was not compiled for this target");
#else
    if (!labneura::cpu_supports_neon()) {
        throw SkipTest("CPU does not support NEON");
    }
    labneura::test::ScopedEnvironmentVariable backend_override("LABNEURA_BACKEND", "NEON");
    run_all("NEON", "NEON");
    check_direct_support<labneura::NEONBackend>(true, "NEON");
#endif
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::invalid_argument("Expected GENERIC, AVX2 or NEON");
        }
        const std::string mode = argv[1];
        if (mode == "GENERIC") {
            check_generic();
        } else if (mode == "AVX2") {
            check_avx2();
        } else if (mode == "NEON") {
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
