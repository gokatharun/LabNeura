#include "labneura/backends/common/backend_factory.h"
#include "labneura/backends/common/cpu_features.h"
#include "labneura/tensor.h"
#include "test_environment.h"
#if defined(LABNEURA_TEST_AVX2)
#include "labneura/backends/cpu/avx2.h"
#endif
#if defined(LABNEURA_TEST_NEON)
#include "labneura/backends/cpu/neon.h"
#endif

#if defined(LABNEURA_TEST_AVX2) && \
    (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#endif

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class SkipTest : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::vector<int> make_multiplicands(std::size_t size, bool rhs) {
    std::vector<int> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        const std::size_t lane = i % 32;
        if (!rhs) {
            if (lane == 0) values[i] = -16;
            else if (lane == 1) values[i] = 100;
            else if (lane == 2) values[i] = -100;
            else if (lane == 3) values[i] = 127;
            else if (lane == 4) values[i] = -128;
            else values[i] = static_cast<int>(lane) - 16;
        } else {
            if (lane == 0) values[i] = 0;
            else if (lane == 1 || lane == 2 || lane == 3 || lane == 4) {
                values[i] = 2;
            } else if (lane == 5) {
                values[i] = -3;
            } else if (lane == 6) {
                values[i] = 4;
            } else {
                values[i] = static_cast<int>((lane * 7) % 13) - 6;
            }
        }
    }
    return values;
}

std::vector<int> expected_multiplication(const std::vector<int>& lhs,
                                         const std::vector<int>& rhs) {
    std::vector<int> expected(lhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        expected[i] = std::clamp(lhs[i] * rhs[i], -128, 127);
    }
    return expected;
}

std::vector<int> make_subtrahend(std::size_t size, bool rhs) {
    std::vector<int> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = rhs ? static_cast<int>((i * 13) % 91) - 45
                        : static_cast<int>((i * 17) % 101) - 50;
    }
    if (size > 0) {
        values[0] = rhs ? -1 : 127;
    }
    if (size > 1) {
        values[1] = rhs ? 1 : -128;
    }
    if (size > 8 && size % 8 != 0) {
        values.back() = rhs ? 1 : -128;
    }
    return values;
}

std::vector<int> expected_subtraction(const std::vector<int>& lhs,
                                      const std::vector<int>& rhs) {
    std::vector<int> expected(lhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        expected[i] = std::clamp(lhs[i] - rhs[i], -128, 127);
    }
    return expected;
}

#if defined(LABNEURA_TEST_AVX2) && \
    (defined(__GNUC__) || defined(__clang__))
bool runtime_supports_avx2() {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return false;
    }

    constexpr unsigned int avx = 1U << 28;
    constexpr unsigned int osxsave = 1U << 27;
    if ((ecx & (avx | osxsave)) != (avx | osxsave)) {
        return false;
    }

    unsigned int xcr0_low, xcr0_high;
    __asm__ volatile("xgetbv" : "=a"(xcr0_low), "=d"(xcr0_high) : "c"(0));
    if ((xcr0_low & 0x6U) != 0x6U) {
        return false;
    }

    return __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) &&
           (ebx & bit_AVX2) != 0;
}
#endif

void check_values(const int8_t* actual, const std::vector<int>& expected,
                  const std::string& context) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error(
                context + ": element " + std::to_string(i) + " expected " +
                std::to_string(expected[i]) + ", got " +
                std::to_string(static_cast<int>(actual[i])));
        }
    }
}

void check_avx2_multiplication() {
#if !defined(LABNEURA_TEST_AVX2)
    std::cout << "SKIP: AVX2 backend was not compiled for this target\n";
    throw SkipTest("AVX2 backend was not compiled for this target");
#elif !defined(__GNUC__) && !defined(__clang__)
    std::cout << "SKIP: AVX2 runtime detection is unavailable for this compiler\n";
    throw SkipTest("AVX2 runtime detection is unavailable for this compiler");
#else
    if (!runtime_supports_avx2()) {
        std::cout << "SKIP: CPU/OS does not support AVX2 state\n";
        throw SkipTest("CPU/OS does not support AVX2 state");
    }

    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", "AVX2");
    const std::vector<std::size_t> sizes = {
        1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 96, 97};
    for (const std::size_t size : sizes) {
        const auto lhs = make_multiplicands(size, false);
        const auto rhs = make_multiplicands(size, true);
        const auto expected = expected_multiplication(lhs, rhs);

        auto lhs_backend = labneura::create_best_backend(
            size, labneura::QuantizationMode::INT8);
        auto rhs_backend = labneura::create_best_backend(
            size, labneura::QuantizationMode::INT8);
        if (dynamic_cast<labneura::AVX2Backend*>(lhs_backend.get()) == nullptr ||
            dynamic_cast<labneura::AVX2Backend*>(rhs_backend.get()) == nullptr) {
            throw std::runtime_error("AVX2 override did not select AVX2 backend");
        }
        std::copy(lhs.begin(), lhs.end(), lhs_backend->data_int8());
        std::copy(rhs.begin(), rhs.end(), rhs_backend->data_int8());
        lhs_backend->mul_inplace(*rhs_backend);
        check_values(lhs_backend->data_int8(), expected,
                     "AVX2 direct multiplication size " +
                         std::to_string(size));

        labneura::Tensor inplace(lhs, labneura::QuantizationMode::INT8);
        const labneura::Tensor rhs_tensor(rhs, labneura::QuantizationMode::INT8);
        inplace.mul_inplace(rhs_tensor);
        check_values(inplace.data_int8(), expected,
                     "AVX2 Tensor in-place multiplication size " +
                         std::to_string(size));

        const labneura::Tensor lhs_tensor(lhs, labneura::QuantizationMode::INT8);
        const labneura::Tensor allocated = lhs_tensor.mul(rhs_tensor);
        check_values(allocated.data_int8(), expected,
                     "AVX2 allocating multiplication size " +
                         std::to_string(size));
    }
#endif
}

void check_neon_subtraction() {
#if !defined(LABNEURA_TEST_NEON)
    std::cout << "SKIP: NEON backend was not compiled for this target\n";
    throw SkipTest("NEON backend was not compiled for this target");
#else
    if (!labneura::cpu_supports_neon()) {
        std::cout << "SKIP: CPU does not support NEON\n";
        throw SkipTest("CPU does not support NEON");
    }

    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", "NEON");
    const std::vector<std::size_t> sizes = {
        1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129};
    for (const std::size_t size : sizes) {
        const auto lhs = make_subtrahend(size, false);
        const auto rhs = make_subtrahend(size, true);
        const auto expected = expected_subtraction(lhs, rhs);

        auto lhs_backend = labneura::create_best_backend(
            size, labneura::QuantizationMode::INT8);
        auto rhs_backend = labneura::create_best_backend(
            size, labneura::QuantizationMode::INT8);
        if (dynamic_cast<labneura::NEONBackend*>(lhs_backend.get()) == nullptr ||
            dynamic_cast<labneura::NEONBackend*>(rhs_backend.get()) == nullptr) {
            throw std::runtime_error("NEON override did not select NEON backend");
        }
        std::copy(lhs.begin(), lhs.end(), lhs_backend->data_int8());
        std::copy(rhs.begin(), rhs.end(), rhs_backend->data_int8());
        lhs_backend->sub_inplace(*rhs_backend);
        check_values(lhs_backend->data_int8(), expected,
                     "NEON direct subtraction size " +
                         std::to_string(size));

        labneura::Tensor inplace(lhs, labneura::QuantizationMode::INT8);
        const labneura::Tensor rhs_tensor(rhs, labneura::QuantizationMode::INT8);
        inplace.sub_inplace(rhs_tensor);
        check_values(inplace.data_int8(), expected,
                     "NEON Tensor in-place subtraction size " +
                         std::to_string(size));

        const labneura::Tensor lhs_tensor(lhs, labneura::QuantizationMode::INT8);
        const labneura::Tensor allocated = lhs_tensor.sub(rhs_tensor);
        check_values(allocated.data_int8(), expected,
                     "NEON allocating subtraction size " +
                         std::to_string(size));
    }
#endif
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::invalid_argument("Expected AVX2 or NEON test selection");
        }
        const std::string backend = argv[1];
        if (backend == "AVX2") {
            check_avx2_multiplication();
        } else if (backend == "NEON") {
            check_neon_subtraction();
        } else {
            throw std::invalid_argument("Unknown backend: " + backend);
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
