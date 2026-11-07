#include "labneura/backends/common/backend_factory.h"
#include "labneura/backends/common/cpu_features.h"
#include "labneura/tensor.h"
#include "test_environment.h"

#include <cmath>
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
    if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void require_throws(Function&& function, const std::string& context) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(context + ": expected exception");
}

void check_values(const labneura::Tensor& tensor,
                  const std::vector<float>& expected,
                  const std::string& context) {
    require(tensor.size() == expected.size(), context + ": numel mismatch");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(std::fabs(tensor.data_fp32()[i] - expected[i]) < 1e-5f,
                context + ": value mismatch at index " + std::to_string(i));
    }
}

void run_backend(const std::string& backend) {
    labneura::test::ScopedEnvironmentVariable override(
        "LABNEURA_BACKEND", backend);
    require(labneura::detect_backend() == backend,
            "Requested backend " + backend + " was not selected");

    const std::vector<float> values = [] {
        std::vector<float> result(24);
        for (std::size_t i = 0; i < result.size(); ++i) {
            result[i] = static_cast<float>(i + 1);
        }
        return result;
    }();
    const std::vector<std::size_t> shape{2, 3, 4};
    auto a = labneura::Tensor::from_flat(values, shape);
    auto b = labneura::Tensor::from_flat(
        std::vector<float>(values.size(), 2.0f), shape);

    require(a.shape() == shape && a.ndim() == 3 && a.numel() == 24,
            backend + ": nD shape metadata mismatch");
    for (std::size_t i = 0; i < values.size(); ++i) {
        require(a.data_fp32()[i] == values[i],
                backend + ": flat row-major data order changed");
    }

    const auto added = a.add(b);
    const auto subtracted = a.sub(b);
    const auto multiplied = a.mul(b);
    const auto divided = a.div(b);
    std::vector<float> expected(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        expected[i] = values[i] + 2.0f;
    }
    check_values(added, expected, backend + ": allocating add");
    for (std::size_t i = 0; i < values.size(); ++i) {
        expected[i] = values[i] - 2.0f;
    }
    check_values(subtracted, expected, backend + ": allocating sub");
    for (std::size_t i = 0; i < values.size(); ++i) {
        expected[i] = values[i] * 2.0f;
    }
    check_values(multiplied, expected, backend + ": allocating mul");
    for (std::size_t i = 0; i < values.size(); ++i) {
        expected[i] = values[i] / 2.0f;
    }
    check_values(divided, expected, backend + ": allocating div");

    auto in_place = labneura::Tensor::from_flat(values, shape);
    in_place.add_inplace(b);
    for (std::size_t i = 0; i < values.size(); ++i) expected[i] = values[i] + 2.0f;
    check_values(in_place, expected, backend + ": in-place add");
    in_place = labneura::Tensor::from_flat(values, shape);
    in_place.sub_inplace(b);
    for (std::size_t i = 0; i < values.size(); ++i) expected[i] = values[i] - 2.0f;
    check_values(in_place, expected, backend + ": in-place sub");
    in_place = labneura::Tensor::from_flat(values, shape);
    in_place.mul_inplace(b);
    for (std::size_t i = 0; i < values.size(); ++i) expected[i] = values[i] * 2.0f;
    check_values(in_place, expected, backend + ": in-place mul");
    in_place = labneura::Tensor::from_flat(values, shape);
    in_place.div_inplace(b);
    for (std::size_t i = 0; i < values.size(); ++i) expected[i] = values[i] / 2.0f;
    check_values(in_place, expected, backend + ": in-place div");

    auto output = labneura::Tensor::from_flat(
        std::vector<float>(values.size(), -1.0f), shape);
    a.add_into(b, output);
    for (std::size_t i = 0; i < values.size(); ++i) expected[i] = values[i] + 2.0f;
    check_values(output, expected, backend + ": add_into");

    const auto four_dimensional =
        labneura::Tensor::from_flat(values, {2, 2, 3, 2});
    const auto four_dimensional_twos = labneura::Tensor::from_flat(
        std::vector<float>(values.size(), 2.0f), {2, 2, 3, 2});
    check_values(four_dimensional.add(four_dimensional_twos), expected,
                 backend + ": four-dimensional add");
    std::vector<float> tail_values(29);
    for (std::size_t i = 0; i < tail_values.size(); ++i) {
        tail_values[i] = static_cast<float>(i + 1);
    }
    const auto tail_input = labneura::Tensor::from_flat(tail_values, {1, 29});
    const auto tail_ones = labneura::Tensor::from_flat(
        std::vector<float>(tail_values.size(), 1.0f), {1, 29});
    for (float& value : tail_values) value += 1.0f;
    check_values(tail_input.add(tail_ones), tail_values,
                 backend + ": vector-tail add");

    auto copy = a;
    copy.data_fp32()[0] = -100.0f;
    require(a.data_fp32()[0] == values[0],
            backend + ": copy did not own independent storage");

    auto same_numel_different_shape =
        labneura::Tensor::from_flat(values, {4, 6});
    require_throws<std::runtime_error>(
        [&] { (void)a.add(same_numel_different_shape); },
        backend + ": equal-numel shape mismatch");
    auto in_place_mismatch = labneura::Tensor::from_flat(values, shape);
    require_throws<std::runtime_error>(
        [&] { in_place_mismatch.add_inplace(same_numel_different_shape); },
        backend + ": in-place equal-numel shape mismatch");
    require_throws<std::invalid_argument>(
        [&] { a.add_into(b, same_numel_different_shape); },
        backend + ": add_into equal-numel shape mismatch");

    const auto empty = labneura::Tensor::from_flat(
        std::vector<float>{}, {2, 0, 7});
    require(empty.shape() == std::vector<std::size_t>({2, 0, 7}) &&
                empty.numel() == 0,
            backend + ": zero-sized dimension semantics mismatch");
    const auto scalar = labneura::Tensor::from_flat(
        std::vector<float>{42.0f}, std::vector<std::size_t>{});
    require(scalar.shape().empty() && scalar.ndim() == 0 &&
                scalar.numel() == 1 && scalar.data_fp32()[0] == 42.0f,
            backend + ": scalar shape semantics mismatch");

    const auto integer_tensor = labneura::Tensor::from_flat(
        std::vector<int>{7, -3, 9, 1}, {2, 2},
        labneura::QuantizationMode::INT8);
    require(integer_tensor.shape() == std::vector<std::size_t>({2, 2}) &&
                integer_tensor.data_int8()[1] == -3,
            backend + ": integer factory construction mismatch");
}

void check_invalid_inputs() {
    using labneura::QuantizationMode;
    require_throws<std::invalid_argument>(
        [] {
            (void)labneura::Tensor::from_flat(
                std::vector<float>{1.0f, 2.0f}, {3});
        },
        "shape/data length mismatch");
    require_throws<std::invalid_argument>(
        [] {
            (void)labneura::Tensor::from_flat(
                std::vector<float>{}, std::vector<std::size_t>{});
        },
        "empty scalar data");

    const auto max_size = std::numeric_limits<std::size_t>::max();
    require_throws<std::overflow_error>(
        [&] {
            (void)labneura::Tensor::from_flat(
                std::vector<float>{}, {max_size, 2});
        },
        "element count overflow");
    require_throws<std::overflow_error>(
        [&] {
            (void)labneura::Tensor::from_flat(
                std::vector<float>{}, {max_size / sizeof(float) + 1});
        },
        "byte size overflow");
    require_throws<std::invalid_argument>(
        [] {
            (void)labneura::Tensor::from_flat(
                std::vector<float>{1.0f}, std::vector<std::size_t>{},
                QuantizationMode::FP64);
        },
        "unsupported dtype");

    const auto empty_with_large_dimensions =
        labneura::Tensor::from_flat(
            std::vector<float>{}, {max_size, 0, max_size});
    require(empty_with_large_dimensions.numel() == 0,
            "zero dimension should yield zero elements");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Expected GENERIC, AVX2 or NEON");
        const std::string backend = argv[1];
        if (backend == "GENERIC") {
            run_backend(backend);
        } else if (backend == "AVX2") {
#if !defined(LABNEURA_TEST_AVX2)
            throw SkipTest("AVX2 backend was not compiled for this target");
#else
            if (!labneura::cpu_supports_avx2()) {
                throw SkipTest("CPU/OS does not support AVX2 state");
            }
            run_backend(backend);
#endif
        } else if (backend == "NEON") {
#if !defined(LABNEURA_TEST_NEON)
            throw SkipTest("NEON backend was not compiled for this target");
#else
            if (!labneura::cpu_supports_neon()) {
                throw SkipTest("CPU does not support NEON");
            }
            run_backend(backend);
#endif
        } else {
            throw std::invalid_argument("Unknown backend: " + backend);
        }
        check_invalid_inputs();
        std::cout << backend << ": nD Tensor tests passed\n";
        return 0;
    } catch (const SkipTest& skip) {
        std::cout << "SKIP: " << skip.what() << '\n';
        return 77;
    } catch (const std::exception& error) {
        std::cerr << "nD Tensor test failed: " << error.what() << '\n';
        return 1;
    }
}
