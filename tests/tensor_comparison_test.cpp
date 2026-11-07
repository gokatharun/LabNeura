#include "labneura/backends/common/backend_factory.h"
#include "labneura/backends/common/cpu_features.h"
#include "labneura/tensor.h"
#include "test_environment.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(LABNEURA_TEST_AVX2)
#include "labneura/backends/cpu/avx2.h"
#endif

#if defined(LABNEURA_TEST_NEON)
#include "labneura/backends/cpu/neon.h"
#endif

namespace {

using ComparisonMethod =
    labneura::Tensor (labneura::Tensor::*)(const labneura::Tensor&) const;

struct ComparisonCase {
    labneura::ComparisonType operation;
    ComparisonMethod method;
    const char* name;
};

constexpr ComparisonCase kComparisons[] = {
    {labneura::ComparisonType::EQ, &labneura::Tensor::eq, "eq"},
    {labneura::ComparisonType::NE, &labneura::Tensor::ne, "ne"},
    {labneura::ComparisonType::LT, &labneura::Tensor::lt, "lt"},
    {labneura::ComparisonType::LE, &labneura::Tensor::le, "le"},
    {labneura::ComparisonType::GT, &labneura::Tensor::gt, "gt"},
    {labneura::ComparisonType::GE, &labneura::Tensor::ge, "ge"},
};

class SkipTest : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void require_throws(Function&& function, const std::string& context,
                    const std::string& expected_message = {}) {
    try {
        function();
    } catch (const Exception& error) {
        if (!expected_message.empty()) {
            require(std::string(error.what()).find(expected_message) !=
                        std::string::npos,
                    context + ": exception message did not contain '" +
                        expected_message + "'");
        }
        return;
    }
    throw std::runtime_error(context + ": expected exception");
}

void require_bool_result(const labneura::Tensor& result,
                         const std::vector<std::size_t>& shape,
                         const std::vector<std::uint8_t>& expected,
                         const std::string& context) {
    require(result.quantization_mode() == labneura::QuantizationMode::BOOL,
            context + ": result dtype was not BOOL");
    require(result.shape() == shape, context + ": result shape mismatch");
    require(result.numel() == expected.size(),
            context + ": result element count mismatch");
    require(result.is_contiguous(), context + ": result is not contiguous");
    require(result.storage_offset() == 0,
            context + ": result has a nonzero storage offset");

    const std::uint8_t* actual = result.data_bool();
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(actual[i] == expected[i] && actual[i] <= 1,
                context + ": BOOL value mismatch at index " +
                    std::to_string(i) + " (expected " +
                    std::to_string(expected[i]) + ", got " +
                    std::to_string(actual[i]) + ")");
    }
}

template <typename Value>
bool reference_compare(labneura::ComparisonType comparison, Value lhs,
                       Value rhs) {
    switch (comparison) {
        case labneura::ComparisonType::EQ: return lhs == rhs;
        case labneura::ComparisonType::NE: return lhs != rhs;
        case labneura::ComparisonType::LT: return lhs < rhs;
        case labneura::ComparisonType::LE: return lhs <= rhs;
        case labneura::ComparisonType::GT: return lhs > rhs;
        case labneura::ComparisonType::GE: return lhs >= rhs;
    }
    throw std::invalid_argument("Unknown reference comparison");
}

template <typename Value>
std::vector<std::uint8_t> reference_equal_shape(
    labneura::ComparisonType comparison, const std::vector<Value>& lhs,
    const std::vector<Value>& rhs) {
    require(lhs.size() == rhs.size(), "Invalid equal-shape reference inputs");
    std::vector<std::uint8_t> expected(lhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        expected[i] =
            reference_compare(comparison, lhs[i], rhs[i]) ? 1 : 0;
    }
    return expected;
}

template <typename Value>
std::vector<std::uint8_t> reference_matrix_row(
    labneura::ComparisonType comparison, const std::vector<Value>& matrix,
    const std::vector<Value>& row, std::size_t columns) {
    std::vector<std::uint8_t> expected(matrix.size());
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        expected[i] =
            reference_compare(comparison, matrix[i], row[i % columns]) ? 1 : 0;
    }
    return expected;
}

template <typename Value>
void test_dtype_comparisons(const std::vector<Value>& lhs_values,
                            const std::vector<Value>& rhs_values,
                            const std::vector<Value>& matrix_values,
                            const std::vector<Value>& row_values,
                            labneura::QuantizationMode dtype,
                            const std::string& dtype_name) {
    using labneura::Tensor;
    Tensor lhs;
    Tensor rhs;
    Tensor matrix;
    Tensor row;
    if constexpr (std::is_floating_point_v<Value>) {
        std::vector<float> lhs_f(lhs_values.begin(), lhs_values.end());
        std::vector<float> rhs_f(rhs_values.begin(), rhs_values.end());
        std::vector<float> matrix_f(matrix_values.begin(), matrix_values.end());
        std::vector<float> row_f(row_values.begin(), row_values.end());
        lhs = Tensor::from_flat(lhs_f, {lhs_f.size()}, dtype);
        rhs = Tensor::from_flat(rhs_f, {rhs_f.size()}, dtype);
        matrix = Tensor::from_flat(matrix_f, {2, 3}, dtype);
        row = Tensor::from_flat(row_f, {3}, dtype);
    } else {
        std::vector<int> lhs_i(lhs_values.begin(), lhs_values.end());
        std::vector<int> rhs_i(rhs_values.begin(), rhs_values.end());
        std::vector<int> matrix_i(matrix_values.begin(), matrix_values.end());
        std::vector<int> row_i(row_values.begin(), row_values.end());
        lhs = Tensor::from_flat(lhs_i, {lhs_i.size()}, dtype);
        rhs = Tensor::from_flat(rhs_i, {rhs_i.size()}, dtype);
        matrix = Tensor::from_flat(matrix_i, {2, 3}, dtype);
        row = Tensor::from_flat(row_i, {3}, dtype);
    }

    for (const ComparisonCase& test : kComparisons) {
        const auto equal_expected =
            reference_equal_shape(test.operation, lhs_values, rhs_values);
        const Tensor equal_result = (lhs.*(test.method))(rhs);
        require_bool_result(equal_result, {lhs_values.size()}, equal_expected,
                            dtype_name + " " + test.name + " equal-shape");

        const auto broadcast_expected =
            reference_matrix_row(test.operation, matrix_values, row_values, 3);
        const Tensor broadcast_result = (matrix.*(test.method))(row);
        require_bool_result(broadcast_result, {2, 3}, broadcast_expected,
                            dtype_name + " " + test.name + " broadcast");
    }
}

void test_supported_numeric_dtypes() {
    using labneura::QuantizationMode;
    test_dtype_comparisons<float>(
        {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f},
        {-1.0f, -1.0f, 0.0f, -2.0f, 4.0f},
        {1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f}, {2.0f, 5.0f, 9.0f},
        QuantizationMode::FP32, "FP32");
    test_dtype_comparisons<float>(
        {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f},
        {-1.0f, -1.0f, 0.0f, -2.0f, 4.0f},
        {1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f}, {2.0f, 5.0f, 9.0f},
        QuantizationMode::FP16, "FP16");
    test_dtype_comparisons<int>(
        {-128, -1, 0, 1, 127}, {-127, -1, 0, -2, 126},
        {-128, -1, 0, 1, 126, 127}, {-127, -1, 1},
        QuantizationMode::INT8, "INT8");
    test_dtype_comparisons<int>(
        {-32768, -1, 0, 1, 32767}, {-32767, -1, 0, -2, 32766},
        {-32768, -1, 0, 1, 32766, 32767}, {-32767, -1, 1},
        QuantizationMode::INT16, "INT16");
}

void test_scalar_and_higher_rank_broadcasts() {
    using labneura::QuantizationMode;
    using labneura::Tensor;

    const Tensor scalar =
        Tensor::from_flat(std::vector<float>{2.0f}, {}, QuantizationMode::FP32);
    const Tensor values =
        Tensor::from_flat(std::vector<float>{1, 2, 3, 4}, {2, 2});
    require_bool_result(scalar.lt(values), {2, 2}, {0, 0, 1, 1},
                        "Scalar versus nD broadcast");
    require_bool_result(values.eq(scalar), {2, 2}, {0, 1, 0, 0},
                        "nD versus scalar broadcast");

    const Tensor lhs = Tensor::from_flat(
        std::vector<int>{1, 4, 7, 2}, {2, 1, 2, 1},
        QuantizationMode::INT16);
    const Tensor rhs = Tensor::from_flat(
        std::vector<int>{0, 4, 8, 2, 5, 6, 9, 3}, {1, 2, 1, 4},
        QuantizationMode::INT16);
    const Tensor result = lhs.lt(rhs);
    std::vector<std::uint8_t> expected;
    expected.reserve(32);
    const int lhs_values[] = {1, 4, 7, 2};
    const int rhs_values[] = {0, 4, 8, 2, 5, 6, 9, 3};
    for (int batch = 0; batch < 2; ++batch) {
        for (int channel = 0; channel < 2; ++channel) {
            for (int height = 0; height < 2; ++height) {
                for (int width = 0; width < 4; ++width) {
                    expected.push_back(
                        lhs_values[batch * 2 + height] <
                                rhs_values[channel * 4 + width]
                            ? 1
                            : 0);
                }
            }
        }
    }
    require_bool_result(result, {2, 2, 2, 4}, expected,
                        "Four-dimensional broadcast");

    const Tensor lhs3 = Tensor::from_flat(
        std::vector<int>{1, 3, 5, 7, 9, 11}, {2, 1, 3},
        QuantizationMode::INT8);
    const Tensor rhs3 = Tensor::from_flat(
        std::vector<int>{0, 4, 8, 2}, {1, 4, 1},
        QuantizationMode::INT8);
    const Tensor result3 = lhs3.le(rhs3);
    std::vector<std::uint8_t> expected3;
    for (std::size_t batch = 0; batch < 2; ++batch) {
        for (const int rhs_value : {0, 4, 8, 2}) {
            for (std::size_t column = 0; column < 3; ++column) {
                const int lhs_value = static_cast<int>(
                    lhs3.data_int8()[batch * 3 + column]);
                expected3.push_back(lhs_value <= rhs_value ? 1 : 0);
            }
        }
    }
    require_bool_result(result3, {2, 4, 3}, expected3,
                        "Three-dimensional broadcast");
}

void test_float_special_values() {
    using labneura::QuantizationMode;
    using labneura::Tensor;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    const std::vector<float> lhs_values{nan, -infinity, infinity, -0.0f, 0.0f};
    const std::vector<float> rhs_values{nan, -infinity, infinity, 0.0f, -0.0f};
    const Tensor lhs32 =
        Tensor::from_flat(lhs_values, {5}, QuantizationMode::FP32);
    const Tensor rhs32 =
        Tensor::from_flat(rhs_values, {5}, QuantizationMode::FP32);
    const Tensor lhs16 =
        Tensor::from_flat(lhs_values, {5}, QuantizationMode::FP16);
    const Tensor rhs16 =
        Tensor::from_flat(rhs_values, {5}, QuantizationMode::FP16);
    require(std::isnan(lhs32.data_fp32()[0]) &&
                std::isnan(rhs32.data_fp32()[0]) &&
                !(lhs32.data_fp32()[0] < rhs32.data_fp32()[0]),
            "FP32 NaN test inputs were not preserved");
    const std::vector<std::vector<std::uint8_t>> expected{
        {0, 1, 1, 1, 1},
        {1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 0},
        {0, 1, 1, 1, 1},
    };
    for (std::size_t i = 0; i < std::size(kComparisons); ++i) {
        require_bool_result((lhs32.*(kComparisons[i].method))(rhs32), {5},
                            expected[i],
                            std::string("FP32 special values ") +
                                kComparisons[i].name);
        require_bool_result((lhs16.*(kComparisons[i].method))(rhs16), {5},
                            expected[i],
                            std::string("FP16 special values ") +
                                kComparisons[i].name);
    }
}

void test_zero_size_views_and_output_independence() {
    using labneura::QuantizationMode;
    using labneura::Tensor;

    const Tensor empty =
        Tensor::from_flat(std::vector<float>{}, {0, 3}, QuantizationMode::FP32);
    const Tensor row =
        Tensor::from_flat(std::vector<float>{1, 2, 3}, {1, 3});
    for (const ComparisonCase& test : kComparisons) {
        const Tensor result = (empty.*(test.method))(row);
        require_bool_result(result, {0, 3}, {},
                            std::string("Zero-sized ") + test.name);
    }

    Tensor parent = Tensor::from_flat(
        std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {4, 3});
    const Tensor original(parent);
    const Tensor view = parent.slice_first_axis(1, 3);
    require(view.storage_offset() == 3,
            "Comparison view did not have a nonzero storage offset");
    const Tensor threshold =
        Tensor::from_flat(std::vector<float>{4, 5, 6}, {3});
    Tensor less = view.lt(threshold);
    require_bool_result(less, {2, 3}, {1, 1, 1, 0, 0, 0},
                        "Offset view comparison");
    require(std::equal(parent.data_fp32(),
                       parent.data_fp32() + parent.size(),
                       original.data_fp32()),
            "Comparison mutated an input view");

    std::uint8_t* result_data = less.data_bool();
    result_data[0] = 0;
    require(std::equal(parent.data_fp32(),
                       parent.data_fp32() + parent.size(),
                       original.data_fp32()) &&
                view.data_fp32()[0] == original.data_fp32()[3],
            "Comparison output aliases an input");
}

template <typename Value>
std::array<std::vector<std::uint8_t>, std::size(kComparisons)>
comparison_results_for_backend(
    const std::string& backend, const std::vector<Value>& lhs_values,
    const std::vector<Value>& rhs_values,
    const std::vector<std::size_t>& lhs_shape,
    const std::vector<std::size_t>& rhs_shape,
    const std::vector<std::size_t>& output_shape,
    labneura::QuantizationMode dtype) {
    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", backend);
    require(labneura::detect_backend() == backend,
            backend + " backend was not explicitly selected for parity test");

    labneura::Tensor lhs;
    labneura::Tensor rhs;
    if constexpr (std::is_floating_point_v<Value>) {
        lhs = labneura::Tensor::from_flat(
            std::vector<float>(lhs_values.begin(), lhs_values.end()),
            lhs_shape, dtype);
        rhs = labneura::Tensor::from_flat(
            std::vector<float>(rhs_values.begin(), rhs_values.end()),
            rhs_shape, dtype);
    } else {
        lhs = labneura::Tensor::from_flat(
            std::vector<int>(lhs_values.begin(), lhs_values.end()), lhs_shape,
            dtype);
        rhs = labneura::Tensor::from_flat(
            std::vector<int>(rhs_values.begin(), rhs_values.end()), rhs_shape,
            dtype);
    }

    std::array<std::vector<std::uint8_t>, std::size(kComparisons)> results;
    for (std::size_t i = 0; i < std::size(kComparisons); ++i) {
        const labneura::Tensor result =
            (lhs.*(kComparisons[i].method))(rhs);
        require(result.shape() == output_shape,
                backend + " " + kComparisons[i].name +
                    " parity output shape mismatch");
        const std::uint8_t* values = result.data_bool();
        results[i].assign(values, values + result.numel());
    }
    return results;
}

template <typename Value>
void test_backend_parity_for_dtype(
    const std::vector<Value>& lhs_values,
    const std::vector<Value>& rhs_values,
    labneura::QuantizationMode dtype, const std::string& dtype_name) {
    for (const std::size_t count : {3, 15, 16, 17, 31, 32, 33}) {
        std::vector<Value> lhs_boundary;
        std::vector<Value> rhs_boundary;
        lhs_boundary.reserve(count);
        rhs_boundary.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            lhs_boundary.push_back(lhs_values[i % lhs_values.size()]);
            rhs_boundary.push_back(rhs_values[i % rhs_values.size()]);
        }
        const auto generic_boundary = comparison_results_for_backend(
            "GENERIC", lhs_boundary, rhs_boundary, {count}, {count}, {count},
            dtype);
        const auto neon_boundary = comparison_results_for_backend(
            "NEON", lhs_boundary, rhs_boundary, {count}, {count}, {count},
            dtype);
        for (std::size_t i = 0; i < std::size(kComparisons); ++i) {
            require(neon_boundary[i] == generic_boundary[i],
                    dtype_name + " NEON/" + kComparisons[i].name +
                        " vector-boundary parity mismatch at " +
                        std::to_string(count));
        }
    }

    const auto generic_equal = comparison_results_for_backend(
        "GENERIC", lhs_values, rhs_values, {lhs_values.size()},
        {rhs_values.size()}, {lhs_values.size()}, dtype);
    const auto neon_equal = comparison_results_for_backend(
        "NEON", lhs_values, rhs_values, {lhs_values.size()},
        {rhs_values.size()}, {lhs_values.size()}, dtype);
    for (std::size_t i = 0; i < std::size(kComparisons); ++i) {
        require(neon_equal[i] == generic_equal[i],
                dtype_name + " NEON/" + kComparisons[i].name +
                    " equal-shape parity mismatch");
    }

    const std::vector<Value> broadcast_lhs{lhs_values[0], lhs_values[1],
                                           lhs_values[2], lhs_values[3],
                                           lhs_values[4], lhs_values[5]};
    const std::vector<Value> broadcast_rhs{rhs_values[0], rhs_values[1],
                                           rhs_values[2]};
    const auto generic_broadcast = comparison_results_for_backend(
        "GENERIC", broadcast_lhs, broadcast_rhs, {2, 3}, {3}, {2, 3},
        dtype);
    const auto neon_broadcast = comparison_results_for_backend(
        "NEON", broadcast_lhs, broadcast_rhs, {2, 3}, {3}, {2, 3}, dtype);
    for (std::size_t i = 0; i < std::size(kComparisons); ++i) {
        require(neon_broadcast[i] == generic_broadcast[i],
                dtype_name + " NEON/" + kComparisons[i].name +
                    " broadcast parity mismatch");
    }
}

void test_generic_neon_parity() {
    using labneura::QuantizationMode;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    test_backend_parity_for_dtype<float>(
        {nan, -infinity, -3, -1, -0.0f, 0.0f, 1, 2, 3, 4, 5, 6, 7, 8, 9,
         10, infinity},
        {nan, -infinity, -2, -1, 0.0f, -0.0f, 2, 2, 4, 3, 5, 7, 6, 9, 8,
         11, infinity},
        QuantizationMode::FP32, "FP32");
    test_backend_parity_for_dtype<float>(
        {nan, -infinity, -3, -1, -0.0f, 0.0f, 1, 2, 3, 4, 5, 6, 7, 8, 9,
         10, infinity},
        {nan, -infinity, -2, -1, 0.0f, -0.0f, 2, 2, 4, 3, 5, 7, 6, 9, 8,
         11, infinity},
        QuantizationMode::FP16, "FP16");
    test_backend_parity_for_dtype<int>(
        {-128, -127, -100, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 126, 127,
         -128},
        {127, -128, -99, -2, 0, 1, 0, 2, 4, 3, 5, 7, 6, 9, 127, 126, 127},
        QuantizationMode::INT8, "INT8");
    test_backend_parity_for_dtype<int>(
        {-32768, -32767, -100, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 32766,
         32767, -32768},
        {32767, -32768, -99, -2, 0, 1, 0, 2, 4, 3, 5, 7, 6, 9, 32767,
         32766, 32767},
        QuantizationMode::INT16, "INT16");
}

void test_errors_and_bool_input_policy() {
    using labneura::QuantizationMode;
    using labneura::Tensor;
    const Tensor lhs = Tensor::from_flat(std::vector<float>{1, 2, 3, 4}, {2, 2});
    const Tensor incompatible =
        Tensor::from_flat(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
    require_throws<std::runtime_error>(
        [&] { (void)lhs.eq(incompatible); }, "Incompatible comparison shapes",
        "Tensor shape mismatch");

    const Tensor different_dtype = Tensor::from_flat(
        std::vector<int>{1, 2, 3, 4}, {2, 2}, QuantizationMode::INT8);
    require_throws<std::runtime_error>(
        [&] { (void)lhs.eq(different_dtype); }, "Comparison dtype mismatch",
        "Quantization mode mismatch");

    const Tensor bool_lhs =
        Tensor::from_flat(std::vector<std::uint8_t>{0, 1}, {2});
    const Tensor bool_rhs =
        Tensor::from_flat(std::vector<std::uint8_t>{1, 1}, {2});
    require_throws<std::runtime_error>(
        [&] { (void)bool_lhs.eq(bool_rhs); }, "BOOL input comparison policy",
        "not supported for BOOL");
}

void run_generic_tests() {
    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", "GENERIC");
    require(labneura::detect_backend() == "GENERIC",
            "Generic backend was not explicitly selected");
    test_supported_numeric_dtypes();
    test_scalar_and_higher_rank_broadcasts();
    test_float_special_values();
    test_zero_size_views_and_output_independence();
    test_errors_and_bool_input_policy();
}

void run_neon_tests() {
    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", "NEON");
    require(labneura::detect_backend() == "NEON",
            "NEON backend was not explicitly selected");
    test_supported_numeric_dtypes();
    test_scalar_and_higher_rank_broadcasts();
    test_float_special_values();
    test_zero_size_views_and_output_independence();
    test_generic_neon_parity();
    test_errors_and_bool_input_policy();
}

void run_unsupported_backend_test(const std::string& backend) {
    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", backend);
    try {
        require(labneura::detect_backend() == backend,
                "Requested backend was not selected");
    } catch (const std::invalid_argument& error) {
        throw SkipTest(error.what());
    }

    const auto lhs =
        labneura::Tensor::from_flat(std::vector<float>{1, 2}, {2});
    const auto rhs =
        labneura::Tensor::from_flat(std::vector<float>{2, 1}, {2});
    for (const ComparisonCase& test : kComparisons) {
        require_throws<std::runtime_error>(
            [&] { (void)(lhs.*(test.method))(rhs); },
            backend + " comparison dispatch",
            "Tensor comparisons are only supported");
        require(labneura::detect_backend() == backend,
                backend + " comparison silently changed backend selection");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Expected GENERIC, NEON, or AVX2\n";
        return 1;
    }
    const std::string backend = argv[1];
    try {
        if (backend == "GENERIC") {
            run_generic_tests();
        } else if (backend == "NEON") {
#if !defined(LABNEURA_TEST_NEON)
            throw SkipTest("NEON backend was not compiled for this target");
#else
            if (!labneura::cpu_supports_neon()) {
                throw SkipTest("CPU does not support NEON");
            }
            run_neon_tests();
#endif
        } else if (backend == "AVX2") {
#if !defined(LABNEURA_TEST_AVX2)
            throw SkipTest("AVX2 backend was not compiled for this target");
#else
            if (!labneura::cpu_supports_avx2()) {
                throw SkipTest("CPU/OS does not support AVX2 state");
            }
            run_unsupported_backend_test(backend);
#endif
        } else {
            throw std::invalid_argument("Unknown backend: " + backend);
        }
        std::cout << backend << ": Tensor comparison tests passed\n";
        return 0;
    } catch (const SkipTest& skip) {
        std::cout << "SKIP: " << skip.what() << '\n';
        return 77;
    } catch (const std::exception& error) {
        std::cerr << "Tensor comparison test failed: " << error.what() << '\n';
        return 1;
    }
}
