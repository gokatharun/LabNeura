#include "labneura/backends/common/backend_factory.h"
#include "labneura/backends/common/cpu_features.h"
#include "labneura/detail/broadcast.h"
#include "labneura/tensor.h"
#include "test_environment.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

labneura::Tensor make_tensor(std::vector<float> values,
                             std::vector<std::size_t> shape) {
    return labneura::Tensor::from_flat(values, shape);
}

labneura::Tensor make_int8_tensor(std::vector<int> values,
                                  std::vector<std::size_t> shape) {
    return labneura::Tensor::from_flat(
        values, shape, labneura::QuantizationMode::INT8);
}

void apply_operation(labneura::Tensor& lhs, const labneura::Tensor& rhs,
                     labneura::OperationType operation, bool inplace) {
    if (inplace) {
        switch (operation) {
            case labneura::OperationType::ADD: lhs.add_inplace(rhs); break;
            case labneura::OperationType::SUB: lhs.sub_inplace(rhs); break;
            case labneura::OperationType::MUL: lhs.mul_inplace(rhs); break;
            case labneura::OperationType::DIV: lhs.div_inplace(rhs); break;
        }
        return;
    }
    switch (operation) {
        case labneura::OperationType::ADD: lhs = lhs.add(rhs); break;
        case labneura::OperationType::SUB: lhs = lhs.sub(rhs); break;
        case labneura::OperationType::MUL: lhs = lhs.mul(rhs); break;
        case labneura::OperationType::DIV: lhs = lhs.div(rhs); break;
    }
}

int8_t reference_int8(labneura::OperationType operation, int8_t lhs,
                      int8_t rhs) {
    int16_t result = 0;
    switch (operation) {
        case labneura::OperationType::ADD:
            result = static_cast<int16_t>(lhs) + static_cast<int16_t>(rhs);
            break;
        case labneura::OperationType::SUB:
            result = static_cast<int16_t>(lhs) - static_cast<int16_t>(rhs);
            break;
        case labneura::OperationType::MUL:
            result = static_cast<int16_t>(lhs) * static_cast<int16_t>(rhs);
            break;
        case labneura::OperationType::DIV:
            if (rhs == 0) return 0;
            result = static_cast<int16_t>(lhs) / static_cast<int16_t>(rhs);
            break;
    }
    return static_cast<int8_t>(
        std::max<int16_t>(-128, std::min<int16_t>(127, result)));
}

std::vector<int8_t> reference_int8_broadcast(
    const labneura::Tensor& lhs, const labneura::Tensor& rhs,
    labneura::OperationType operation) {
    const auto plan = labneura::detail::make_broadcast_plan(
        lhs.shape(), lhs.strides(), rhs.shape(), rhs.strides());
    const std::size_t output_numel =
        labneura::detail::checked_broadcast_numel(plan.shape);
    std::vector<int8_t> result(output_numel);
    for (std::size_t linear = 0; linear < output_numel; ++linear) {
        std::size_t remainder = linear;
        std::size_t lhs_offset = 0;
        std::size_t rhs_offset = 0;
        for (std::size_t axis = plan.shape.size(); axis > 0; --axis) {
            const std::size_t dimension = plan.shape[axis - 1];
            const std::size_t coordinate = remainder % dimension;
            remainder /= dimension;
            lhs_offset += coordinate * plan.lhs_strides[axis - 1];
            rhs_offset += coordinate * plan.rhs_strides[axis - 1];
        }
        result[linear] =
            reference_int8(operation, lhs.data_int8()[lhs_offset],
                           rhs.data_int8()[rhs_offset]);
    }
    return result;
}

void require_int8_values(const labneura::Tensor& actual,
                         const std::vector<std::size_t>& expected_shape,
                         const std::vector<int8_t>& expected,
                         const std::string& context) {
    require(actual.shape() == expected_shape, context + ": shape mismatch");
    require(actual.quantization_mode() == labneura::QuantizationMode::INT8,
            context + ": dtype mismatch");
    require(actual.size() == expected.size(), context + ": size mismatch");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(actual.data_int8()[i] == expected[i],
                context + ": value mismatch at " + std::to_string(i));
    }
}

std::vector<int8_t> copy_int8_values(const labneura::Tensor& tensor) {
    std::vector<int8_t> values;
    values.reserve(tensor.size());
    for (std::size_t i = 0; i < tensor.size(); ++i) {
        values.push_back(tensor.data_int8()[i]);
    }
    return values;
}

void require_values(const labneura::Tensor& actual,
                    const std::vector<std::size_t>& expected_shape,
                    const std::vector<float>& expected,
                    const std::string& context) {
    require(actual.shape() == expected_shape, context + ": shape mismatch");
    require(actual.quantization_mode() == labneura::QuantizationMode::FP32,
            context + ": dtype mismatch");
    require(actual.size() == expected.size(), context + ": size mismatch");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float value = actual.data_fp32()[i];
        if (std::isnan(expected[i])) {
            require(std::isnan(value), context + ": expected NaN");
        } else if (std::isinf(expected[i])) {
            require(value == expected[i], context + ": infinity mismatch");
        } else {
            require(value == expected[i],
                    context + ": value mismatch at " + std::to_string(i));
        }
    }
}

void test_scalar_and_matrix_vector_operations() {
    using labneura::Tensor;

    const Tensor scalar_lhs = make_tensor({6.0f}, {});
    const Tensor scalar_rhs = make_tensor({2.0f}, {});
    require_values(scalar_lhs.add(scalar_rhs), {}, {8.0f}, "scalar ADD");
    require_values(scalar_lhs.sub(scalar_rhs), {}, {4.0f}, "scalar SUB");
    require_values(scalar_lhs.mul(scalar_rhs), {}, {12.0f}, "scalar MUL");
    require_values(scalar_lhs.div(scalar_rhs), {}, {3.0f}, "scalar DIV");

    const Tensor matrix =
        make_tensor({1, 2, 3, 4, 5, 6}, {2, 3});
    const Tensor vector = make_tensor({10, 20, 30}, {3});
    const std::vector<float> lhs_before(
        matrix.data_fp32(), matrix.data_fp32() + matrix.size());
    const std::vector<float> rhs_before(
        vector.data_fp32(), vector.data_fp32() + vector.size());

    const Tensor added = matrix.add(vector);
    require_values(added, {2, 3},
                   {11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f},
                   "matrix/vector ADD");
    require_values(matrix.sub(vector), {2, 3},
                   {-9.0f, -18.0f, -27.0f, -6.0f, -15.0f, -24.0f},
                   "matrix/vector SUB");
    require_values(matrix.mul(vector), {2, 3},
                   {10.0f, 40.0f, 90.0f, 40.0f, 100.0f, 180.0f},
                   "matrix/vector MUL");
    require_values(matrix.div(vector), {2, 3},
                   {0.1f, 0.1f, 0.1f, 0.4f, 0.25f, 0.2f},
                   "matrix/vector DIV");
    require(std::vector<float>(matrix.data_fp32(),
                               matrix.data_fp32() + matrix.size()) == lhs_before,
            "Allocating arithmetic modified the left input");
    require(std::vector<float>(vector.data_fp32(),
                               vector.data_fp32() + vector.size()) == rhs_before,
            "Allocating arithmetic modified the right input");

    Tensor independent = added;
    independent.add_inplace(make_tensor({1.0f}, {}));
    require_values(added, {2, 3},
                   {11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f},
                   "allocating result storage independence");
}

void test_multidimensional_and_empty_broadcasts() {
    using labneura::Tensor;

    const Tensor lhs =
        make_tensor({1, 2, 3, 4, 5, 6}, {2, 1, 3});
    const Tensor rhs =
        make_tensor({10, 20, 30, 40}, {1, 4, 1});
    require_values(lhs.add(rhs), {2, 4, 3},
                   {11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43,
                    14, 15, 16, 24, 25, 26, 34, 35, 36, 44, 45, 46},
                   "3-D multiple-axis broadcast");

    const Tensor empty_lhs = make_tensor({}, {0, 3});
    const Tensor row = make_tensor({1, 2, 3}, {1, 3});
    require_values(empty_lhs.add(row), {0, 3}, {},
                   "zero-sized allocating broadcast");
    Tensor empty_inplace = make_tensor({}, {0, 3});
    empty_inplace.add_inplace(row);
    require(empty_inplace.size() == 0 &&
                empty_inplace.shape() == std::vector<std::size_t>({0, 3}),
            "zero-sized in-place broadcast changed logical metadata");
}

void test_inplace_broadcast_and_offset_views() {
    using labneura::Tensor;

    const std::vector<float> rhs_values{10.0f, 20.0f, 30.0f};
    for (const auto operation : {labneura::OperationType::ADD,
                                 labneura::OperationType::SUB,
                                 labneura::OperationType::MUL,
                                 labneura::OperationType::DIV}) {
        Tensor source = make_tensor(
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
             7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
            {4, 3});
        Tensor view = source.slice_first_axis(1, 3);
        const Tensor rhs = make_tensor(rhs_values, {3});
        require(view.storage_offset() == 3,
                "offset-view setup did not produce a nonzero offset");

        switch (operation) {
            case labneura::OperationType::ADD: view.add_inplace(rhs); break;
            case labneura::OperationType::SUB: view.sub_inplace(rhs); break;
            case labneura::OperationType::MUL: view.mul_inplace(rhs); break;
            case labneura::OperationType::DIV: view.div_inplace(rhs); break;
        }

        std::vector<float> expected{1, 2, 3};
        const std::vector<float> original{4, 5, 6, 7, 8, 9};
        for (std::size_t i = 0; i < original.size(); ++i) {
            switch (operation) {
                case labneura::OperationType::ADD:
                    expected.push_back(original[i] + rhs_values[i % 3]);
                    break;
                case labneura::OperationType::SUB:
                    expected.push_back(original[i] - rhs_values[i % 3]);
                    break;
                case labneura::OperationType::MUL:
                    expected.push_back(original[i] * rhs_values[i % 3]);
                    break;
                case labneura::OperationType::DIV:
                    expected.push_back(original[i] / rhs_values[i % 3]);
                    break;
            }
        }
        expected.insert(expected.end(), {10, 11, 12});
        require_values(source, {4, 3}, expected,
                       "in-place offset-view shared mutation");
    }
}

void test_errors_aliases_and_ieee_values() {
    using labneura::Tensor;

    const Tensor matrix = make_tensor({1, 2, 3, 4, 5, 6}, {2, 3});
    const Tensor incompatible = make_tensor(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {4, 3});
    try {
        (void)matrix.add(incompatible);
        throw std::runtime_error("Incompatible shapes were accepted");
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()).find("shape mismatch") !=
                    std::string::npos,
                "Incompatible-shape error was not actionable");
    }

    Tensor scalar = make_tensor({1.0f}, {});
    try {
        scalar.add_inplace(make_tensor({1.0f, 2.0f}, {2}));
        throw std::runtime_error("In-place destination expansion was accepted");
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()).find("cannot expand") !=
                    std::string::npos,
                "Destination-expansion error was not actionable");
    }

    Tensor alias_source = make_tensor({1.0f, 2.0f, 3.0f}, {3});
    Tensor first_element = alias_source.select_first_axis(0);
    try {
        alias_source.add_inplace(first_element);
        throw std::runtime_error(
            "Broadcasting an overlapping scalar alias was accepted");
    } catch (const std::invalid_argument&) {
    }

    const Tensor lhs = make_tensor(
        {0.0f, -0.0f, 1.0f, -1.0f,
         std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::quiet_NaN()},
        {2, 3});
    const Tensor zero = make_tensor({0.0f}, {});
    const Tensor quotient = lhs.div(zero);
    require(std::isnan(quotient.data_fp32()[0]) &&
                std::isnan(quotient.data_fp32()[1]) &&
                std::isinf(quotient.data_fp32()[2]) &&
                !std::signbit(quotient.data_fp32()[2]) &&
                std::isinf(quotient.data_fp32()[3]) &&
                std::signbit(quotient.data_fp32()[3]) &&
                std::isinf(quotient.data_fp32()[4]) &&
                std::isnan(quotient.data_fp32()[5]),
            "FP32 division did not preserve IEEE special-value behavior");

    const Tensor scales = make_tensor(
        {std::numeric_limits<float>::max(), 1.0e-30f}, {2});
    const Tensor factors = make_tensor({2.0f}, {});
    const Tensor product = scales.mul(factors);
    require(std::isinf(product.data_fp32()[0]) &&
                product.data_fp32()[1] > 0.0f &&
                std::isfinite(product.data_fp32()[1]),
            "FP32 large/small arithmetic edge case failed");
}

void test_int8_broadcast_operation(labneura::OperationType operation,
                                   const std::string& name) {
    using labneura::Tensor;
    const std::vector<std::pair<Tensor, Tensor>> cases{
        {make_int8_tensor({100}, {}),
         make_int8_tensor({50}, {})},
        {make_int8_tensor({-128, -100, -5, 5, 100, 127}, {6}),
         make_int8_tensor({2}, {})},
        {make_int8_tensor({100, -100, 1, -1, 127, -128}, {2, 3}),
         make_int8_tensor({50, -50, 0}, {3})},
        {make_int8_tensor({-128, -100, -1, 0, 1, 100}, {2, 3}),
         make_int8_tensor({-1, 2, 0, 127, -128, 1}, {2, 3})},
        {make_int8_tensor({-128, -100, -1, 0, 1, 100}, {2, 1, 3}),
         make_int8_tensor({-1, 0, 2, 127}, {1, 4, 1})},
        {make_int8_tensor({}, {0, 3}),
         make_int8_tensor({1, 2, 3}, {1, 3})},
    };

    for (const auto& test_case : cases) {
        const Tensor& lhs = test_case.first;
        const Tensor& rhs = test_case.second;
        const auto lhs_before = copy_int8_values(lhs);
        const auto rhs_before = copy_int8_values(rhs);
        const auto expected = reference_int8_broadcast(lhs, rhs, operation);
        Tensor result = lhs;
        apply_operation(result, rhs, operation, false);
        const auto plan = labneura::detail::make_broadcast_plan(
            lhs.shape(), lhs.strides(), rhs.shape(), rhs.strides());
        require_int8_values(result, plan.shape, expected,
                            name + " allocating broadcast");
        require(result.storage_offset() == 0,
                name + ": allocating output has a nonzero offset");
        require(result.is_contiguous(),
                name + ": allocating output is not contiguous");
        require_int8_values(lhs, lhs.shape(), lhs_before,
                            name + ": allocating operation changed lhs");
        require_int8_values(rhs, rhs.shape(), rhs_before,
                            name + ": allocating operation changed rhs");
    }

    Tensor source = make_int8_tensor(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {4, 3});
    Tensor view = source.slice_first_axis(1, 3);
    const Tensor rhs = make_int8_tensor({2, 3, 4}, {3});
    const auto source_before_view_operation = copy_int8_values(source);
    const auto expected_view = reference_int8_broadcast(view, rhs, operation);
    Tensor allocated = view;
    apply_operation(allocated, rhs, operation, false);
    require_int8_values(allocated, {2, 3}, expected_view,
                        name + " offset-view allocation");
    require_int8_values(source, {4, 3}, source_before_view_operation,
                        name + ": allocating view operation changed source");
    require(view.storage_offset() == 3,
            name + ": offset-view setup failed");

    const std::vector<int8_t> before(source.data_int8(),
                                     source.data_int8() + source.size());
    apply_operation(view, rhs, operation, true);
    auto expected_source = before;
    for (std::size_t row = 1; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            const std::size_t index = row * 3 + column;
            expected_source[index] =
                reference_int8(operation, before[index],
                               rhs.data_int8()[column]);
        }
    }
    require_int8_values(source, {4, 3}, expected_source,
                        name + " in-place offset-view broadcast");

    Tensor scalar_destination =
        make_int8_tensor({-128, -100, -1, 0, 1, 100}, {2, 3});
    const Tensor scalar_rhs = make_int8_tensor({2}, {});
    const auto expected_scalar =
        reference_int8_broadcast(scalar_destination, scalar_rhs, operation);
    apply_operation(scalar_destination, scalar_rhs, operation, true);
    require_int8_values(scalar_destination, {2, 3}, expected_scalar,
                        name + " in-place scalar broadcast");
}

void test_int8_broadcasting() {
    using labneura::Tensor;
    using labneura::OperationType;
    test_int8_broadcast_operation(OperationType::ADD, "INT8 ADD");
    test_int8_broadcast_operation(OperationType::SUB, "INT8 SUB");
    test_int8_broadcast_operation(OperationType::MUL, "INT8 MUL");
    test_int8_broadcast_operation(OperationType::DIV, "INT8 DIV");

    const auto backend = labneura::detect_backend();
    require(backend == "GENERIC", "INT8 broadcast test did not select Generic");

    Tensor alias_source = make_int8_tensor({1, 2, 3}, {3});
    try {
        alias_source.add_inplace(alias_source.select_first_axis(0));
        throw std::runtime_error("INT8 partial overlap was accepted");
    } catch (const std::invalid_argument&) {
    }

    Tensor scalar = make_int8_tensor({1}, {});
    try {
        scalar.add_inplace(make_int8_tensor({1, 2}, {2}));
        throw std::runtime_error("INT8 destination expansion was accepted");
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()).find("cannot expand") !=
                    std::string::npos,
                "INT8 destination expansion error was not actionable");
    }

    const Tensor division_lhs = make_int8_tensor({-128, 127, -7, 7}, {4});
    const Tensor division_rhs = make_int8_tensor({-1, 0, 3, -3}, {4});
    const Tensor quotient = division_lhs.div(division_rhs);
    require_int8_values(quotient, {4}, {127, 0, -2, -2},
                        "INT8 division boundaries");

    for (const auto operation : {OperationType::ADD, OperationType::SUB,
                                 OperationType::MUL, OperationType::DIV}) {
        const Tensor lhs =
            make_int8_tensor({127, -128, 100, -100, -128, 127}, {2, 3});
        const Tensor rhs = make_int8_tensor({1, -1, 50, -50, -1, 0}, {2, 3});
        const auto expected = reference_int8_broadcast(lhs, rhs, operation);
        Tensor result = lhs;
        apply_operation(result, rhs, operation, false);
        require_int8_values(result, {2, 3}, expected,
                            "INT8 arithmetic boundary saturation");
    }

    Tensor exact_alias = make_int8_tensor({100, -100, 2}, {3});
    exact_alias.add_inplace(exact_alias);
    require_int8_values(exact_alias, {3}, {127, -128, 4},
                        "INT8 exact in-place alias");

    Tensor empty_destination = make_int8_tensor({}, {0, 3});
    empty_destination.add_inplace(make_int8_tensor({1, 2, 3}, {1, 3}));
    require(empty_destination.shape() == std::vector<std::size_t>({0, 3}) &&
                empty_destination.size() == 0,
            "INT8 empty in-place broadcast changed destination metadata");

    try {
        (void)make_int8_tensor({1, 2, 3, 4, 5, 6}, {2, 3})
            .add(make_int8_tensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                                  {4, 3}));
        throw std::runtime_error("INT8 incompatible shapes were accepted");
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()).find("shape mismatch") !=
                    std::string::npos,
                "INT8 incompatible-shape error was not actionable");
    }
}

void run_generic_tests() {
    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", "GENERIC");
    require(labneura::detect_backend() == "GENERIC",
            "Generic backend was not selected");
    test_scalar_and_matrix_vector_operations();
    test_multidimensional_and_empty_broadcasts();
    test_inplace_broadcast_and_offset_views();
    test_errors_aliases_and_ieee_values();
    test_int8_broadcasting();
}

}  // namespace

int main() {
    try {
        run_generic_tests();
        std::cout << "GENERIC: FP32 and INT8 broadcasting tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Broadcasting test failed: " << error.what() << '\n';
        return 1;
    }
}
