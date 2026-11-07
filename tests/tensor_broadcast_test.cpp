#include "labneura/backends/common/backend_factory.h"
#include "labneura/backends/common/cpu_features.h"
#include "labneura/detail/broadcast.h"
#include "labneura/tensor.h"
#include "test_environment.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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

labneura::Tensor make_fp16_tensor(std::vector<float> values,
                                  std::vector<std::size_t> shape) {
    return labneura::Tensor::from_flat(
        values, shape, labneura::QuantizationMode::FP16);
}

labneura::Tensor make_int8_tensor(std::vector<int> values,
                                  std::vector<std::size_t> shape) {
    return labneura::Tensor::from_flat(
        values, shape, labneura::QuantizationMode::INT8);
}

labneura::Tensor make_int16_tensor(std::vector<int> values,
                                   std::vector<std::size_t> shape) {
    return labneura::Tensor::from_flat(
        values, shape, labneura::QuantizationMode::INT16);
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

int16_t reference_int16(labneura::OperationType operation, int16_t lhs,
                        int16_t rhs) {
    int32_t result = 0;
    switch (operation) {
        case labneura::OperationType::ADD:
            result = static_cast<int32_t>(lhs) + static_cast<int32_t>(rhs);
            break;
        case labneura::OperationType::SUB:
            result = static_cast<int32_t>(lhs) - static_cast<int32_t>(rhs);
            break;
        case labneura::OperationType::MUL:
            result = static_cast<int32_t>(lhs) * static_cast<int32_t>(rhs);
            break;
        case labneura::OperationType::DIV:
            if (rhs == 0) return 0;
            result = static_cast<int32_t>(lhs) / static_cast<int32_t>(rhs);
            break;
    }
    return static_cast<int16_t>(
        std::max<int32_t>(-32768, std::min<int32_t>(32767, result)));
}

std::vector<int16_t> reference_int16_broadcast(
    const labneura::Tensor& lhs, const labneura::Tensor& rhs,
    labneura::OperationType operation) {
    const auto plan = labneura::detail::make_broadcast_plan(
        lhs.shape(), lhs.strides(), rhs.shape(), rhs.strides());
    const std::size_t output_numel =
        labneura::detail::checked_broadcast_numel(plan.shape);
    std::vector<int16_t> result(output_numel);
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
            reference_int16(operation, lhs.data_int16()[lhs_offset],
                            rhs.data_int16()[rhs_offset]);
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

void require_int16_values(const labneura::Tensor& actual,
                          const std::vector<std::size_t>& expected_shape,
                          const std::vector<int16_t>& expected,
                          const std::string& context) {
    require(actual.shape() == expected_shape, context + ": shape mismatch");
    require(actual.quantization_mode() == labneura::QuantizationMode::INT16,
            context + ": dtype mismatch");
    require(actual.size() == expected.size(), context + ": size mismatch");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(actual.data_int16()[i] == expected[i],
                context + ": value mismatch at " + std::to_string(i));
    }
}

void require_fp16_bits(const labneura::Tensor& actual,
                       const std::vector<std::size_t>& expected_shape,
                       const std::vector<uint16_t>& expected,
                       const std::string& context) {
    require(actual.shape() == expected_shape, context + ": shape mismatch");
    require(actual.quantization_mode() == labneura::QuantizationMode::FP16,
            context + ": dtype mismatch");
    require(actual.size() == expected.size(), context + ": size mismatch");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const uint16_t bits = static_cast<uint16_t>(actual.data_fp16()[i]);
        require(bits == expected[i],
                context + ": binary16 bits mismatch at " +
                    std::to_string(i) + ", expected " +
                    std::to_string(expected[i]) + ", got " +
                    std::to_string(bits));
    }
}

void set_fp16_bits(labneura::Tensor& tensor,
                   const std::vector<uint16_t>& bits) {
    require(tensor.size() == bits.size(),
            "FP16 bit-pattern setup size mismatch");
    for (std::size_t i = 0; i < bits.size(); ++i) {
        std::memcpy(&tensor.data_fp16()[i], &bits[i], sizeof(bits[i]));
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

std::vector<int16_t> copy_int16_values(const labneura::Tensor& tensor) {
    std::vector<int16_t> values;
    values.reserve(tensor.size());
    for (std::size_t i = 0; i < tensor.size(); ++i) {
        values.push_back(tensor.data_int16()[i]);
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

void test_int16_broadcast_operation(labneura::OperationType operation,
                                    const std::string& name) {
    using labneura::Tensor;
    const std::vector<std::pair<Tensor, Tensor>> cases{
        {make_int16_tensor({32767}, {}),
         make_int16_tensor({1}, {})},
        {make_int16_tensor({-32768, -100, -7, 7, 100, 32767}, {6}),
         make_int16_tensor({3}, {})},
        {make_int16_tensor({32760, -32760, -7, 7, 32767, -32768}, {2, 3}),
         make_int16_tensor({10, -10, 3}, {3})},
        {make_int16_tensor({32760, -32760, -7, 7, 32767, -32768}, {2, 3}),
         make_int16_tensor({1, 2, 3}, {1, 3})},
        {make_int16_tensor({32760, -32760, -7, 7, 32767, -32768}, {2, 3}),
         make_int16_tensor({10, -10}, {2, 1})},
        {make_int16_tensor({-32768, -30000, -7, 0, 7, 30000}, {2, 1, 3}),
         make_int16_tensor({-1, 0, 2, 32767}, {1, 4, 1})},
        {make_int16_tensor({}, {0, 3}),
         make_int16_tensor({1, 2, 3}, {1, 3})},
    };

    for (const auto& test_case : cases) {
        const Tensor& lhs = test_case.first;
        const Tensor& rhs = test_case.second;
        const auto lhs_before = copy_int16_values(lhs);
        const auto rhs_before = copy_int16_values(rhs);
        const auto expected = reference_int16_broadcast(lhs, rhs, operation);
        Tensor result = lhs;
        apply_operation(result, rhs, operation, false);
        const auto plan = labneura::detail::make_broadcast_plan(
            lhs.shape(), lhs.strides(), rhs.shape(), rhs.strides());
        require_int16_values(result, plan.shape, expected,
                             name + " allocating broadcast");
        require(result.is_contiguous() && result.storage_offset() == 0,
                name + ": output is not contiguous independent storage");
        require_int16_values(lhs, lhs.shape(), lhs_before,
                             name + ": allocating operation changed lhs");
        require_int16_values(rhs, rhs.shape(), rhs_before,
                             name + ": allocating operation changed rhs");
    }

    Tensor source = make_int16_tensor(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {4, 3});
    Tensor view = source.slice_first_axis(1, 3);
    const Tensor rhs = make_int16_tensor({32767, 3, -32768}, {3});
    const auto source_before = copy_int16_values(source);
    const auto expected_view = reference_int16_broadcast(view, rhs, operation);
    Tensor allocated = view;
    apply_operation(allocated, rhs, operation, false);
    require_int16_values(allocated, {2, 3}, expected_view,
                         name + " offset-view allocation");
    require_int16_values(source, {4, 3}, source_before,
                         name + ": allocating view operation changed source");
    require(view.storage_offset() == 3, name + ": offset-view setup failed");

    apply_operation(view, rhs, operation, true);
    auto expected_source = source_before;
    for (std::size_t row = 1; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            const std::size_t index = row * 3 + column;
            expected_source[index] = reference_int16(
                operation, source_before[index], rhs.data_int16()[column]);
        }
    }
    require_int16_values(source, {4, 3}, expected_source,
                         name + " in-place offset-view broadcast");

    Tensor destination = make_int16_tensor(
        {-32768, -30000, -7, 0, 7, 30000}, {2, 3});
    const Tensor scalar = make_int16_tensor({3}, {});
    const auto expected_scalar =
        reference_int16_broadcast(destination, scalar, operation);
    apply_operation(destination, scalar, operation, true);
    require_int16_values(destination, {2, 3}, expected_scalar,
                         name + " in-place scalar broadcast");
}

void test_int16_broadcasting() {
    using labneura::OperationType;
    using labneura::Tensor;

    test_int16_broadcast_operation(OperationType::ADD, "INT16 ADD");
    test_int16_broadcast_operation(OperationType::SUB, "INT16 SUB");
    test_int16_broadcast_operation(OperationType::MUL, "INT16 MUL");
    test_int16_broadcast_operation(OperationType::DIV, "INT16 DIV");

    const Tensor lhs =
        make_int16_tensor({32767, -32768, -7, 7, 30000, -30000}, {2, 3});
    const Tensor rhs =
        make_int16_tensor({1, -1, 3, 0, -3, 2}, {2, 3});
    for (const auto operation : {OperationType::ADD, OperationType::SUB,
                                 OperationType::MUL, OperationType::DIV}) {
        const auto expected = reference_int16_broadcast(lhs, rhs, operation);
        Tensor inplace = lhs;
        apply_operation(inplace, rhs, operation, true);
        require_int16_values(inplace, {2, 3}, expected,
                             "INT16 equal-shape in-place operation");
        Tensor allocating = lhs;
        apply_operation(allocating, rhs, operation, false);
        require_int16_values(allocating, {2, 3}, expected,
                             "INT16 equal-shape allocating operation");
    }

    const Tensor division_lhs = make_int16_tensor(
        {-32768, 32767, -7, 7, 32767}, {5});
    const Tensor division_rhs = make_int16_tensor({-1, 0, 3, -3, 2}, {5});
    require_int16_values(division_lhs.div(division_rhs), {5},
                         {32767, 0, -2, -2, 16383},
                         "INT16 division boundaries");

    Tensor alias_source = make_int16_tensor({1, 2, 3}, {3});
    try {
        alias_source.add_inplace(alias_source.select_first_axis(0));
        throw std::runtime_error("INT16 partial overlap was accepted");
    } catch (const std::invalid_argument&) {
    }

    Tensor exact_alias = make_int16_tensor({30000, -30000, 2}, {3});
    exact_alias.add_inplace(exact_alias);
    require_int16_values(exact_alias, {3}, {32767, -32768, 4},
                         "INT16 exact in-place alias");

    Tensor empty = make_int16_tensor({}, {0, 3});
    empty.add_inplace(make_int16_tensor({1, 2, 3}, {1, 3}));
    require(empty.size() == 0 &&
                empty.shape() == std::vector<std::size_t>({0, 3}),
            "INT16 empty in-place broadcast changed metadata");

    try {
        (void)make_int16_tensor({1, 2, 3, 4, 5, 6}, {2, 3})
            .add(make_int16_tensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                                  {4, 3}));
        throw std::runtime_error("INT16 incompatible shapes were accepted");
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()).find("shape mismatch") !=
                    std::string::npos,
                "INT16 incompatible-shape error was not actionable");
    }

    Tensor scalar = make_int16_tensor({1}, {});
    try {
        scalar.add_inplace(make_int16_tensor({1, 2}, {2}));
        throw std::runtime_error("INT16 destination expansion was accepted");
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()).find("cannot expand") !=
                    std::string::npos,
                "INT16 destination-expansion error was not actionable");
    }
}

void test_fp16_broadcasting() {
    using labneura::OperationType;
    using labneura::Tensor;
    using labneura::QuantizationMode;

    struct OperationCase {
        OperationType operation;
        const char* name;
        std::vector<uint16_t> equal_shape;
        std::vector<uint16_t> scalar;
        std::vector<uint16_t> multi_axis;
        std::vector<uint16_t> offset_view;
    };
    const std::vector<OperationCase> cases = {
        {OperationType::ADD, "FP16 ADD",
         {0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800},
         {0x4400, 0x4600, 0x4800},
         {0x4400, 0x4600, 0x4800, 0x4600, 0x4800, 0x4900,
          0x4900, 0x4A00, 0x4B00, 0x4A00, 0x4B00, 0x4C00},
         {0x3C00, 0x4000, 0x4200, 0x4600, 0x4700, 0x4800,
          0x4880, 0x4900, 0x4980, 0x4900, 0x4980, 0x4A00}},
        {OperationType::SUB, "FP16 SUB",
         {0xBC00, 0x0000, 0x3C00, 0x4000, 0x4200, 0x4400},
         {0x0000, 0x4000, 0x4400},
         {0x0000, 0x4000, 0x4400, 0xC000, 0x0000, 0x4000,
          0x4600, 0x4800, 0x4900, 0x4400, 0x4600, 0x4800},
         {0x3C00, 0x4000, 0x4200, 0x4000, 0x4200, 0x4400,
          0x4500, 0x4600, 0x4700, 0x4900, 0x4980, 0x4A00}},
        {OperationType::MUL, "FP16 MUL",
         {0x4000, 0x4400, 0x4600, 0x4800, 0x4900, 0x4A00},
         {0x4400, 0x4800, 0x4A00},
         {0x4400, 0x4800, 0x4A00, 0x4800, 0x4C00, 0x4E00,
          0x4C00, 0x4D00, 0x4E00, 0x5000, 0x5100, 0x5200},
         {0x3C00, 0x4000, 0x4200, 0x4800, 0x4900, 0x4A00,
          0x4B00, 0x4C00, 0x4C80, 0x4900, 0x4980, 0x4A00}},
        {OperationType::DIV, "FP16 DIV",
         {0x3800, 0x3C00, 0x3E00, 0x4000, 0x4100, 0x4200},
         {0x3C00, 0x4000, 0x4200},
         {0x3C00, 0x4000, 0x4200, 0x3800, 0x3C00, 0x3E00,
          0x4400, 0x4500, 0x4600, 0x4000, 0x4100, 0x4200},
         {0x3C00, 0x4000, 0x4200, 0x4000, 0x4100, 0x4200,
          0x4300, 0x4400, 0x4480, 0x4900, 0x4980, 0x4A00}}
    };

    const std::vector<float> equal_lhs{1, 2, 3, 4, 5, 6};
    const std::vector<float> equal_rhs{2, 2, 2, 2, 2, 2};
    const std::vector<float> scalar_lhs{2, 4, 6};
    const Tensor scalar_rhs = make_fp16_tensor({2}, {});
    const std::vector<float> multi_lhs{2, 4, 6, 8, 10, 12};
    const std::vector<float> multi_rhs{2, 4};

    for (const auto& test : cases) {
        const Tensor lhs =
            make_fp16_tensor(equal_lhs, {2, 3});
        const Tensor rhs =
            make_fp16_tensor(equal_rhs, {2, 3});
        Tensor allocated = lhs;
        apply_operation(allocated, rhs, test.operation, false);
        require_fp16_bits(allocated, {2, 3}, test.equal_shape,
                          std::string(test.name) + " equal-shape allocating");
        Tensor inplace = lhs;
        apply_operation(inplace, rhs, test.operation, true);
        require_fp16_bits(inplace, {2, 3}, test.equal_shape,
                          std::string(test.name) + " equal-shape in-place");

        const Tensor scalar_result =
            [&]() {
                Tensor result;
                apply_operation(result = make_fp16_tensor(scalar_lhs, {3}),
                                scalar_rhs, test.operation, false);
                return result;
            }();
        require_fp16_bits(scalar_result, {3}, test.scalar,
                          std::string(test.name) + " scalar broadcast");

        const Tensor multi_result =
            [&]() {
                Tensor result;
                apply_operation(
                    result = make_fp16_tensor(multi_lhs, {2, 1, 3}),
                    make_fp16_tensor(multi_rhs, {1, 2, 1}),
                    test.operation, false);
                return result;
            }();
        require_fp16_bits(multi_result, {2, 2, 3}, test.multi_axis,
                          std::string(test.name) +
                              " multi-axis 3-D broadcast");

        Tensor source = make_fp16_tensor(
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {4, 3});
        Tensor view = source.slice_first_axis(1, 3);
        require(view.storage_offset() == 3,
                "FP16 view did not have a nonzero storage offset");
        const Tensor view_rhs = make_fp16_tensor({2}, {});
        apply_operation(view, view_rhs, test.operation, true);
        require_fp16_bits(source, {4, 3}, test.offset_view,
                          std::string(test.name) +
                              " in-place offset-view shared mutation");
    }

    for (const auto operation : {OperationType::ADD, OperationType::SUB,
                                 OperationType::MUL, OperationType::DIV}) {
        Tensor empty = make_fp16_tensor({}, {0, 3});
        const Tensor row = make_fp16_tensor({1, 2, 3}, {1, 3});
        Tensor result = empty;
        apply_operation(result, row, operation, false);
        require(result.size() == 0 &&
                    result.shape() == std::vector<std::size_t>({0, 3}) &&
                    result.quantization_mode() == QuantizationMode::FP16,
                "FP16 zero-sized allocating broadcast changed metadata");
        apply_operation(empty, row, operation, true);
        require(empty.size() == 0 &&
                    empty.shape() == std::vector<std::size_t>({0, 3}),
                "FP16 zero-sized in-place broadcast changed metadata");

        Tensor scalar_destination = make_fp16_tensor({1}, {});
        try {
            apply_operation(
                scalar_destination,
                make_fp16_tensor({1, 2}, {2}), operation, true);
            throw std::runtime_error(
                "FP16 in-place destination expansion was accepted");
        } catch (const std::runtime_error& error) {
            require(std::string(error.what()).find("cannot expand") !=
                        std::string::npos,
                    "FP16 destination-expansion error was not actionable");
        }
    }

    Tensor raw_lhs = make_fp16_tensor({0}, {});
    Tensor raw_rhs = make_fp16_tensor({0}, {});
    set_fp16_bits(raw_lhs, {0x3C00});
    set_fp16_bits(raw_rhs, {0x3800});
    require_fp16_bits(raw_lhs, {}, {0x3C00}, "FP16 raw lhs bits");
    require_fp16_bits(raw_rhs, {}, {0x3800}, "FP16 raw rhs bits");
    require_fp16_bits(raw_lhs.add(raw_rhs), {}, {0x3E00},
                      "FP16 storage bits interpreted as binary16 values");

    Tensor original = make_fp16_tensor(equal_lhs, {2, 3});
    Tensor independent = original.add(
        make_fp16_tensor(equal_rhs, {2, 3}));
    independent.add_inplace(make_fp16_tensor({1}, {}));
    require_fp16_bits(original, {2, 3},
                      {0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600},
                      "FP16 allocating input independence");
}

void run_fp16_generic_tests() {
    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", "GENERIC");
    require(labneura::detect_backend() == "GENERIC",
            "Generic backend was not selected for FP16 broadcasting");
    test_fp16_broadcasting();
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
    test_int16_broadcasting();
    test_fp16_broadcasting();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "FP16") {
            run_fp16_generic_tests();
            std::cout << "GENERIC: FP16 broadcasting tests passed\n";
        } else if (argc == 1) {
            run_generic_tests();
            std::cout << "GENERIC: FP32, FP16, INT8, and INT16 broadcasting tests passed\n";
        } else {
            throw std::invalid_argument("Expected no argument or FP16");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Broadcasting test failed: " << error.what() << '\n';
        return 1;
    }
}
