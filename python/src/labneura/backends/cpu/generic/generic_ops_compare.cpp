#include "labneura/backends/cpu/generic.h"

#include "labneura/types.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace labneura {
namespace {

template <typename Element>
bool compare_values(ComparisonType comparison, Element lhs, Element rhs) {
    switch (comparison) {
        case ComparisonType::EQ: return lhs == rhs;
        case ComparisonType::NE: return lhs != rhs;
        case ComparisonType::LT: return lhs < rhs;
        case ComparisonType::LE: return lhs <= rhs;
        case ComparisonType::GT: return lhs > rhs;
        case ComparisonType::GE: return lhs >= rhs;
    }
    throw std::invalid_argument("Unknown Tensor comparison operation");
}

std::size_t maximum_input_offset(
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& strides) {
    if (shape.size() != strides.size()) {
        throw std::invalid_argument(
            "Comparison broadcast shape and stride ranks must match");
    }

    std::size_t maximum = 0;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        if (shape[axis] <= 1 || strides[axis] == 0) {
            continue;
        }
        const std::size_t count = shape[axis] - 1;
        if (count > std::numeric_limits<std::size_t>::max() / strides[axis]) {
            throw std::overflow_error(
                "Comparison input offset overflows size_t");
        }
        const std::size_t advance = count * strides[axis];
        if (maximum >
            std::numeric_limits<std::size_t>::max() - advance) {
            throw std::overflow_error(
                "Comparison input offset overflows size_t");
        }
        maximum += advance;
    }
    return maximum;
}

template <typename Element>
void execute_comparison(const Element* lhs, const Element* rhs,
                        std::uint8_t* output,
                        const detail::BroadcastPlan& plan,
                        std::size_t output_numel,
                        ComparisonType comparison) {
    for (std::size_t linear_index = 0; linear_index < output_numel;
         ++linear_index) {
        std::size_t remainder = linear_index;
        std::size_t lhs_offset = 0;
        std::size_t rhs_offset = 0;
        for (std::size_t axis = plan.shape.size(); axis > 0; --axis) {
            const std::size_t dimension = plan.shape[axis - 1];
            const std::size_t coordinate = remainder % dimension;
            remainder /= dimension;
            lhs_offset += coordinate * plan.lhs_strides[axis - 1];
            rhs_offset += coordinate * plan.rhs_strides[axis - 1];
        }
        output[linear_index] =
            compare_values(comparison, lhs[lhs_offset], rhs[rhs_offset])
                ? std::uint8_t{1}
                : std::uint8_t{0};
    }
}

void execute_fp16_comparison(const std::int16_t* lhs,
                             const std::int16_t* rhs,
                             std::uint8_t* output,
                             const detail::BroadcastPlan& plan,
                             std::size_t output_numel,
                             ComparisonType comparison) {
    for (std::size_t linear_index = 0; linear_index < output_numel;
         ++linear_index) {
        std::size_t remainder = linear_index;
        std::size_t lhs_offset = 0;
        std::size_t rhs_offset = 0;
        for (std::size_t axis = plan.shape.size(); axis > 0; --axis) {
            const std::size_t dimension = plan.shape[axis - 1];
            const std::size_t coordinate = remainder % dimension;
            remainder /= dimension;
            lhs_offset += coordinate * plan.lhs_strides[axis - 1];
            rhs_offset += coordinate * plan.rhs_strides[axis - 1];
        }
        const float lhs_value = dequantize_fp16(
            static_cast<std::uint16_t>(lhs[lhs_offset]));
        const float rhs_value = dequantize_fp16(
            static_cast<std::uint16_t>(rhs[rhs_offset]));
        output[linear_index] =
            compare_values(comparison, lhs_value, rhs_value)
                ? std::uint8_t{1}
                : std::uint8_t{0};
    }
}

}  // namespace

void GenericBackend::compare_broadcast(
    const GenericBackend& other, GenericBackend& output,
    const detail::BroadcastPlan& plan, ComparisonType comparison,
    std::size_t lhs_numel, std::size_t rhs_numel) const {
    if (quantization_mode_ != other.quantization_mode_ ||
        output.quantization_mode_ != QuantizationMode::BOOL) {
        throw std::runtime_error(
            "Comparison requires matching input dtypes and BOOL output");
    }
    if (quantization_mode_ != QuantizationMode::FP32 &&
        quantization_mode_ != QuantizationMode::FP16 &&
        quantization_mode_ != QuantizationMode::INT16 &&
        quantization_mode_ != QuantizationMode::INT8) {
        throw std::runtime_error(
            "Generic comparisons support FP32, FP16, INT16, and INT8 tensors");
    }
    if (size_ != lhs_numel || other.size_ != rhs_numel) {
        throw std::runtime_error(
            "Comparison operand logical element count mismatch");
    }
    if (plan.shape.size() != plan.lhs_strides.size() ||
        plan.shape.size() != plan.rhs_strides.size()) {
        throw std::invalid_argument(
            "Comparison broadcast plan shape and stride ranks must match");
    }

    const std::size_t output_numel =
        detail::checked_broadcast_numel(plan.shape);
    if (output.size_ != output_numel) {
        throw std::runtime_error(
            "Comparison output logical element count mismatch");
    }
    if (output_numel == 0) {
        return;
    }
    if (maximum_input_offset(plan.shape, plan.lhs_strides) >= size_ ||
        maximum_input_offset(plan.shape, plan.rhs_strides) >= other.size_) {
        throw std::out_of_range(
            "Comparison broadcast accesses beyond input logical elements");
    }

    std::uint8_t* output_data = output.data_bool();
    if (quantization_mode_ == QuantizationMode::FP32) {
        execute_comparison(data_fp32(), other.data_fp32(), output_data, plan,
                           output_numel, comparison);
    } else if (quantization_mode_ == QuantizationMode::FP16) {
        execute_fp16_comparison(data_fp16(), other.data_fp16(), output_data,
                                plan, output_numel, comparison);
    } else if (quantization_mode_ == QuantizationMode::INT16) {
        execute_comparison(data_int16(), other.data_int16(), output_data, plan,
                           output_numel, comparison);
    } else {
        execute_comparison(data_int8(), other.data_int8(), output_data, plan,
                           output_numel, comparison);
    }
}

}  // namespace labneura
