#include "labneura/backends/cpu/generic.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace labneura {
namespace {

std::size_t maximum_input_offset(
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& strides, const char* operation_name) {
    if (shape.size() != strides.size()) {
        throw std::invalid_argument(
            std::string(operation_name) +
            " broadcast shape and stride ranks must match");
    }

    std::size_t maximum = 0;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        if (shape[axis] <= 1 || strides[axis] == 0) {
            continue;
        }
        const std::size_t count = shape[axis] - 1;
        if (count > std::numeric_limits<std::size_t>::max() / strides[axis]) {
            throw std::overflow_error(
                std::string(operation_name) +
                " input offset overflows size_t");
        }
        const std::size_t advance = count * strides[axis];
        if (maximum >
            std::numeric_limits<std::size_t>::max() - advance) {
            throw std::overflow_error(
                std::string(operation_name) +
                " input offset overflows size_t");
        }
        maximum += advance;
    }
    return maximum;
}

enum class LogicalBinaryOperation { And, Or };

template <LogicalBinaryOperation Operation>
void logical_binary_broadcast(
    const GenericBackend& lhs_backend, const GenericBackend& rhs_backend,
    GenericBackend& output, const detail::BroadcastPlan& plan,
    std::size_t lhs_numel, std::size_t rhs_numel) {
    constexpr const char* operation_name =
        Operation == LogicalBinaryOperation::And ? "Logical AND" : "Logical OR";
    if (lhs_backend.quantization_mode() != QuantizationMode::BOOL ||
        rhs_backend.quantization_mode() != QuantizationMode::BOOL ||
        output.quantization_mode() != QuantizationMode::BOOL) {
        throw std::runtime_error(std::string(operation_name) +
                                 " requires BOOL input and output tensors");
    }
    if (lhs_backend.size() != lhs_numel ||
        rhs_backend.size() != rhs_numel) {
        throw std::runtime_error(std::string(operation_name) +
                                 " operand logical element count mismatch");
    }
    if (plan.shape.size() != plan.lhs_strides.size() ||
        plan.shape.size() != plan.rhs_strides.size()) {
        throw std::invalid_argument(std::string(operation_name) +
                                    " broadcast plan shape and stride ranks must match");
    }

    const std::size_t output_numel =
        detail::checked_broadcast_numel(plan.shape);
    if (output.size() != output_numel) {
        throw std::runtime_error(std::string(operation_name) +
                                 " output logical element count mismatch");
    }
    if (output_numel == 0) {
        return;
    }
    if (maximum_input_offset(plan.shape, plan.lhs_strides, operation_name) >=
            lhs_backend.size() ||
        maximum_input_offset(plan.shape, plan.rhs_strides, operation_name) >=
            rhs_backend.size()) {
        throw std::out_of_range(std::string(operation_name) +
                                " broadcast accesses beyond input logical elements");
    }

    const std::uint8_t* lhs = lhs_backend.data_bool();
    const std::uint8_t* rhs = rhs_backend.data_bool();
    std::uint8_t* destination = output.data_bool();
    for (std::size_t index = 0; index < output_numel; ++index) {
        std::size_t remainder = index;
        std::size_t lhs_offset = 0;
        std::size_t rhs_offset = 0;
        for (std::size_t axis = plan.shape.size(); axis > 0; --axis) {
            const std::size_t dimension = plan.shape[axis - 1];
            const std::size_t coordinate = remainder % dimension;
            remainder /= dimension;
            lhs_offset += coordinate * plan.lhs_strides[axis - 1];
            rhs_offset += coordinate * plan.rhs_strides[axis - 1];
        }
        const bool lhs_value = lhs[lhs_offset] != 0;
        const bool rhs_value = rhs[rhs_offset] != 0;
        if constexpr (Operation == LogicalBinaryOperation::And) {
            destination[index] =
                static_cast<std::uint8_t>(lhs_value && rhs_value);
        } else {
            destination[index] =
                static_cast<std::uint8_t>(lhs_value || rhs_value);
        }
    }
}

}  // namespace

void GenericBackend::logical_not_into(GenericBackend& output) const {
    if (quantization_mode_ != QuantizationMode::BOOL ||
        output.quantization_mode_ != QuantizationMode::BOOL) {
        throw std::runtime_error(
            "Logical NOT requires BOOL input and output tensors");
    }
    if (size_ != output.size_) {
        throw std::runtime_error(
            "Logical NOT input and output sizes must match");
    }

    const std::uint8_t* input_values = data_bool();
    std::uint8_t* output_values = output.data_bool();
    for (std::size_t i = 0; i < size_; ++i) {
        output_values[i] =
            input_values[i] == 0 ? std::uint8_t{1} : std::uint8_t{0};
    }
}

void GenericBackend::logical_and_broadcast(
    const GenericBackend& other, GenericBackend& output,
    const detail::BroadcastPlan& plan, std::size_t lhs_numel,
    std::size_t rhs_numel) const {
    logical_binary_broadcast<LogicalBinaryOperation::And>(
        *this, other, output, plan, lhs_numel, rhs_numel);
}

void GenericBackend::logical_or_broadcast(
    const GenericBackend& other, GenericBackend& output,
    const detail::BroadcastPlan& plan, std::size_t lhs_numel,
    std::size_t rhs_numel) const {
    logical_binary_broadcast<LogicalBinaryOperation::Or>(
        *this, other, output, plan, lhs_numel, rhs_numel);
}

}  // namespace labneura
