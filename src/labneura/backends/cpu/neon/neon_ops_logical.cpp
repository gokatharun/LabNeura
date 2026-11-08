#if defined(__ARM_NEON) || defined(__aarch64__)

#include "labneura/backends/cpu/neon.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace labneura {
namespace {

std::size_t maximum_input_offset(
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& strides) {
    if (shape.size() != strides.size()) {
        throw std::invalid_argument(
            "Logical AND broadcast shape and stride ranks must match");
    }

    std::size_t maximum = 0;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        if (shape[axis] <= 1 || strides[axis] == 0) {
            continue;
        }
        const std::size_t count = shape[axis] - 1;
        if (count > std::numeric_limits<std::size_t>::max() / strides[axis]) {
            throw std::overflow_error(
                "Logical AND input offset overflows size_t");
        }
        const std::size_t advance = count * strides[axis];
        if (maximum >
            std::numeric_limits<std::size_t>::max() - advance) {
            throw std::overflow_error(
                "Logical AND input offset overflows size_t");
        }
        maximum += advance;
    }
    return maximum;
}

bool contiguous_in_plan(const detail::BroadcastPlan& plan,
                        const std::vector<std::size_t>& strides) {
    std::size_t expected_stride = 1;
    for (std::size_t axis = plan.shape.size(); axis > 0; --axis) {
        const std::size_t dimension = plan.shape[axis - 1];
        if (dimension > 1 && strides[axis - 1] != expected_stride) {
            return false;
        }
        expected_stride *= dimension;
    }
    return true;
}

void broadcast_offsets(const detail::BroadcastPlan& plan,
                       std::size_t linear_index,
                       std::size_t& lhs_offset,
                       std::size_t& rhs_offset) {
    lhs_offset = 0;
    rhs_offset = 0;
    for (std::size_t axis = plan.shape.size(); axis > 0; --axis) {
        const std::size_t dimension = plan.shape[axis - 1];
        const std::size_t coordinate = linear_index % dimension;
        linear_index /= dimension;
        lhs_offset += coordinate * plan.lhs_strides[axis - 1];
        rhs_offset += coordinate * plan.rhs_strides[axis - 1];
    }
}

}  // namespace

void NEONBackend::logical_not_into(NEONBackend& output) const {
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
    const uint8x16_t zero = vdupq_n_u8(0);
    const uint8x16_t one = vdupq_n_u8(1);
    std::size_t index = 0;
    for (; size_ - index >= 16; index += 16) {
        const uint8x16_t input = vld1q_u8(input_values + index);
        const uint8x16_t is_false = vceqq_u8(input, zero);
        vst1q_u8(output_values + index, vandq_u8(is_false, one));
    }
    for (; index < size_; ++index) {
        output_values[index] =
            input_values[index] == 0 ? std::uint8_t{1} : std::uint8_t{0};
    }
}

void NEONBackend::logical_and_broadcast(
    const NEONBackend& other, NEONBackend& output,
    const detail::BroadcastPlan& plan, std::size_t lhs_numel,
    std::size_t rhs_numel) const {
    if (quantization_mode_ != QuantizationMode::BOOL ||
        other.quantization_mode_ != QuantizationMode::BOOL ||
        output.quantization_mode_ != QuantizationMode::BOOL) {
        throw std::runtime_error(
            "Logical AND requires BOOL input and output tensors");
    }
    if (size_ != lhs_numel || other.size_ != rhs_numel) {
        throw std::runtime_error(
            "Logical AND operand logical element count mismatch");
    }
    if (plan.shape.size() != plan.lhs_strides.size() ||
        plan.shape.size() != plan.rhs_strides.size()) {
        throw std::invalid_argument(
            "Logical AND broadcast plan shape and stride ranks must match");
    }

    const std::size_t output_numel =
        detail::checked_broadcast_numel(plan.shape);
    if (output.size_ != output_numel) {
        throw std::runtime_error(
            "Logical AND output logical element count mismatch");
    }
    if (output_numel == 0) {
        return;
    }
    if (maximum_input_offset(plan.shape, plan.lhs_strides) >= size_ ||
        maximum_input_offset(plan.shape, plan.rhs_strides) >= other.size_) {
        throw std::out_of_range(
            "Logical AND broadcast accesses beyond input logical elements");
    }

    const std::uint8_t* lhs = data_bool();
    const std::uint8_t* rhs = other.data_bool();
    std::uint8_t* destination = output.data_bool();
    const bool contiguous =
        size_ == output_numel && other.size_ == output_numel &&
        contiguous_in_plan(plan, plan.lhs_strides) &&
        contiguous_in_plan(plan, plan.rhs_strides);
    std::size_t index = 0;
    for (; output_numel - index >= 16; index += 16) {
        uint8x16_t lhs_values;
        uint8x16_t rhs_values;
        if (contiguous) {
            lhs_values = vld1q_u8(lhs + index);
            rhs_values = vld1q_u8(rhs + index);
        } else {
            alignas(16) std::uint8_t lhs_lanes[16];
            alignas(16) std::uint8_t rhs_lanes[16];
            for (std::size_t lane = 0; lane < 16; ++lane) {
                std::size_t lhs_offset;
                std::size_t rhs_offset;
                broadcast_offsets(plan, index + lane, lhs_offset, rhs_offset);
                lhs_lanes[lane] = lhs[lhs_offset];
                rhs_lanes[lane] = rhs[rhs_offset];
            }
            lhs_values = vld1q_u8(lhs_lanes);
            rhs_values = vld1q_u8(rhs_lanes);
        }
        vst1q_u8(destination + index, vandq_u8(lhs_values, rhs_values));
    }
    for (; index < output_numel; ++index) {
        std::size_t lhs_offset;
        std::size_t rhs_offset;
        broadcast_offsets(plan, index, lhs_offset, rhs_offset);
        destination[index] =
            static_cast<std::uint8_t>(lhs[lhs_offset] & rhs[rhs_offset]);
    }
}

}  // namespace labneura

#endif
