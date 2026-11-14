#if defined(__ARM_NEON) || defined(__aarch64__)

#include "labneura/backends/cpu/neon.h"
#include "labneura/types.hpp"

#include <arm_neon.h>
#include <array>
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

bool compare_float(ComparisonType comparison, float lhs, float rhs) {
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

template <typename Element>
bool compare_integer(ComparisonType comparison, Element lhs, Element rhs) {
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

uint32x4_t compare_fp32_vector(ComparisonType comparison,
                               float32x4_t lhs, float32x4_t rhs) {
    switch (comparison) {
        case ComparisonType::EQ: return vceqq_f32(lhs, rhs);
        case ComparisonType::NE: return vmvnq_u32(vceqq_f32(lhs, rhs));
        case ComparisonType::LT: return vcltq_f32(lhs, rhs);
        case ComparisonType::LE: return vcleq_f32(lhs, rhs);
        case ComparisonType::GT: return vcgtq_f32(lhs, rhs);
        case ComparisonType::GE: return vcgeq_f32(lhs, rhs);
    }
    throw std::invalid_argument("Unknown Tensor comparison operation");
}

uint16x8_t compare_int16_vector(ComparisonType comparison,
                                int16x8_t lhs, int16x8_t rhs) {
    switch (comparison) {
        case ComparisonType::EQ: return vceqq_s16(lhs, rhs);
        case ComparisonType::NE: return vmvnq_u16(vceqq_s16(lhs, rhs));
        case ComparisonType::LT: return vcltq_s16(lhs, rhs);
        case ComparisonType::LE: return vcleq_s16(lhs, rhs);
        case ComparisonType::GT: return vcgtq_s16(lhs, rhs);
        case ComparisonType::GE: return vcgeq_s16(lhs, rhs);
    }
    throw std::invalid_argument("Unknown Tensor comparison operation");
}

uint8x16_t compare_int8_vector(ComparisonType comparison,
                               int8x16_t lhs, int8x16_t rhs) {
    switch (comparison) {
        case ComparisonType::EQ: return vceqq_s8(lhs, rhs);
        case ComparisonType::NE: return vmvnq_u8(vceqq_s8(lhs, rhs));
        case ComparisonType::LT: return vcltq_s8(lhs, rhs);
        case ComparisonType::LE: return vcleq_s8(lhs, rhs);
        case ComparisonType::GT: return vcgtq_s8(lhs, rhs);
        case ComparisonType::GE: return vcgeq_s8(lhs, rhs);
    }
    throw std::invalid_argument("Unknown Tensor comparison operation");
}

void compare_fp32(const float* lhs, const float* rhs, std::uint8_t* output,
                  const detail::BroadcastPlan& plan, std::size_t count,
                  ComparisonType comparison) {
    constexpr std::size_t lanes = 4;
    if (contiguous_in_plan(plan, plan.lhs_strides) &&
        contiguous_in_plan(plan, plan.rhs_strides)) {
        std::size_t index = 0;
        for (; index + lanes <= count; index += lanes) {
            const uint32x4_t mask = compare_fp32_vector(
                comparison, vld1q_f32(lhs + index), vld1q_f32(rhs + index));
            std::array<std::uint32_t, lanes> mask_lanes{};
            vst1q_u32(mask_lanes.data(), mask);
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                output[index + lane] = mask_lanes[lane] != 0 ? 1 : 0;
            }
        }
        for (; index < count; ++index) {
            output[index] = compare_float(comparison, lhs[index], rhs[index])
                                ? 1
                                : 0;
        }
        return;
    }

    std::size_t index = 0;
    for (; index + lanes <= count; index += lanes) {
        std::array<float, lanes> lhs_lanes{};
        std::array<float, lanes> rhs_lanes{};
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            std::size_t lhs_offset;
            std::size_t rhs_offset;
            broadcast_offsets(plan, index + lane, lhs_offset, rhs_offset);
            lhs_lanes[lane] = lhs[lhs_offset];
            rhs_lanes[lane] = rhs[rhs_offset];
        }
        const uint32x4_t mask = compare_fp32_vector(
            comparison, vld1q_f32(lhs_lanes.data()),
            vld1q_f32(rhs_lanes.data()));
        std::array<std::uint32_t, lanes> mask_lanes{};
        vst1q_u32(mask_lanes.data(), mask);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            output[index + lane] = mask_lanes[lane] != 0 ? 1 : 0;
        }
    }
    for (; index < count; ++index) {
        std::size_t lhs_offset;
        std::size_t rhs_offset;
        broadcast_offsets(plan, index, lhs_offset, rhs_offset);
        output[index] =
            compare_float(comparison, lhs[lhs_offset], rhs[rhs_offset]) ? 1 : 0;
    }
}

void compare_fp16(const std::int16_t* lhs, const std::int16_t* rhs,
                  std::uint8_t* output, const detail::BroadcastPlan& plan,
                  std::size_t count, ComparisonType comparison) {
    constexpr std::size_t lanes = 4;
    std::size_t index = 0;
    for (; index + lanes <= count; index += lanes) {
        std::array<float, lanes> lhs_lanes{};
        std::array<float, lanes> rhs_lanes{};
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            std::size_t lhs_offset;
            std::size_t rhs_offset;
            broadcast_offsets(plan, index + lane, lhs_offset, rhs_offset);
            lhs_lanes[lane] = dequantize_fp16(
                static_cast<std::uint16_t>(lhs[lhs_offset]));
            rhs_lanes[lane] = dequantize_fp16(
                static_cast<std::uint16_t>(rhs[rhs_offset]));
        }
        const uint32x4_t mask = compare_fp32_vector(
            comparison, vld1q_f32(lhs_lanes.data()),
            vld1q_f32(rhs_lanes.data()));
        std::array<std::uint32_t, lanes> mask_lanes{};
        vst1q_u32(mask_lanes.data(), mask);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            output[index + lane] = mask_lanes[lane] != 0 ? 1 : 0;
        }
    }
    for (; index < count; ++index) {
        std::size_t lhs_offset;
        std::size_t rhs_offset;
        broadcast_offsets(plan, index, lhs_offset, rhs_offset);
        const float lhs_value = dequantize_fp16(
            static_cast<std::uint16_t>(lhs[lhs_offset]));
        const float rhs_value = dequantize_fp16(
            static_cast<std::uint16_t>(rhs[rhs_offset]));
        output[index] =
            compare_float(comparison, lhs_value, rhs_value) ? 1 : 0;
    }
}

void compare_int16(const std::int16_t* lhs, const std::int16_t* rhs,
                   std::uint8_t* output, const detail::BroadcastPlan& plan,
                   std::size_t count, ComparisonType comparison) {
    constexpr std::size_t lanes = 8;
    const uint16x8_t one = vdupq_n_u16(1);
    if (contiguous_in_plan(plan, plan.lhs_strides) &&
        contiguous_in_plan(plan, plan.rhs_strides)) {
        std::size_t index = 0;
        for (; index + lanes <= count; index += lanes) {
            const uint16x8_t mask = vandq_u16(
                compare_int16_vector(comparison, vld1q_s16(lhs + index),
                                     vld1q_s16(rhs + index)),
                one);
            std::array<std::uint16_t, lanes> mask_lanes{};
            vst1q_u16(mask_lanes.data(), mask);
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                output[index + lane] =
                    static_cast<std::uint8_t>(mask_lanes[lane]);
            }
        }
        for (; index < count; ++index) {
            output[index] =
                compare_integer(comparison, lhs[index], rhs[index]) ? 1 : 0;
        }
        return;
    }

    std::size_t index = 0;
    for (; index + lanes <= count; index += lanes) {
        std::array<std::int16_t, lanes> lhs_lanes{};
        std::array<std::int16_t, lanes> rhs_lanes{};
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            std::size_t lhs_offset;
            std::size_t rhs_offset;
            broadcast_offsets(plan, index + lane, lhs_offset, rhs_offset);
            lhs_lanes[lane] = lhs[lhs_offset];
            rhs_lanes[lane] = rhs[rhs_offset];
        }
        const uint16x8_t mask = vandq_u16(
            compare_int16_vector(comparison, vld1q_s16(lhs_lanes.data()),
                                vld1q_s16(rhs_lanes.data())),
            one);
        std::array<std::uint16_t, lanes> mask_lanes{};
        vst1q_u16(mask_lanes.data(), mask);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            output[index + lane] =
                static_cast<std::uint8_t>(mask_lanes[lane]);
        }
    }
    for (; index < count; ++index) {
        std::size_t lhs_offset;
        std::size_t rhs_offset;
        broadcast_offsets(plan, index, lhs_offset, rhs_offset);
        output[index] =
            compare_integer(comparison, lhs[lhs_offset], rhs[rhs_offset]) ? 1
                                                                         : 0;
    }
}

void compare_int8(const std::int8_t* lhs, const std::int8_t* rhs,
                  std::uint8_t* output, const detail::BroadcastPlan& plan,
                  std::size_t count, ComparisonType comparison) {
    constexpr std::size_t lanes = 16;
    const uint8x16_t one = vdupq_n_u8(1);
    if (contiguous_in_plan(plan, plan.lhs_strides) &&
        contiguous_in_plan(plan, plan.rhs_strides)) {
        std::size_t index = 0;
        for (; index + lanes <= count; index += lanes) {
            const uint8x16_t mask = vandq_u8(
                compare_int8_vector(comparison, vld1q_s8(lhs + index),
                                    vld1q_s8(rhs + index)),
                one);
            vst1q_u8(output + index, mask);
        }
        for (; index < count; ++index) {
            output[index] =
                compare_integer(comparison, lhs[index], rhs[index]) ? 1 : 0;
        }
        return;
    }

    std::size_t index = 0;
    for (; index + lanes <= count; index += lanes) {
        std::array<std::int8_t, lanes> lhs_lanes{};
        std::array<std::int8_t, lanes> rhs_lanes{};
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            std::size_t lhs_offset;
            std::size_t rhs_offset;
            broadcast_offsets(plan, index + lane, lhs_offset, rhs_offset);
            lhs_lanes[lane] = lhs[lhs_offset];
            rhs_lanes[lane] = rhs[rhs_offset];
        }
        const uint8x16_t mask = vandq_u8(
            compare_int8_vector(comparison, vld1q_s8(lhs_lanes.data()),
                               vld1q_s8(rhs_lanes.data())),
            one);
        vst1q_u8(output + index, mask);
    }
    for (; index < count; ++index) {
        std::size_t lhs_offset;
        std::size_t rhs_offset;
        broadcast_offsets(plan, index, lhs_offset, rhs_offset);
        output[index] =
            compare_integer(comparison, lhs[lhs_offset], rhs[rhs_offset]) ? 1
                                                                         : 0;
    }
}

}  // namespace

void NEONBackend::compare_broadcast(
    const NEONBackend& other, NEONBackend& output,
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
            "NEON comparisons support FP32, FP16, INT16, and INT8 tensors");
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
    switch (quantization_mode_) {
        case QuantizationMode::FP32:
            compare_fp32(data_fp32(), other.data_fp32(), output_data, plan,
                         output_numel, comparison);
            break;
        case QuantizationMode::FP16:
            compare_fp16(data_fp16(), other.data_fp16(), output_data, plan,
                         output_numel, comparison);
            break;
        case QuantizationMode::INT16:
            compare_int16(data_int16(), other.data_int16(), output_data, plan,
                          output_numel, comparison);
            break;
        case QuantizationMode::INT8:
            compare_int8(data_int8(), other.data_int8(), output_data, plan,
                         output_numel, comparison);
            break;
        default:
            throw std::runtime_error("Unsupported NEON comparison dtype");
    }
}

}  // namespace labneura

#endif
