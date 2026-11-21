#include "labneura/backends/cpu/generic.h"
#include "labneura/runtime.h"
#include "labneura/types.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace labneura {

namespace {

constexpr std::size_t kParallelAddThreshold = 1000000;
constexpr std::size_t kParallelAddGrainSize = 16384;

void add_fp32_range(float* destination, const float* source,
                    std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
        destination[i] += source[i];
    }
}

std::size_t checked_stride_advance(std::size_t dimension,
                                   std::size_t stride) {
    if (dimension <= 1 || stride == 0) {
        return 0;
    }
    if (dimension - 1 >
        std::numeric_limits<std::size_t>::max() / stride) {
        throw std::overflow_error("Broadcast input offset overflows size_t");
    }
    return (dimension - 1) * stride;
}

std::size_t checked_offset_add(std::size_t offset, std::size_t advance) {
    if (offset > std::numeric_limits<std::size_t>::max() - advance) {
        throw std::overflow_error("Broadcast input offset overflows size_t");
    }
    return offset + advance;
}

std::size_t maximum_broadcast_offset(
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& strides) {
    if (shape.size() != strides.size()) {
        throw std::invalid_argument("Broadcast output shape/stride rank mismatch");
    }
    std::size_t maximum = 0;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        maximum = checked_offset_add(
            maximum, checked_stride_advance(shape[axis], strides[axis]));
    }
    return maximum;
}

void validate_broadcast_range(const TensorStorage& storage,
                              std::size_t storage_offset,
                              std::size_t maximum_relative_offset) {
    if (storage_offset >= storage.logical_size ||
        maximum_relative_offset >= storage.logical_size - storage_offset) {
        throw std::out_of_range(
            "Broadcast input accesses beyond storage logical elements");
    }
}

template <typename Element>
Element apply_broadcast_operation(OperationType op_type, Element lhs,
                                  Element rhs) {
    if constexpr (std::is_same_v<Element, int8_t> ||
                  std::is_same_v<Element, int16_t>) {
        using Wide = std::conditional_t<std::is_same_v<Element, int8_t>,
                                        int16_t, int32_t>;
        Wide result = 0;
        switch (op_type) {
            case OperationType::ADD:
                result = static_cast<Wide>(lhs) + static_cast<Wide>(rhs);
                break;
            case OperationType::SUB:
                result = static_cast<Wide>(lhs) - static_cast<Wide>(rhs);
                break;
            case OperationType::MUL:
                result = static_cast<Wide>(lhs) * static_cast<Wide>(rhs);
                break;
            case OperationType::DIV:
                if (rhs == 0) {
                    return 0;
                }
                result = static_cast<Wide>(lhs) / static_cast<Wide>(rhs);
                break;
            default:
                throw std::invalid_argument(
                    "Unsupported broadcast arithmetic operation");
        }
        const Wide minimum = static_cast<Wide>(
            std::numeric_limits<Element>::min());
        const Wide maximum = static_cast<Wide>(
            std::numeric_limits<Element>::max());
        return static_cast<Element>(std::max(minimum, std::min(maximum, result)));
    } else {
        switch (op_type) {
            case OperationType::ADD: return lhs + rhs;
            case OperationType::SUB: return lhs - rhs;
            case OperationType::MUL: return lhs * rhs;
            case OperationType::DIV: return lhs / rhs;
        }
    }
    throw std::invalid_argument("Unsupported broadcast arithmetic operation");
}

template <typename Element>
void execute_broadcast(const Element* lhs_data, std::size_t lhs_storage_offset,
                       const Element* rhs_data, std::size_t rhs_storage_offset,
                       Element* output_data,
                       std::size_t output_storage_offset,
                       std::size_t output_numel,
                       const detail::BroadcastPlan& plan,
                       OperationType op_type) {
    for (std::size_t linear_index = 0; linear_index < output_numel;
         ++linear_index) {
        std::size_t remainder = linear_index;
        std::size_t lhs_offset = lhs_storage_offset;
        std::size_t rhs_offset = rhs_storage_offset;
        for (std::size_t axis = plan.shape.size(); axis > 0; --axis) {
            const std::size_t dimension = plan.shape[axis - 1];
            const std::size_t coordinate = remainder % dimension;
            remainder /= dimension;
            lhs_offset += coordinate * plan.lhs_strides[axis - 1];
            rhs_offset += coordinate * plan.rhs_strides[axis - 1];
        }
        output_data[output_storage_offset + linear_index] =
            apply_broadcast_operation(op_type, lhs_data[lhs_offset],
                                      rhs_data[rhs_offset]);
    }
}

inline float bit_pattern_to_fp16_float(uint16_t bits) {
    return dequantize_fp16(bits);
}

inline uint16_t fp16_float_to_bit_pattern(float value) {
    return quantize_fp16(value);
}

inline void apply_fp16_broadcast_element(OperationType op_type,
                                        const int16_t* lhs_data,
                                        std::size_t lhs_offset,
                                        const int16_t* rhs_data,
                                        std::size_t rhs_offset,
                                        int16_t* output_data,
                                        std::size_t output_offset) {
    uint16_t lhs_bits = static_cast<uint16_t>(lhs_data[lhs_offset]);
    uint16_t rhs_bits = static_cast<uint16_t>(rhs_data[rhs_offset]);
    const float lhs_value = bit_pattern_to_fp16_float(lhs_bits);
    const float rhs_value = bit_pattern_to_fp16_float(rhs_bits);
    float result = 0.0f;
    switch (op_type) {
        case OperationType::ADD: result = lhs_value + rhs_value; break;
        case OperationType::SUB: result = lhs_value - rhs_value; break;
        case OperationType::MUL: result = lhs_value * rhs_value; break;
        case OperationType::DIV: result = lhs_value / rhs_value; break;
        default: throw std::invalid_argument("Unsupported FP16 broadcast arithmetic operation");
    }
    uint16_t out_bits = fp16_float_to_bit_pattern(result);
    std::memcpy(&output_data[output_offset], &out_bits, sizeof(out_bits));
}

void execute_fp16_broadcast(const int16_t* lhs_data, std::size_t lhs_storage_offset,
                            const int16_t* rhs_data, std::size_t rhs_storage_offset,
                            int16_t* output_data,
                            std::size_t output_storage_offset,
                            std::size_t output_numel,
                            const detail::BroadcastPlan& plan,
                            OperationType op_type) {
    for (std::size_t linear_index = 0; linear_index < output_numel; ++linear_index) {
        std::size_t remainder = linear_index;
        std::size_t lhs_offset = lhs_storage_offset;
        std::size_t rhs_offset = rhs_storage_offset;
        for (std::size_t axis = plan.shape.size(); axis > 0; --axis) {
            const std::size_t dimension = plan.shape[axis - 1];
            const std::size_t coordinate = remainder % dimension;
            remainder /= dimension;
            lhs_offset += coordinate * plan.lhs_strides[axis - 1];
            rhs_offset += coordinate * plan.rhs_strides[axis - 1];
        }
        apply_fp16_broadcast_element(op_type, lhs_data, lhs_offset, rhs_data,
                                    rhs_offset, output_data,
                                    output_storage_offset + linear_index);
    }
}

}  // namespace

void GenericBackend::operation_fp32(const TensorBackend& other, OperationType op_type) {
    const float* other_data = other.data_fp32();
    float* this_data = data_fp32();

    if (op_type == OperationType::ADD) {
        if (size_ >= kParallelAddThreshold &&
            execution_thread_count() > 1) {
            parallel_for(0, size_, kParallelAddGrainSize,
                         [this_data, other_data](std::size_t begin,
                                                 std::size_t end) {
                             add_fp32_range(this_data, other_data, begin, end);
                         });
        } else {
            add_fp32_range(this_data, other_data, 0, size_);
        }
        return;
    }
    if (op_type == OperationType::MUL) {
        for (std::size_t i = 0; i < size_; ++i) this_data[i] *= other_data[i];
        return;
    }
    if (op_type == OperationType::DIV) {
        for (std::size_t i = 0; i < size_; ++i) this_data[i] /= other_data[i];
        return;
    }
    for (std::size_t i = 0; i < size_; ++i) this_data[i] -= other_data[i];
}

void GenericBackend::operation_fp16(const TensorBackend& other, OperationType op_type) {
    const int16_t* other_data = other.data_fp16();
    int16_t* this_data = data_fp16();

    for (std::size_t i = 0; i < size_; ++i) {
        const uint16_t lhs_bits = static_cast<uint16_t>(this_data[i]);
        const uint16_t rhs_bits = static_cast<uint16_t>(other_data[i]);
        const float lhs = dequantize_fp16(lhs_bits);
        const float rhs = dequantize_fp16(rhs_bits);
        float result = 0.0f;
        switch (op_type) {
            case OperationType::ADD: result = lhs + rhs; break;
            case OperationType::SUB: result = lhs - rhs; break;
            case OperationType::MUL: result = lhs * rhs; break;
            case OperationType::DIV: result = lhs / rhs; break;
            default: throw std::invalid_argument("Unsupported FP16 operation");
        }
        const uint16_t out_bits = quantize_fp16(result);
        std::memcpy(&this_data[i], &out_bits, sizeof(out_bits));
    }
}

bool GenericBackend::operation_into(const TensorBackend& other, TensorBackend& out,
                                    OperationType op_type) const {
    if (quantization_mode_ != QuantizationMode::FP32 ||
        op_type != OperationType::ADD) {
        return false;
    }
    if (out.size() != size_ || out.quantization_mode() != quantization_mode_) {
        throw std::runtime_error("Output tensor does not match operand size or mode");
    }

    float* destination = out.data_fp32();
    const float* left = data_fp32();
    const float* right = other.data_fp32();
    const auto add_range = [destination, left, right](std::size_t begin,
                                                       std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            destination[i] = left[i] + right[i];
        }
    };
    if (size_ >= kParallelAddThreshold && execution_thread_count() > 1) {
        parallel_for(0, size_, kParallelAddGrainSize, add_range);
    } else {
        add_range(0, size_);
    }
    return true;
}

void GenericBackend::operation_broadcast(
    const GenericBackend& other, GenericBackend& output,
    const detail::BroadcastPlan& plan, OperationType op_type,
    std::size_t lhs_numel, std::size_t rhs_numel) const {
    if (quantization_mode_ != other.quantization_mode_ ||
        quantization_mode_ != output.quantization_mode_ ||
        (quantization_mode_ != QuantizationMode::FP32 &&
         quantization_mode_ != QuantizationMode::FP16 &&
         quantization_mode_ != QuantizationMode::INT8 &&
         quantization_mode_ != QuantizationMode::INT16)) {
        throw std::runtime_error(
            "Generic broadcasting supports matching FP32, FP16, INT8, or INT16 tensors");
    }
    if (size_ != lhs_numel || other.size_ != rhs_numel) {
        throw std::runtime_error(
            "Broadcast operand logical element count mismatch");
    }
    if (plan.shape.size() != plan.lhs_strides.size() ||
        plan.shape.size() != plan.rhs_strides.size()) {
        throw std::invalid_argument(
            "Broadcast plan shape and stride ranks must match");
    }

    const std::size_t output_numel =
        detail::checked_broadcast_numel(plan.shape);
    if (output.size_ != output_numel) {
        throw std::runtime_error(
            "Broadcast output logical element count mismatch");
    }
    if (output_numel == 0) {
        return;
    }

    const std::size_t lhs_max_offset =
        maximum_broadcast_offset(plan.shape, plan.lhs_strides);
    const std::size_t rhs_max_offset =
        maximum_broadcast_offset(plan.shape, plan.rhs_strides);
    validate_broadcast_range(*storage_, storage_offset_, lhs_max_offset);
    validate_broadcast_range(*other.storage_, other.storage_offset_,
                             rhs_max_offset);
    validate_broadcast_range(*output.storage_, output.storage_offset_,
                             output_numel - 1);

    if (quantization_mode_ == QuantizationMode::FP32) {
        execute_broadcast(
            static_cast<const float*>(storage_->data()), storage_offset_,
            static_cast<const float*>(other.storage_->data()),
            other.storage_offset_,
            static_cast<float*>(output.storage_->data()), output.storage_offset_,
            output_numel, plan, op_type);
    } else if (quantization_mode_ == QuantizationMode::FP16) {
        execute_fp16_broadcast(
            static_cast<const int16_t*>(storage_->data()), storage_offset_,
            static_cast<const int16_t*>(other.storage_->data()),
            other.storage_offset_,
            static_cast<int16_t*>(output.storage_->data()), output.storage_offset_,
            output_numel, plan, op_type);
    } else if (quantization_mode_ == QuantizationMode::INT8) {
        execute_broadcast(
            static_cast<const int8_t*>(storage_->data()), storage_offset_,
            static_cast<const int8_t*>(other.storage_->data()),
            other.storage_offset_,
            static_cast<int8_t*>(output.storage_->data()), output.storage_offset_,
            output_numel, plan, op_type);
    } else {
        execute_broadcast(
            static_cast<const int16_t*>(storage_->data()), storage_offset_,
            static_cast<const int16_t*>(other.storage_->data()),
            other.storage_offset_,
            static_cast<int16_t*>(output.storage_->data()),
            output.storage_offset_, output_numel, plan, op_type);
    }
}

} // namespace labneura
