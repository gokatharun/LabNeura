#if defined(__AVX2__)

#include "labneura/backends/cpu/avx2.h"
#include "labneura/types.hpp"

#include <immintrin.h>
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

__m256 compare_fp32_vector(ComparisonType comparison, __m256 lhs,
                           __m256 rhs) {
    switch (comparison) {
        case ComparisonType::EQ:
            return _mm256_cmp_ps(lhs, rhs, _CMP_EQ_OQ);
        case ComparisonType::NE:
            return _mm256_cmp_ps(lhs, rhs, _CMP_NEQ_UQ);
        case ComparisonType::LT:
            return _mm256_cmp_ps(lhs, rhs, _CMP_LT_OQ);
        case ComparisonType::LE:
            return _mm256_cmp_ps(lhs, rhs, _CMP_LE_OQ);
        case ComparisonType::GT:
            return _mm256_cmp_ps(lhs, rhs, _CMP_GT_OQ);
        case ComparisonType::GE:
            return _mm256_cmp_ps(lhs, rhs, _CMP_GE_OQ);
    }
    throw std::invalid_argument("Unknown Tensor comparison operation");
}

__m256i compare_int16_vector(ComparisonType comparison, __m256i lhs,
                             __m256i rhs) {
    switch (comparison) {
        case ComparisonType::EQ:
            return _mm256_cmpeq_epi16(lhs, rhs);
        case ComparisonType::NE:
            return _mm256_xor_si256(_mm256_cmpeq_epi16(lhs, rhs),
                                    _mm256_set1_epi32(-1));
        case ComparisonType::LT:
            return _mm256_cmpgt_epi16(rhs, lhs);
        case ComparisonType::LE:
            return _mm256_xor_si256(_mm256_cmpgt_epi16(lhs, rhs),
                                    _mm256_set1_epi32(-1));
        case ComparisonType::GT:
            return _mm256_cmpgt_epi16(lhs, rhs);
        case ComparisonType::GE:
            return _mm256_xor_si256(_mm256_cmpgt_epi16(rhs, lhs),
                                    _mm256_set1_epi32(-1));
    }
    throw std::invalid_argument("Unknown Tensor comparison operation");
}

__m256i compare_int8_vector(ComparisonType comparison, __m256i lhs,
                            __m256i rhs) {
    switch (comparison) {
        case ComparisonType::EQ:
            return _mm256_cmpeq_epi8(lhs, rhs);
        case ComparisonType::NE:
            return _mm256_xor_si256(_mm256_cmpeq_epi8(lhs, rhs),
                                    _mm256_set1_epi32(-1));
        case ComparisonType::LT:
            return _mm256_cmpgt_epi8(rhs, lhs);
        case ComparisonType::LE:
            return _mm256_xor_si256(_mm256_cmpgt_epi8(lhs, rhs),
                                    _mm256_set1_epi32(-1));
        case ComparisonType::GT:
            return _mm256_cmpgt_epi8(lhs, rhs);
        case ComparisonType::GE:
            return _mm256_xor_si256(_mm256_cmpgt_epi8(rhs, lhs),
                                    _mm256_set1_epi32(-1));
    }
    throw std::invalid_argument("Unknown Tensor comparison operation");
}

void store_fp32_mask(__m256 mask, std::uint8_t* output,
                     std::size_t output_offset) {
    const int bits = _mm256_movemask_ps(mask);
    for (std::size_t lane = 0; lane < 8; ++lane) {
        output[output_offset + lane] =
            (bits & (1 << lane)) != 0 ? std::uint8_t{1} : std::uint8_t{0};
    }
}

void store_int16_mask(__m256i mask, std::uint8_t* output,
                      std::size_t output_offset) {
    const __m256i canonical =
        _mm256_and_si256(mask, _mm256_set1_epi16(1));
    const __m128i low = _mm256_castsi256_si128(canonical);
    const __m128i high = _mm256_extracti128_si256(canonical, 1);
    const __m128i packed = _mm_packus_epi16(low, high);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output + output_offset),
                     packed);
}

template <typename Element, std::size_t Lanes, typename VectorCompare,
          typename VectorStore>
void compare_integer_broadcast(const Element* lhs, const Element* rhs,
                               std::uint8_t* output,
                               const detail::BroadcastPlan& plan,
                               std::size_t count, ComparisonType comparison,
                               VectorCompare vector_compare,
                               VectorStore vector_store) {
    std::size_t index = 0;
    if (contiguous_in_plan(plan, plan.lhs_strides) &&
        contiguous_in_plan(plan, plan.rhs_strides)) {
        for (; index + Lanes <= count; index += Lanes) {
            const __m256i lhs_vector = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(lhs + index));
            const __m256i rhs_vector = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(rhs + index));
            vector_store(vector_compare(comparison, lhs_vector, rhs_vector),
                         output, index);
        }
    } else {
        alignas(32) std::array<Element, Lanes> lhs_values{};
        alignas(32) std::array<Element, Lanes> rhs_values{};
        for (; index + Lanes <= count; index += Lanes) {
            for (std::size_t lane = 0; lane < Lanes; ++lane) {
                std::size_t lhs_offset;
                std::size_t rhs_offset;
                broadcast_offsets(plan, index + lane, lhs_offset, rhs_offset);
                lhs_values[lane] = lhs[lhs_offset];
                rhs_values[lane] = rhs[rhs_offset];
            }
            const __m256i lhs_vector = _mm256_load_si256(
                reinterpret_cast<const __m256i*>(lhs_values.data()));
            const __m256i rhs_vector = _mm256_load_si256(
                reinterpret_cast<const __m256i*>(rhs_values.data()));
            vector_store(vector_compare(comparison, lhs_vector, rhs_vector),
                         output, index);
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

void compare_fp32(const float* lhs, const float* rhs, std::uint8_t* output,
                  const detail::BroadcastPlan& plan, std::size_t count,
                  ComparisonType comparison) {
    constexpr std::size_t lanes = 8;
    std::size_t index = 0;
    if (contiguous_in_plan(plan, plan.lhs_strides) &&
        contiguous_in_plan(plan, plan.rhs_strides)) {
        for (; index + lanes <= count; index += lanes) {
            store_fp32_mask(
                compare_fp32_vector(
                    comparison, _mm256_loadu_ps(lhs + index),
                    _mm256_loadu_ps(rhs + index)),
                output, index);
        }
    } else {
        alignas(32) std::array<float, lanes> lhs_values{};
        alignas(32) std::array<float, lanes> rhs_values{};
        for (; index + lanes <= count; index += lanes) {
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                std::size_t lhs_offset;
                std::size_t rhs_offset;
                broadcast_offsets(plan, index + lane, lhs_offset, rhs_offset);
                lhs_values[lane] = lhs[lhs_offset];
                rhs_values[lane] = rhs[rhs_offset];
            }
            store_fp32_mask(
                compare_fp32_vector(
                    comparison, _mm256_load_ps(lhs_values.data()),
                    _mm256_load_ps(rhs_values.data())),
                output, index);
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
    constexpr std::size_t lanes = 8;
    std::size_t index = 0;
    alignas(32) std::array<float, lanes> lhs_values{};
    alignas(32) std::array<float, lanes> rhs_values{};
    for (; index + lanes <= count; index += lanes) {
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            std::size_t lhs_offset;
            std::size_t rhs_offset;
            if (contiguous_in_plan(plan, plan.lhs_strides) &&
                contiguous_in_plan(plan, plan.rhs_strides)) {
                lhs_offset = index + lane;
                rhs_offset = index + lane;
            } else {
                broadcast_offsets(plan, index + lane, lhs_offset, rhs_offset);
            }
            lhs_values[lane] = dequantize_fp16(
                static_cast<std::uint16_t>(lhs[lhs_offset]));
            rhs_values[lane] = dequantize_fp16(
                static_cast<std::uint16_t>(rhs[rhs_offset]));
        }
        store_fp32_mask(
            compare_fp32_vector(comparison,
                                _mm256_load_ps(lhs_values.data()),
                                _mm256_load_ps(rhs_values.data())),
            output, index);
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

}  // namespace

void AVX2Backend::compare_broadcast(
    const AVX2Backend& other, AVX2Backend& output,
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
            "AVX2 comparisons support FP32, FP16, INT16, and INT8 tensors");
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
            compare_integer_broadcast<std::int16_t, 16>(
                data_int16(), other.data_int16(), output_data, plan,
                output_numel, comparison, compare_int16_vector,
                store_int16_mask);
            break;
        case QuantizationMode::INT8:
            compare_integer_broadcast<std::int8_t, 32>(
                data_int8(), other.data_int8(), output_data, plan, output_numel,
                comparison, compare_int8_vector,
                [](const __m256i mask, std::uint8_t* destination,
                   std::size_t offset) {
                    const __m256i canonical =
                        _mm256_and_si256(mask, _mm256_set1_epi8(1));
                    _mm256_storeu_si256(
                        reinterpret_cast<__m256i*>(destination + offset),
                        canonical);
                });
            break;
        default:
            throw std::runtime_error("Unsupported AVX2 comparison dtype");
    }
}

}  // namespace labneura

#endif
