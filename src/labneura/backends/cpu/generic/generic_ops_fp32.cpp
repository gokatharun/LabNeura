#include "labneura/backends/cpu/generic.h"
#include "labneura/runtime.h"
#include <algorithm>

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

} // namespace labneura
