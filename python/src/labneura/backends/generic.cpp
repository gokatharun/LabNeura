#include "labneura/backends/generic.h"
#include <cstdlib>
#include "labneura/utils/alignment.hpp"

namespace labneura {

GenericBackend::GenericBackend(std::size_t size, QuantizationMode mode)
    : size_(size), quantization_mode_(mode) {
    // 64-byte aligned allocation with fallback to malloc (generic fallback backend)
    auto free_aligned_float = [](float* p){ if (p) std::free(p); };
    auto free_aligned_int8 = [](int8_t* p){ if (p) std::free(p); };
    auto allocate_aligned = [](std::size_t bytes){
        void* p = nullptr;
        if (posix_memalign(&p, labneura::util::DEFAULT_ALIGN, bytes) == 0) return p;
        return std::malloc(bytes);
    };
    if (mode == QuantizationMode::FP32) {
        float* p = static_cast<float*>(allocate_aligned(size * sizeof(float)));
        data_fp32_ = std::unique_ptr<float, void(*)(float*)>(p, free_aligned_float);
    } else if (mode == QuantizationMode::INT8) {
        int8_t* p = static_cast<int8_t*>(allocate_aligned(size * sizeof(int8_t)));
        data_int8_ = std::unique_ptr<int8_t, void(*)(int8_t*)>(p, free_aligned_int8);
    }
}

std::size_t GenericBackend::size() const {
    return size_;
}

QuantizationMode GenericBackend::quantization_mode() const {
    return quantization_mode_;
}

float* GenericBackend::data_fp32() {
    if (quantization_mode_ != QuantizationMode::FP32) {
        throw std::runtime_error("Tensor is not in FP32 mode");
    }
    return data_fp32_.get();
}

const float* GenericBackend::data_fp32() const {
    if (quantization_mode_ != QuantizationMode::FP32) {
        throw std::runtime_error("Tensor is not in FP32 mode");
    }
    return data_fp32_.get();
}

int8_t* GenericBackend::data_int8() {
    if (quantization_mode_ != QuantizationMode::INT8) {
        throw std::runtime_error("Tensor is not in INT8 mode");
    }
    return data_int8_.get();
}

const int8_t* GenericBackend::data_int8() const {
    if (quantization_mode_ != QuantizationMode::INT8) {
        throw std::runtime_error("Tensor is not in INT8 mode");
    }
    return data_int8_.get();
}

std::unique_ptr<TensorBackend> GenericBackend::clone() const {
    auto backend = std::make_unique<GenericBackend>(size_, quantization_mode_);
    if (quantization_mode_ == QuantizationMode::FP32) {
        std::copy(data_fp32_.get(), data_fp32_.get() + size_, 
                 backend->data_fp32());
    } else if (quantization_mode_ == QuantizationMode::INT8) {
        std::copy(data_int8_.get(), data_int8_.get() + size_, 
                 backend->data_int8());
    }
    return backend;
}

void GenericBackend::operation_fp32(const TensorBackend& other, OperationType op_type) {
    const float* other_data = other.data_fp32();
    float* this_data = data_fp32();

    if (op_type == OperationType::ADD) {
        for (std::size_t i = 0; i < size_; ++i) this_data[i] += other_data[i];
        return;
    }
    if (op_type == OperationType::MUL) {
        for (std::size_t i = 0; i < size_; ++i) this_data[i] *= other_data[i];
        return;
    }
    for (std::size_t i = 0; i < size_; ++i) this_data[i] -= other_data[i];
}
void GenericBackend::operation_int8(const TensorBackend& other, OperationType op_type) {
    const int8_t* other_data = other.data_int8();
    int8_t* this_data = data_int8();

    if (op_type == OperationType::ADD) {
        for (std::size_t i = 0; i < size_; ++i) {
            int16_t sum = static_cast<int16_t>(this_data[i]) + static_cast<int16_t>(other_data[i]);
            this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(sum))));
        }
        return;
    }
    if (op_type == OperationType::MUL) {
        for (std::size_t i = 0; i < size_; ++i) {
            int16_t prod = static_cast<int16_t>(this_data[i]) * static_cast<int16_t>(other_data[i]);
            this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(prod))));
        }
        return;
    }
    for (std::size_t i = 0; i < size_; ++i) {
        int16_t diff = static_cast<int16_t>(this_data[i]) - static_cast<int16_t>(other_data[i]);
        this_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(diff))));
    }
}

}  // namespace labneura


