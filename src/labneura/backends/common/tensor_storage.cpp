#include "labneura/backends/common/tensor_storage.h"
#include "labneura/backends/common/dtype_validation.h"

#include "labneura/utils/alignment.hpp"

#include <limits>
#include <new>
#include <stdexcept>

namespace labneura {
namespace {

std::size_t dtype_size(QuantizationMode dtype) {
    switch (dtype) {
        case QuantizationMode::FP32:
            return sizeof(float);
        case QuantizationMode::FP16:
        case QuantizationMode::INT16:
            return sizeof(int16_t);
        case QuantizationMode::INT8:
            return sizeof(int8_t);
        case QuantizationMode::BOOL:
            return sizeof(std::uint8_t);
        case QuantizationMode::FP64:
        case QuantizationMode::INT32:
            throw std::invalid_argument("Unsupported tensor dtype");
    }
    throw std::invalid_argument("Unsupported tensor dtype");
}

void* allocate_data(std::size_t count, QuantizationMode dtype,
                    std::size_t alignment) {
    switch (dtype) {
        case QuantizationMode::FP32:
            return util::allocate_aligned<float>(count, alignment);
        case QuantizationMode::FP16:
        case QuantizationMode::INT16:
            return util::allocate_aligned<int16_t>(count, alignment);
        case QuantizationMode::INT8:
            return util::allocate_aligned<int8_t>(count, alignment);
        case QuantizationMode::BOOL:
            return util::allocate_aligned<std::uint8_t>(count, alignment);
        case QuantizationMode::FP64:
        case QuantizationMode::INT32:
            throw std::invalid_argument("Unsupported tensor dtype");
    }
    throw std::invalid_argument("Unsupported tensor dtype");
}

}  // namespace

TensorStorage::TensorStorage(void* data, std::size_t logical_size_value,
                             std::size_t allocation_capacity_value,
                             QuantizationMode dtype_value,
                             std::size_t alignment_value, Deleter deleter)
    : logical_size(logical_size_value),
      allocation_capacity(allocation_capacity_value),
      dtype(dtype_value),
      alignment(alignment_value),
      data_(data),
      deleter_(deleter) {
    validate_tensor_dtype_supported(dtype);
    if (data_ == nullptr || deleter_ == nullptr) {
        throw std::invalid_argument("TensorStorage requires owned data and a deleter");
    }
    if (allocation_capacity < logical_size) {
        throw std::invalid_argument(
            "TensorStorage capacity cannot be smaller than its logical size");
    }
    if (allocation_capacity >
        std::numeric_limits<std::size_t>::max() / dtype_size(dtype)) {
        throw std::bad_array_new_length();
    }
    if (alignment < sizeof(void*) ||
        (alignment & (alignment - 1)) != 0 ||
        !util::is_aligned(data_, alignment)) {
        throw std::invalid_argument("TensorStorage data does not meet its alignment");
    }
}

TensorStorage::~TensorStorage() {
    deleter_(data_);
}

std::shared_ptr<TensorStorage> TensorStorage::allocate(
    std::size_t logical_size, std::size_t allocation_capacity,
    QuantizationMode dtype, std::size_t alignment) {
    validate_tensor_dtype_supported(dtype);
    if (allocation_capacity < logical_size) {
        throw std::invalid_argument(
            "TensorStorage capacity cannot be smaller than its logical size");
    }
    if (allocation_capacity >
        std::numeric_limits<std::size_t>::max() / dtype_size(dtype)) {
        throw std::bad_array_new_length();
    }

    void* data = allocate_data(allocation_capacity, dtype, alignment);
    return adopt(data, logical_size, allocation_capacity, dtype, alignment,
                 util::aligned_free);
}

std::shared_ptr<TensorStorage> TensorStorage::adopt(
    void* data, std::size_t logical_size, std::size_t allocation_capacity,
    QuantizationMode dtype, std::size_t alignment, Deleter deleter) {
    TensorStorage* storage = nullptr;
    try {
        storage = new TensorStorage(
            data, logical_size, allocation_capacity, dtype, alignment, deleter);
    } catch (...) {
        if (data != nullptr && deleter != nullptr) {
            deleter(data);
        }
        throw;
    }
    return std::shared_ptr<TensorStorage>(storage);
}

}  // namespace labneura
