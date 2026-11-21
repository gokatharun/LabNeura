#pragma once

#include "labneura/tensor.h"
#include "labneura/visibility.h"

#include <cstddef>
#include <memory>

namespace labneura {

class LABNEURA_API TensorStorage final {
public:
    using Deleter = void (*)(void*);

    TensorStorage(const TensorStorage&) = delete;
    TensorStorage& operator=(const TensorStorage&) = delete;
    ~TensorStorage();

    static std::shared_ptr<TensorStorage> allocate(
        std::size_t logical_size, std::size_t allocation_capacity,
        QuantizationMode dtype, std::size_t alignment);
    static std::shared_ptr<TensorStorage> adopt(
        void* data, std::size_t logical_size,
        std::size_t allocation_capacity, QuantizationMode dtype,
        std::size_t alignment, Deleter deleter);

    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }

    const std::size_t logical_size;
    const std::size_t allocation_capacity;
    const QuantizationMode dtype;
    const std::size_t alignment;

private:
    TensorStorage(void* data, std::size_t logical_size,
                  std::size_t allocation_capacity, QuantizationMode dtype,
                  std::size_t alignment, Deleter deleter);

    void* data_;
    Deleter deleter_;
};

}  // namespace labneura
