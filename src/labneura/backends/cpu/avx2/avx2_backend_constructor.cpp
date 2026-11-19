#include "labneura/backends/common/cpu_features.h"
#include "labneura/backends/cpu/avx2.h"
#include "labneura/utils/alignment.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace labneura {
namespace {

std::shared_ptr<TensorStorage> make_avx2_storage(
    std::size_t size, QuantizationMode mode) {
    if (!cpu_supports_avx2()) {
        throw std::runtime_error(
            "AVX2 backend construction requires CPU/OS AVX2 support");
    }
    return TensorStorage::allocate(
        size, util::checked_round_up(size, 8), mode, util::AVX2_ALIGN);
}

}  // namespace

AVX2Backend::AVX2Backend(std::size_t size, QuantizationMode mode)
    : TensorBackend(make_avx2_storage(size, mode)),
      size_(size),
      aligned_size_(storage_->allocation_capacity),
      quantization_mode_(mode) {
    if (mode == QuantizationMode::FP32) {
        data_fp32_ = static_cast<float*>(storage_->data());
        std::fill(data_fp32_ + size_, data_fp32_ + aligned_size_, 0.0f);
    } else if (mode == QuantizationMode::FP16) {
        data_fp16_ = static_cast<int16_t*>(storage_->data());
        std::fill(data_fp16_ + size_, data_fp16_ + aligned_size_, int16_t{0});
    } else if (mode == QuantizationMode::INT16) {
        data_int16_ = static_cast<int16_t*>(storage_->data());
        std::fill(data_int16_ + size_, data_int16_ + aligned_size_, int16_t{0});
    } else if (mode == QuantizationMode::BOOL) {
        data_bool_ = static_cast<std::uint8_t*>(storage_->data());
        std::fill(data_bool_, data_bool_ + aligned_size_, std::uint8_t{0});
    } else {
        data_int8_ = static_cast<int8_t*>(storage_->data());
        std::fill(data_int8_ + size_, data_int8_ + aligned_size_, int8_t{0});
    }
}

}  // namespace labneura
