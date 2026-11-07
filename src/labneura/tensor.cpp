#include "labneura/tensor.h"
#include "labneura/backends/cpu/generic.h"
#include "labneura/backends/cpu/neon.h"
#include "labneura/backends/cpu/avx2.h"
#include "labneura/backends/common/backend_factory.h"
#include "labneura/backends/common/dtype_validation.h"
#include "labneura/detail/broadcast.h"
#include "labneura/types.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <typeinfo>

namespace labneura {

namespace {
std::size_t dtype_size(QuantizationMode mode) {
    switch (mode) {
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
            throw std::invalid_argument("Unsupported tensor dtype: FP64");
        case QuantizationMode::INT32:
            throw std::invalid_argument("Unsupported tensor dtype: INT32");
        default:
            throw std::invalid_argument("Unsupported tensor dtype: unknown");
    }
}

std::size_t checked_shape_numel(const std::vector<std::size_t>& shape) {
    if (std::find(shape.begin(), shape.end(), std::size_t{0}) != shape.end()) {
        return 0;
    }

    std::size_t element_count = 1;
    for (const std::size_t dimension : shape) {
        if (element_count >
            std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error(
                "Tensor shape element count overflows size_t");
        }
        element_count *= dimension;
    }
    return element_count;
}

void check_allocation_size(std::size_t size, QuantizationMode mode) {
    validate_tensor_dtype_supported(mode);
    const std::size_t element_bytes = dtype_size(mode);
    if (size > std::numeric_limits<std::size_t>::max() / element_bytes) {
        throw std::overflow_error("Tensor allocation byte size overflows size_t");
    }
}

std::size_t checked_offset_advance(std::size_t index,
                                   std::size_t stride) {
    if (stride != 0 &&
        index > std::numeric_limits<std::size_t>::max() / stride) {
        throw std::overflow_error("Tensor storage offset overflows size_t");
    }
    return index * stride;
}

std::size_t checked_offset_add(std::size_t offset, std::size_t advance) {
    if (offset > std::numeric_limits<std::size_t>::max() - advance) {
        throw std::overflow_error("Tensor storage offset overflows size_t");
    }
    return offset + advance;
}

std::vector<float> flatten_rows(const std::vector<std::vector<float>>& rows,
                                QuantizationMode mode) {
    validate_tensor_dtype_supported(mode);
    std::vector<float> flat;
    if (rows.empty()) return flat;
    const std::size_t cols = rows.front().size();
    const std::vector<std::size_t> shape{rows.size(), cols};
    const std::size_t element_count = checked_shape_numel(shape);
    check_allocation_size(element_count, mode);
    flat.reserve(element_count);
    for (const auto& row : rows) {
        if (row.size() != cols) {
            throw std::invalid_argument(
                "2-D Tensor rows must all have the same length");
        }
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}
}  // namespace

// Tensor implementation
Tensor::Tensor()
    : quantization_mode_(QuantizationMode::FP32),
      shape_{0},
      strides_(checked_contiguous_strides(shape_)) {
    init_backend(0);
}

Tensor::Tensor(QuantizationMode mode, std::unique_ptr<TensorBackend> backend, std::vector<std::size_t> shape)
    : quantization_mode_(mode),
      backend_(std::move(backend)),
      storage_(backend_ ? backend_->storage_ : nullptr),
      shape_(std::move(shape)),
      strides_(checked_contiguous_strides(shape_)) {}

Tensor::Tensor(const std::vector<std::vector<float>>& rows, QuantizationMode mode)
    : Tensor(flatten_rows(rows, mode), mode) {
    shape_ = {rows.size(), rows.empty() ? std::size_t{0} : rows.front().size()};
    strides_ = checked_contiguous_strides(shape_);
}

Tensor::Tensor(const std::vector<float>& data, QuantizationMode mode)
    : quantization_mode_(mode),
      shape_{data.size()},
      strides_(checked_contiguous_strides(shape_)) {
    init_backend(data.size());
    if (!data.empty()) {
        if (mode == QuantizationMode::FP32) {
            float* fp32_data = data_fp32();
            for (std::size_t i = 0; i < data.size(); ++i) {
                fp32_data[i] = data[i];
            }
        } else if (mode == QuantizationMode::FP16) {
            int16_t* fp16_data = data_fp16();
            for (std::size_t i = 0; i < data.size(); ++i) {
                // Proper FP16 encoding (stored in int16_t for NEON compatibility)
                uint16_t encoded = quantize_fp16(data[i]);
                std::memcpy(&fp16_data[i], &encoded, sizeof(uint16_t));
            }
        } else if (mode == QuantizationMode::INT8) {
            int8_t* int8_data = data_int8();
            for (std::size_t i = 0; i < data.size(); ++i) {
                int v = static_cast<int>(std::lrintf(data[i]));
                int8_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, v)));
            }
        } else if (mode == QuantizationMode::INT16) {
            int16_t* int16_data = data_int16();
            for (std::size_t i = 0; i < data.size(); ++i) {
                int v = static_cast<int>(std::lrintf(data[i]));
                int16_data[i] = static_cast<int16_t>(std::max(-32768, std::min(32767, v)));
            }
        } else if (mode == QuantizationMode::BOOL) {
            std::uint8_t* bool_data = data_bool();
            for (std::size_t i = 0; i < data.size(); ++i) {
                bool_data[i] = data[i] != 0.0f ? 1 : 0;
            }
        } else {
            throw std::invalid_argument("Unsupported tensor dtype");
        }
    }
}

Tensor::Tensor(const std::vector<int>& data, QuantizationMode mode)
    : quantization_mode_(mode),
      shape_{data.size()},
      strides_(checked_contiguous_strides(shape_)) {
    init_backend(data.size());
    if (mode == QuantizationMode::FP32) {
        float* fp32_data = data_fp32();
        for (std::size_t i = 0; i < data.size(); ++i) {
            fp32_data[i] = static_cast<float>(data[i]);
        }
    } else if (mode == QuantizationMode::FP16) {
        int16_t* fp16_data = data_fp16();
        for (std::size_t i = 0; i < data.size(); ++i) {
            // Proper FP16 encoding from int (convert to float first, then quantize)
            uint16_t encoded = quantize_fp16(static_cast<float>(data[i]));
            std::memcpy(&fp16_data[i], &encoded, sizeof(uint16_t));
        }
    } else if (mode == QuantizationMode::INT8) {
        int8_t* int8_data = data_int8();
        for (std::size_t i = 0; i < data.size(); ++i) {
            int8_data[i] = static_cast<int8_t>(
                std::max(-128, std::min(127, data[i]))
            );
        }
    } else if (mode == QuantizationMode::INT16) {
        int16_t* int16_data = data_int16();
        for (std::size_t i = 0; i < data.size(); ++i) {
            int16_data[i] = static_cast<int16_t>(
                std::max(-32768, std::min(32767, data[i]))
            );
        }
    } else if (mode == QuantizationMode::BOOL) {
        std::uint8_t* bool_data = data_bool();
        for (std::size_t i = 0; i < data.size(); ++i) {
            bool_data[i] = data[i] != 0 ? 1 : 0;
        }
    }
}

Tensor::Tensor(const std::vector<std::uint8_t>& data, QuantizationMode mode)
    : quantization_mode_(mode),
      shape_{data.size()},
      strides_(checked_contiguous_strides(shape_)) {
    if (mode != QuantizationMode::BOOL) {
        throw std::invalid_argument(
            "std::uint8_t sequence construction requires BOOL dtype");
    }
    init_backend(data.size());
    std::uint8_t* bool_data = data_bool();
    for (std::size_t i = 0; i < data.size(); ++i) {
        bool_data[i] = data[i] != 0 ? 1 : 0;
    }
}

Tensor Tensor::from_flat(const std::vector<float>& data,
                         const std::vector<std::size_t>& shape,
                         QuantizationMode mode) {
    const std::size_t element_count = checked_numel(shape);
    validate_allocation_size(element_count, mode);
    if (element_count != data.size()) {
        throw std::invalid_argument(
            "Tensor shape element count does not match data length");
    }

    const std::vector<std::size_t> strides =
        checked_contiguous_strides(shape);
    Tensor tensor(data, mode);
    tensor.shape_ = shape;
    tensor.strides_ = strides;
    return tensor;
}

Tensor Tensor::from_flat(const std::vector<std::uint8_t>& data,
                         const std::vector<std::size_t>& shape,
                         QuantizationMode mode) {
    const std::size_t element_count = checked_numel(shape);
    validate_allocation_size(element_count, mode);
    if (element_count != data.size()) {
        throw std::invalid_argument(
            "Tensor shape element count does not match data length");
    }
    if (mode != QuantizationMode::BOOL) {
        throw std::invalid_argument(
            "std::uint8_t sequence construction requires BOOL dtype");
    }

    Tensor tensor(data, mode);
    tensor.shape_ = shape;
    tensor.strides_ = checked_contiguous_strides(shape);
    return tensor;
}

Tensor Tensor::from_flat(const std::vector<int>& data,
                         const std::vector<std::size_t>& shape,
                         QuantizationMode mode) {
    const std::size_t element_count = checked_numel(shape);
    validate_allocation_size(element_count, mode);
    if (element_count != data.size()) {
        throw std::invalid_argument(
            "Tensor shape element count does not match data length");
    }

    const std::vector<std::size_t> strides =
        checked_contiguous_strides(shape);
    Tensor tensor(data, mode);
    tensor.shape_ = shape;
    tensor.strides_ = strides;
    return tensor;
}

Tensor Tensor::reshape(const std::vector<std::size_t>& shape) const {
    if (!backend_) {
        throw std::runtime_error("Tensor backend not initialized");
    }

    const std::size_t requested_numel = checked_numel(shape);
    if (requested_numel != size()) {
        throw std::invalid_argument(
            "reshape requires the new shape to have the same number of elements");
    }
    validate_allocation_size(requested_numel, quantization_mode_);
    const std::vector<std::size_t> strides =
        checked_contiguous_strides(shape);

    Tensor result(*this);
    result.shape_ = shape;
    result.strides_ = strides;
    result.storage_offset_ = 0;
    return result;
}

Tensor Tensor::reshape_view(const std::vector<std::size_t>& shape) const {
    if (!backend_ || !storage_) {
        throw std::runtime_error("Tensor backend not initialized");
    }
    if (!is_contiguous()) {
        throw std::invalid_argument(
            "reshape_view requires a contiguous source tensor");
    }

    const std::size_t requested_numel = checked_numel(shape);
    if (requested_numel != numel()) {
        throw std::invalid_argument(
            "reshape_view requires the new shape to have the same number of elements");
    }
    validate_allocation_size(requested_numel, quantization_mode_);
    const std::vector<std::size_t> strides =
        checked_contiguous_strides(shape);

    Tensor result(quantization_mode_,
                  backend_->create_storage_alias(requested_numel,
                                                 storage_offset_),
                  shape);
    result.strides_ = strides;
    result.storage_offset_ = storage_offset_;
    return result;
}

Tensor Tensor::slice_first_axis(std::size_t start, std::size_t stop) const {
    if (!backend_ || !storage_) {
        throw std::runtime_error("Tensor backend not initialized");
    }
    if (shape_.empty()) {
        throw std::invalid_argument(
            "slice_first_axis requires a tensor with at least one dimension");
    }
    if (!is_contiguous()) {
        throw std::invalid_argument(
            "slice_first_axis requires a contiguous source tensor");
    }
    if (start > stop) {
        throw std::invalid_argument(
            "slice_first_axis requires start <= stop");
    }
    if (stop > shape_[0]) {
        throw std::out_of_range(
            "slice_first_axis range exceeds the first dimension");
    }

    std::vector<std::size_t> shape = shape_;
    shape[0] = stop - start;
    const std::size_t logical_size = checked_numel(shape);
    const std::size_t offset = checked_offset_add(
        storage_offset_, checked_offset_advance(start, strides_[0]));

    Tensor result(quantization_mode_,
                  backend_->create_storage_alias(logical_size, offset),
                  std::move(shape));
    result.storage_offset_ = offset;
    return result;
}

Tensor Tensor::select_first_axis(std::size_t index) const {
    if (!backend_ || !storage_) {
        throw std::runtime_error("Tensor backend not initialized");
    }
    if (shape_.empty()) {
        throw std::invalid_argument(
            "select_first_axis requires a tensor with at least one dimension");
    }
    if (!is_contiguous()) {
        throw std::invalid_argument(
            "select_first_axis requires a contiguous source tensor");
    }
    if (index >= shape_[0]) {
        throw std::out_of_range(
            "select_first_axis index exceeds the first dimension");
    }

    std::vector<std::size_t> shape(shape_.begin() + 1, shape_.end());
    const std::size_t logical_size = checked_numel(shape);
    const std::size_t offset = checked_offset_add(
        storage_offset_, checked_offset_advance(index, strides_[0]));

    Tensor result(quantization_mode_,
                  backend_->create_storage_alias(logical_size, offset),
                  std::move(shape));
    result.storage_offset_ = offset;
    return result;
}

Tensor Tensor::flatten() const {
    return reshape({size()});
}

Tensor::Tensor(float scalar, QuantizationMode mode)
    : quantization_mode_(mode),
      shape_{1},
      strides_(checked_contiguous_strides(shape_)) {
    init_backend(1);
    if (mode == QuantizationMode::FP32) {
        data_fp32()[0] = scalar;
    } else if (mode == QuantizationMode::FP16) {
        uint16_t encoded = quantize_fp16(scalar);
        std::memcpy(&data_fp16()[0], &encoded, sizeof(uint16_t));
    } else if (mode == QuantizationMode::INT8) {
        int v = static_cast<int>(std::lrintf(scalar));
        data_int8()[0] = static_cast<int8_t>(std::max(-128, std::min(127, v)));
    } else if (mode == QuantizationMode::INT16) {
        int v = static_cast<int>(std::lrintf(scalar));
        data_int16()[0] = static_cast<int16_t>(std::max(-32768, std::min(32767, v)));
    } else if (mode == QuantizationMode::BOOL) {
        data_bool()[0] = scalar != 0.0f ? 1 : 0;
    } else {
        throw std::invalid_argument("Unsupported tensor dtype");
    }
}

Tensor::Tensor(int scalar, QuantizationMode mode)
    : quantization_mode_(mode),
      shape_{1},
      strides_(checked_contiguous_strides(shape_)) {
    init_backend(1);
    if (mode == QuantizationMode::FP32) {
        data_fp32()[0] = static_cast<float>(scalar);
    } else if (mode == QuantizationMode::FP16) {
        uint16_t encoded = quantize_fp16(static_cast<float>(scalar));
        std::memcpy(&data_fp16()[0], &encoded, sizeof(uint16_t));
    } else if (mode == QuantizationMode::INT8) {
        data_int8()[0] = static_cast<int8_t>(
            std::max(-128, std::min(127, scalar))
        );
    } else if (mode == QuantizationMode::INT16) {
        data_int16()[0] = static_cast<int16_t>(
            std::max(-32768, std::min(32767, scalar))
        );
    } else if (mode == QuantizationMode::BOOL) {
        data_bool()[0] = scalar != 0 ? 1 : 0;
    }
}

Tensor::~Tensor() = default;

Tensor::Tensor(const Tensor& other)
    : quantization_mode_(other.quantization_mode_),
      backend_(other.backend_ ? other.backend_->clone() : nullptr),
      storage_(backend_ ? backend_->storage_ : nullptr),
      shape_(other.shape_),
      strides_(other.strides_),
      storage_offset_(0) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        std::unique_ptr<TensorBackend> copied_backend =
            other.backend_ ? other.backend_->clone() : nullptr;
        std::vector<std::size_t> copied_shape = other.shape_;
        std::vector<std::size_t> copied_strides = other.strides_;
        quantization_mode_ = other.quantization_mode_;
        backend_ = std::move(copied_backend);
        storage_ = backend_ ? backend_->storage_ : nullptr;
        shape_ = std::move(copied_shape);
        strides_ = std::move(copied_strides);
        storage_offset_ = 0;
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : quantization_mode_(other.quantization_mode_),
      backend_(std::move(other.backend_)),
      storage_(std::move(other.storage_)),
      shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      storage_offset_(other.storage_offset_) {
    other.storage_offset_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        quantization_mode_ = other.quantization_mode_;
        backend_ = std::move(other.backend_);
        storage_ = std::move(other.storage_);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        storage_offset_ = other.storage_offset_;
        other.storage_offset_ = 0;
    }
    return *this;
}

std::size_t Tensor::size() const {
    return backend_ ? backend_->size() : 0;
}

bool Tensor::is_contiguous() const {
    return strides_ == checked_contiguous_strides(shape_);
}

float* Tensor::data_fp32() {
    return backend_ ? backend_->data_fp32() : nullptr;
}

const float* Tensor::data_fp32() const {
    return backend_ ? backend_->data_fp32() : nullptr;
}

int8_t* Tensor::data_int8() {
    return backend_ ? backend_->data_int8() : nullptr;
}

const int8_t* Tensor::data_int8() const {
    return backend_ ? backend_->data_int8() : nullptr;
}

int16_t* Tensor::data_fp16() {
    return backend_ ? backend_->data_fp16() : nullptr;
}

const int16_t* Tensor::data_fp16() const {
    return backend_ ? backend_->data_fp16() : nullptr;
}

int16_t* Tensor::data_int16() {
    return backend_ ? backend_->data_int16() : nullptr;
}

const int16_t* Tensor::data_int16() const {
    return backend_ ? backend_->data_int16() : nullptr;
}

std::uint8_t* Tensor::data_bool() {
    return backend_ ? backend_->data_bool() : nullptr;
}

const std::uint8_t* Tensor::data_bool() const {
    return backend_
               ? static_cast<const TensorBackend&>(*backend_).data_bool()
               : nullptr;
}

Tensor Tensor::add(const Tensor& other) const {
    return binary_op(other, OperationType::ADD);
}

void Tensor::add_into(const Tensor& other, Tensor& out) const {
    if (quantization_mode_ != QuantizationMode::FP32 ||
        other.quantization_mode_ != QuantizationMode::FP32 ||
        out.quantization_mode_ != QuantizationMode::FP32) {
        throw std::invalid_argument("add_into requires FP32 tensors");
    }
    if (!backend_ || !other.backend_ || !out.backend_) {
        throw std::runtime_error("Tensor backend not initialized");
    }
    if (shape_ != other.shape_ || shape_ != out.shape_) {
        throw std::invalid_argument("add_into requires identical tensor shapes");
    }
    const TensorBackend& input_backend = *backend_;
    const TensorBackend& other_backend = *other.backend_;
    const TensorBackend& output_backend = *out.backend_;
    if (typeid(input_backend) != typeid(other_backend) ||
        typeid(input_backend) != typeid(output_backend)) {
        throw std::invalid_argument(
            "add_into requires all tensors to use the same backend");
    }
    const auto partially_overlaps = [](const Tensor& lhs, const Tensor& rhs) {
        if (lhs.storage_.get() != rhs.storage_.get() ||
            lhs.storage_offset_ == rhs.storage_offset_) {
            return false;
        }
        const std::size_t lhs_offset = lhs.storage_offset_;
        const std::size_t rhs_offset = rhs.storage_offset_;
        return lhs_offset < rhs_offset
                   ? rhs_offset - lhs_offset < lhs.numel()
                   : lhs_offset - rhs_offset < lhs.numel();
    };
    if (partially_overlaps(*this, out) ||
        partially_overlaps(other, out)) {
        throw std::invalid_argument(
            "add_into does not support partially overlapping storage ranges");
    }
    if (!backend_->operation_into(*other.backend_, *out.backend_, OperationType::ADD)) {
        throw std::runtime_error(
            "FP32 ADD into an existing output is not supported by this backend");
    }
}

Tensor Tensor::mul(const Tensor& other) const {
    return binary_op(other, OperationType::MUL);
}

Tensor Tensor::sub(const Tensor& other) const {
    return binary_op(other, OperationType::SUB);
}

Tensor Tensor::div(const Tensor& other) const {
    return binary_op(other, OperationType::DIV);
}

Tensor Tensor::binary_op(const Tensor& other, OperationType op) const {
    if (backend_ && other.backend_) {
        const auto* lhs_generic =
            dynamic_cast<const GenericBackend*>(backend_.get());
        const auto* rhs_generic =
            dynamic_cast<const GenericBackend*>(other.backend_.get());
        const bool generic_broadcast =
            shape_ != other.shape_ &&
            quantization_mode_ == other.quantization_mode_ &&
            (quantization_mode_ == QuantizationMode::FP32 ||
             quantization_mode_ == QuantizationMode::FP16 ||
             quantization_mode_ == QuantizationMode::INT8 ||
             quantization_mode_ == QuantizationMode::INT16) &&
            lhs_generic && rhs_generic;
        if (shape_ != other.shape_ && !generic_broadcast) {
            check_same_shape(other);
        }
        if (generic_broadcast) {
            detail::BroadcastPlan plan;
            try {
                plan = detail::make_broadcast_plan(
                    shape_, strides_, other.shape_, other.strides_);
            } catch (const std::invalid_argument& error) {
                throw std::runtime_error(
                    std::string("Tensor shape mismatch: ") + error.what());
            }
            if (quantization_mode_ != other.quantization_mode_) {
                throw std::runtime_error("Quantization mode mismatch");
            }
            const std::size_t output_numel =
                detail::checked_broadcast_numel(plan.shape);
            auto output_backend =
                std::make_unique<GenericBackend>(output_numel,
                                                 quantization_mode_);
            lhs_generic->operation_broadcast(
                *rhs_generic, *output_backend, plan, op, numel(), other.numel());
            return Tensor(quantization_mode_, std::move(output_backend),
                          std::move(plan.shape));
        }
        if (auto out = backend_->operation_new(*other.backend_, op)) {
            return Tensor(quantization_mode_, std::move(out), shape_);
        }
    }

    Tensor result(*this);
    switch (op) {
        case OperationType::ADD: result.add_inplace(other); break;
        case OperationType::SUB: result.sub_inplace(other); break;
        case OperationType::MUL: result.mul_inplace(other); break;
        default: result.div_inplace(other); break;
    }
    return result;
}

void Tensor::add_inplace(const Tensor& other) {
    binary_inplace(other, OperationType::ADD);
}

void Tensor::mul_inplace(const Tensor& other) {
    binary_inplace(other, OperationType::MUL);
}   

void Tensor::sub_inplace(const Tensor& other) {
    binary_inplace(other, OperationType::SUB);
}

void Tensor::div_inplace(const Tensor& other) {
    binary_inplace(other, OperationType::DIV);
}

void Tensor::binary_inplace(const Tensor& other, OperationType op) {
    if (!backend_ || !other.backend_) {
        throw std::runtime_error("Tensor backend not initialized");
    }
    auto* lhs_backend = dynamic_cast<GenericBackend*>(backend_.get());
    const auto* rhs_backend =
        dynamic_cast<const GenericBackend*>(other.backend_.get());
    const bool generic_broadcast =
        shape_ != other.shape_ &&
        quantization_mode_ == other.quantization_mode_ &&
        (quantization_mode_ == QuantizationMode::FP32 ||
         quantization_mode_ == QuantizationMode::FP16 ||
         quantization_mode_ == QuantizationMode::INT8 ||
         quantization_mode_ == QuantizationMode::INT16) &&
        lhs_backend && rhs_backend;
    if (shape_ == other.shape_ || !generic_broadcast) {
        check_same_shape(other);
        backend_->operation(*other.backend_, op);
        return;
    }

    detail::BroadcastPlan plan;
    try {
        plan = detail::make_broadcast_plan(
            shape_, strides_, other.shape_, other.strides_);
    } catch (const std::invalid_argument& error) {
        throw std::runtime_error(
            std::string("Tensor shape mismatch: ") + error.what());
    }
    if (plan.shape != shape_) {
        throw std::runtime_error(
            "Tensor shape mismatch: in-place broadcasting cannot expand the destination");
    }
    if (quantization_mode_ != other.quantization_mode_) {
        throw std::runtime_error("Quantization mode mismatch");
    }
    const bool shares_storage = storage_.get() == other.storage_.get();
    if (shares_storage && size() != 0 && other.size() != 0) {
        const std::size_t lhs_offset = storage_offset_;
        const std::size_t rhs_offset = other.storage_offset_;
        const bool overlaps =
            lhs_offset <= rhs_offset
                ? rhs_offset - lhs_offset < size()
                : lhs_offset - rhs_offset < other.size();
        const bool exact_elementwise_alias =
            lhs_offset == rhs_offset &&
            plan.lhs_strides == plan.rhs_strides;
        if (overlaps && !exact_elementwise_alias) {
            throw std::invalid_argument(
                "In-place operation does not support partially overlapping tensor views");
        }
    }

    lhs_backend->operation_broadcast(
        *rhs_backend, *lhs_backend, plan, op, numel(), other.numel());
}

void Tensor::check_same_shape(const Tensor& other) const {
    // Different element counts are reported by the backend as a size mismatch.
    if (size() == other.size() && shape_ != other.shape_) {
        throw std::runtime_error("Tensor shape mismatch");
    }
}

void Tensor::init_backend(std::size_t size) {
    validate_tensor_dtype_supported(quantization_mode_);
    validate_allocation_size(size, quantization_mode_);
    backend_ = create_best_backend(size, quantization_mode_);
    storage_ = backend_ ? backend_->storage_ : nullptr;
}

std::size_t Tensor::checked_numel(const std::vector<std::size_t>& shape) {
    return checked_shape_numel(shape);
}

std::vector<std::size_t> Tensor::checked_contiguous_strides(
    const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t axis = shape.size(); axis > 0; --axis) {
        const std::size_t index = axis - 1;
        strides[index] = stride;
        if (shape[index] != 0 &&
            stride > std::numeric_limits<std::size_t>::max() / shape[index]) {
            throw std::overflow_error(
                "Tensor row-major stride overflows size_t");
        }
        stride *= shape[index];
    }
    return strides;
}

void Tensor::validate_allocation_size(std::size_t size,
                                      QuantizationMode mode) {
    check_allocation_size(size, mode);
}

} // namespace labneura
