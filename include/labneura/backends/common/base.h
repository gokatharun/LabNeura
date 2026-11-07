#pragma once
#include <memory>
#include "labneura/tensor.h"
#include "labneura/visibility.h"

namespace labneura {

class LABNEURA_API TensorBackend {
public:
    virtual ~TensorBackend() = default;

    // ---- Metadata ----
    virtual std::size_t size() const = 0;
    virtual QuantizationMode quantization_mode() const = 0;

    // ---- Raw data access ----
    virtual float* data_fp32() = 0;
    virtual const float* data_fp32() const = 0;
    virtual int8_t* data_int8() = 0;
    virtual const int8_t* data_int8() const = 0;
    // Optional FP16/INT16 accessors (default throw)
    virtual int16_t* data_fp16();
    virtual const int16_t* data_fp16() const;
    virtual int16_t* data_int16();
    virtual const int16_t* data_int16() const;

    // ---- Cloning ----
    virtual std::unique_ptr<TensorBackend> clone() const = 0;

    // ---- Public operations (NON-virtual interface) ----
    void add_inplace(const TensorBackend& other);
    void sub_inplace(const TensorBackend& other);
    void mul_inplace(const TensorBackend& other);
    void div_inplace(const TensorBackend& other);
    void operation(const TensorBackend& other, OperationType op);

    // Allocating operation: out = this (op) other, without modifying either input.
    // Returns nullptr when this backend has no direct-output kernel for the current
    // dtype; callers then fall back to clone() + in-place. Validation errors are
    // reported with the same exceptions as operation().
    std::unique_ptr<TensorBackend> operation_new(const TensorBackend& other, OperationType op) const;

protected:
    // ---- Optional direct-output hooks (defaults: unsupported) ----
    // Returns an uninitialised output backend of the same type, size and mode, or
    // nullptr if operation_into() is not supported for this backend/mode.
    virtual std::unique_ptr<TensorBackend> create_output() const;
    // Computes out = this (op) other for equal-size, equal-mode operands. Exact
    // aliasing (out's buffer == an input's buffer) is supported by elementwise
    // kernels; partial overlap is not allowed.
    // Returns false if unsupported.
    virtual bool operation_into(const TensorBackend& other, TensorBackend& out, OperationType op) const;

    // ---- Backend-specific kernels (virtual hooks) ----
    virtual void operation_fp32(const TensorBackend& other, OperationType op) = 0;
    virtual void operation_int8(const TensorBackend& other, OperationType op) = 0;
    // Optional FP16/INT16 kernels (default throw)
    virtual void operation_fp16(const TensorBackend& other, OperationType op);
    virtual void operation_int16(const TensorBackend& other, OperationType op);

    friend class Tensor;
};

} // namespace labneura