#pragma once
#include <memory>
#include "labneura/tensor.h"

namespace labneura {

class TensorBackend {
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

    // ---- Cloning ----
    virtual std::unique_ptr<TensorBackend> clone() const = 0;

    // ---- Public operations (NON-virtual interface) ----
    void add_inplace(const TensorBackend& other);
    void sub_inplace(const TensorBackend& other);
    void mul_inplace(const TensorBackend& other);
    void operation(const TensorBackend& other, OperationType op);

protected:
    // ---- Backend-specific kernels (virtual hooks) ----
    virtual void operation_fp32(const TensorBackend& other, OperationType op) = 0;
    virtual void operation_int8(const TensorBackend& other, OperationType op) = 0;
};

} // namespace labneura