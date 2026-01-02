#pragma once
#include "labneura/tensor.h"

namespace labneura {
    // Abstract TensorBackend interface
    class TensorBackend {
    public:
        virtual ~TensorBackend() = default;
        
        virtual std::size_t size() const = 0;
        virtual QuantizationMode quantization_mode() const = 0;
        
        virtual float* data_fp32() = 0;
        virtual const float* data_fp32() const = 0;
        virtual int8_t* data_int8() = 0;
        virtual const int8_t* data_int8() const = 0;
        
        virtual std::unique_ptr<TensorBackend> clone() const = 0;
        virtual void add_inplace(const TensorBackend& other) = 0;
        virtual void sub_inplace(const TensorBackend& other) = 0;
        virtual void mul_inplace(const TensorBackend& other) = 0;
        virtual void operation(const TensorBackend& other, OperationType op_type) = 0;
        virtual void operation_fp32(const TensorBackend& other, OperationType op_type) = 0;
        virtual void operation_int8(const TensorBackend& other, OperationType op_type) = 0;
    };
}