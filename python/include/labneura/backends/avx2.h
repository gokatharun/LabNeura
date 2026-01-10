#pragma once
#include "labneura/backends/base.h"

namespace labneura {

class AVX2Backend : public TensorBackend {
public:
    explicit AVX2Backend(std::size_t size, QuantizationMode mode);

    std::size_t size() const override;
    QuantizationMode quantization_mode() const override;

    // Data accessors for all types
    double* data_fp64() override;
    const double* data_fp64() const override;
    float* data_fp32() override;
    const float* data_fp32() const override;
    int16_t* data_fp16() override;
    const int16_t* data_fp16() const override;
    int32_t* data_int32() override;
    const int32_t* data_int32() const override;
    int16_t* data_int16() override;
    const int16_t* data_int16() const override;
    int8_t* data_int8() override;
    const int8_t* data_int8() const override;

    std::unique_ptr<TensorBackend> clone() const override;
    
    // Operation kernels for all types
    void operation_fp64(const TensorBackend& other, OperationType op_type) override;
    void operation_fp32(const TensorBackend& other, OperationType op_type) override;
    void operation_fp16(const TensorBackend& other, OperationType op_type) override;
    void operation_int32(const TensorBackend& other, OperationType op_type) override;
    void operation_int16(const TensorBackend& other, OperationType op_type) override;
    void operation_int8(const TensorBackend& other, OperationType op_type) override;

    // FP16 helpers
    void store_fp16_value(std::size_t idx, float value);

private:
    std::size_t size_;
    std::size_t aligned_size_;
    QuantizationMode quantization_mode_;

    // Data storage for all types
    std::unique_ptr<double, void(*)(double*)> data_fp64_{nullptr, nullptr};
    std::unique_ptr<float, void(*)(float*)> data_fp32_{nullptr, nullptr};
    std::unique_ptr<int16_t, void(*)(int16_t*)> data_fp16_{nullptr, nullptr};
    std::unique_ptr<float, void(*)(float*)> data_fp16_shadow_{nullptr, nullptr};
    std::unique_ptr<int32_t, void(*)(int32_t*)> data_int32_{nullptr, nullptr};
    std::unique_ptr<int16_t, void(*)(int16_t*)> data_int16_{nullptr, nullptr};
    std::unique_ptr<int8_t, void(*)(int8_t*)> data_int8_{nullptr, nullptr};
};

} // namespace labneura