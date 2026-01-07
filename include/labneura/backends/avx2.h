#pragma once
#include "labneura/backends/base.h"

namespace labneura {

class AVX2Backend : public TensorBackend {
public:
    explicit AVX2Backend(std::size_t size, QuantizationMode mode);

    std::size_t size() const override;
    QuantizationMode quantization_mode() const override;

    float* data_fp32() override;
    const float* data_fp32() const override;

    int8_t* data_int8() override;
    const int8_t* data_int8() const override;

    std::unique_ptr<TensorBackend> clone() const override;
    void operation_fp32(const TensorBackend& other, OperationType op_type) override;
    void operation_int8(const TensorBackend& other, OperationType op_type) override;

private:
    std::size_t size_;
    std::size_t aligned_size_;
    QuantizationMode quantization_mode_;

    std::unique_ptr<float, void(*)(float*)> data_fp32_{nullptr, nullptr};
    std::unique_ptr<int8_t, void(*)(int8_t*)> data_int8_{nullptr, nullptr};
};

} // namespace labneura