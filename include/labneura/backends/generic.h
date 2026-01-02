#pragma once
#include "labneura/backends/base.h"

namespace labneura {
    class GenericBackend : public TensorBackend {
        public:
            explicit GenericBackend(std::size_t size, QuantizationMode mode);
            std::size_t size() const override;
            QuantizationMode quantization_mode() const override;
            float* data_fp32() override;
            const float* data_fp32() const override;
            int8_t* data_int8() override;
            const int8_t* data_int8() const override;
            std::unique_ptr<TensorBackend> clone() const override;
            void add_inplace(const TensorBackend& other) override;
            void sub_inplace(const TensorBackend& other) override;
            void mul_inplace(const TensorBackend& other) override;
        private:
            std::size_t size_;
            QuantizationMode quantization_mode_;
            std::unique_ptr<float[]> data_fp32_;
            std::unique_ptr<int8_t[]> data_int8_;
    };
}  // namespace labneura