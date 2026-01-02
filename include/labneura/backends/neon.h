#pragma once

#include "labneura/backends/base.h"

namespace labneura {
    class NEONBackend : public TensorBackend {

        public:
            explicit NEONBackend(std::size_t size, QuantizationMode mode);
            std::size_t size() const override;
            QuantizationMode quantization_mode() const override;
            float* data_fp32() override;
            const float* data_fp32() const override;
            int8_t* data_int8() override;
            const int8_t* data_int8() const override;
            std::unique_ptr<TensorBackend> clone() const override;
            void add_inplace(const TensorBackend& other) override;
            void mul_inplace(const TensorBackend& other) override;
            void sub_inplace(const TensorBackend& other) override;

        private:
            std::size_t size_;
            std::size_t aligned_size_;
            QuantizationMode quantization_mode_;
            std::unique_ptr<float[]> data_fp32_;
            std::unique_ptr<int8_t[]> data_int8_;

            // FP32 kernels
            void add_inplace_fp32(const TensorBackend& other);
            void mul_inplace_fp32(const TensorBackend& other);
            void sub_inplace_fp32(const TensorBackend& other);

            // INT8 kernels
            void add_inplace_int8(const TensorBackend& other);
            void mul_inplace_int8(const TensorBackend& other);
            void sub_inplace_int8(const TensorBackend& other);
    };

} // namespace labneura
