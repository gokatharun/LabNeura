#pragma once

#include "labneura/backends/common/base.h"

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
            int16_t* data_fp16() override;
            const int16_t* data_fp16() const override;
            int16_t* data_int16() override;
            const int16_t* data_int16() const override;
            std::unique_ptr<TensorBackend> clone() const override;
            void operation_int8(const TensorBackend& other, OperationType op_type) override;
            void operation_fp32(const TensorBackend& other, OperationType op_type) override;
            void operation_fp16(const TensorBackend& other, OperationType op_type) override;
            void operation_int16(const TensorBackend& other, OperationType op_type) override;

        protected:
            std::unique_ptr<TensorBackend> create_storage_alias(
                std::size_t logical_size,
                std::size_t storage_offset) const override;
            std::unique_ptr<TensorBackend> create_output() const override;
            bool operation_into(const TensorBackend& other, TensorBackend& out, OperationType op_type) const override;

        private:
            NEONBackend(std::shared_ptr<TensorStorage> storage,
                        std::size_t logical_size,
                        std::size_t storage_offset);

            std::size_t size_;
            std::size_t aligned_size_;
            QuantizationMode quantization_mode_;
            // Non-owning typed aliases; TensorStorage owns the allocation.
            float* data_fp32_{nullptr};
            int8_t* data_int8_{nullptr};
            int16_t* data_fp16_{nullptr};
            int16_t* data_int16_{nullptr};

    };

} // namespace labneura
