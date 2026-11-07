#pragma once
#include "labneura/backends/common/base.h"

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
            void operation_fp32(const TensorBackend& other, OperationType op_type) override;
            void operation_int8(const TensorBackend& other, OperationType op_type) override;
        protected:
            std::unique_ptr<TensorBackend> create_storage_alias(
                std::size_t logical_size,
                std::size_t storage_offset) const override;
            bool operation_into(const TensorBackend& other, TensorBackend& out,
                                OperationType op_type) const override;
        private:
            GenericBackend(std::shared_ptr<TensorStorage> storage,
                           std::size_t logical_size,
                           std::size_t storage_offset);

            std::size_t size_;
            QuantizationMode quantization_mode_;
            // Non-owning typed aliases; TensorStorage owns the allocation.
            float* data_fp32_{nullptr};
            int8_t* data_int8_{nullptr};
    };
}  // namespace labneura