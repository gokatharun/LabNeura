#pragma once
#include "labneura/backends/common/base.h"
#include "labneura/detail/broadcast.h"

namespace labneura {
    class Tensor;

    class GenericBackend : public TensorBackend {
        public:
            explicit GenericBackend(std::size_t size, QuantizationMode mode);
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
            std::uint8_t* data_bool() override;
            const std::uint8_t* data_bool() const override;
            std::unique_ptr<TensorBackend> clone() const override;
            void operation_fp32(const TensorBackend& other, OperationType op_type) override;
            void operation_fp16(const TensorBackend& other, OperationType op_type) override;
            void operation_int8(const TensorBackend& other, OperationType op_type) override;
            void operation_int16(const TensorBackend& other, OperationType op_type) override;
        protected:
            std::unique_ptr<TensorBackend> create_storage_alias(
                std::size_t logical_size,
                std::size_t storage_offset) const override;
            bool operation_into(const TensorBackend& other, TensorBackend& out,
                                OperationType op_type) const override;
        private:
            void operation_broadcast(
                const GenericBackend& other, GenericBackend& output,
                const detail::BroadcastPlan& plan, OperationType op_type,
                std::size_t lhs_numel, std::size_t rhs_numel) const;
            void compare_broadcast(
                const GenericBackend& other, GenericBackend& output,
                const detail::BroadcastPlan& plan, ComparisonType comparison,
                std::size_t lhs_numel, std::size_t rhs_numel) const;
            void logical_not_into(GenericBackend& output) const;
            void logical_and_broadcast(
                const GenericBackend& other, GenericBackend& output,
                const detail::BroadcastPlan& plan, std::size_t lhs_numel,
                std::size_t rhs_numel) const;
            void logical_or_broadcast(
                const GenericBackend& other, GenericBackend& output,
                const detail::BroadcastPlan& plan, std::size_t lhs_numel,
                std::size_t rhs_numel) const;

            GenericBackend(std::shared_ptr<TensorStorage> storage,
                           std::size_t logical_size,
                           std::size_t storage_offset);

            std::size_t size_;
            QuantizationMode quantization_mode_;
            // Non-owning typed aliases; TensorStorage owns the allocation.
            float* data_fp32_{nullptr};
            int8_t* data_int8_{nullptr};
            int16_t* data_fp16_{nullptr};
            int16_t* data_int16_{nullptr};
            std::uint8_t* data_bool_{nullptr};

            friend class Tensor;
    };
}  // namespace labneura