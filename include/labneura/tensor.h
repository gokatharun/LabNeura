#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <variant>
#include <stdexcept>

#pragma once

#if defined(__x86_64__) || defined(_M_X64)
    #define LABNEURA_X86 1
#else
    #define LABNEURA_X86 0
#endif

#if LABNEURA_X86 && defined(__AVX2__)
    #define LABNEURA_HAVE_AVX2 1
#else
    #define LABNEURA_HAVE_AVX2 0
#endif
#if LABNEURA_X86 && defined(__AVX512F__)
    #define LABNEURA_HAVE_AVX512 1
#else
    #define LABNEURA_HAVE_AVX512 0
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define LABNEURA_HAVE_NEON 1
#else
    #define LABNEURA_HAVE_NEON 0
#endif

namespace labneura {

// Quantization modes
enum class QuantizationMode {
    FP64,      // 64-bit floating point
    FP32,      // 32-bit floating point (default)
    FP16,      // 16-bit floating point
    INT32,     // 32-bit signed integer quantization
    INT16,     // 16-bit signed integer quantization
    INT8,      // 8-bit signed integer quantization
};

enum OperationType {
    ADD,
    SUB,
    MUL,
    DIV
};

// Forward declaration for backend implementation
class TensorBackend;
class TensorStorage;

// Main Tensor class: handles mixed int/float input with architecture-aware storage
class Tensor {
public:
    // Constructor with array/scalar and optional quantization
    // Supports int, float, or std::vector<int/float>
    Tensor();
    explicit Tensor(const std::vector<float>& data, QuantizationMode mode = QuantizationMode::FP32);
    explicit Tensor(const std::vector<int>& data, QuantizationMode mode = QuantizationMode::FP32);
    // 2-D tensor from equal-length rows (stored flat, row-major)
    explicit Tensor(const std::vector<std::vector<float>>& rows, QuantizationMode mode = QuantizationMode::FP32);
    explicit Tensor(float scalar, QuantizationMode mode = QuantizationMode::FP32);
    explicit Tensor(int scalar, QuantizationMode mode = QuantizationMode::FP32);

    // Construct a contiguous tensor from flat row-major data and an explicit shape.
    // An empty shape represents a scalar and therefore requires exactly one value.
    static Tensor from_flat(const std::vector<float>& data,
                            const std::vector<std::size_t>& shape,
                            QuantizationMode mode = QuantizationMode::FP32);
    static Tensor from_flat(const std::vector<int>& data,
                            const std::vector<std::size_t>& shape,
                            QuantizationMode mode = QuantizationMode::FP32);

    // Return an independently owned contiguous copy with the requested shape.
    Tensor reshape(const std::vector<std::size_t>& shape) const;
    // Return a contiguous reshape that explicitly shares storage with this tensor.
    Tensor reshape_view(const std::vector<std::size_t>& shape) const;
    // Return a first-axis range view sharing storage with this tensor.
    Tensor slice_first_axis(std::size_t start, std::size_t stop) const;
    // Return a view selecting one element along the first axis.
    Tensor select_first_axis(std::size_t index) const;
    // Return an independently owned one-dimensional copy in row-major order.
    Tensor flatten() const;

    // Destructor
    ~Tensor();

    // Copy and move semantics
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Shape and size queries
    std::size_t size() const;
    std::size_t numel() const { return backend_ ? checked_numel(shape_) : 0; }
    QuantizationMode quantization_mode() const { return quantization_mode_; }
    const std::vector<std::size_t>& shape() const { return shape_; }
    // Strides and storage offset are measured in elements.
    const std::vector<std::size_t>& strides() const { return strides_; }
    std::size_t storage_offset() const { return storage_offset_; }
    bool is_contiguous() const;
    std::size_t ndim() const { return shape_.size(); }
    std::size_t rows() const { return shape_.size() == 2 ? shape_[0] : 1; }
    std::size_t cols() const { return shape_.size() == 2 ? shape_[1] : size(); }

    // Get raw data pointer (for internal use and operations)
    float* data_fp32();
    const float* data_fp32() const;
    int8_t* data_int8();
    const int8_t* data_int8() const;
    int16_t* data_fp16();
    const int16_t* data_fp16() const;
    int16_t* data_int16();
    const int16_t* data_int16() const;

    // Operations
    // Addition: result = this + other
    Tensor add(const Tensor& other) const;
    // Write this + other into an existing same-shape, same-backend FP32 tensor.
    // Exact aliasing of out with either input is supported and overwrites it.
    void add_into(const Tensor& other, Tensor& out) const;
    // Multiplication: result = this * other
    Tensor mul(const Tensor& other) const;
    // Subtraction: result = this - other
    Tensor sub(const Tensor& other) const;
    // Division: result = this / other
    Tensor div(const Tensor& other) const;
    
    // In-place addition: this += other
    void add_inplace(const Tensor& other);
    // In-place multiplication: this *= other
    void mul_inplace(const Tensor& other);
    // In-place subtraction: this -= other
    void sub_inplace(const Tensor& other);
    // In-place division: this /= other
    void div_inplace(const Tensor& other);

private:
    QuantizationMode quantization_mode_;
    std::unique_ptr<TensorBackend> backend_;
    std::shared_ptr<TensorStorage> storage_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::size_t storage_offset_{0};

    void check_same_shape(const Tensor& other) const;

    // Allocating binary operation. Computes directly into a new output when the
    // backend supports it, otherwise clones this tensor and runs the in-place kernel.
    Tensor(QuantizationMode mode, std::unique_ptr<TensorBackend> backend, std::vector<std::size_t> shape);
    Tensor binary_op(const Tensor& other, OperationType op) const;
    void binary_inplace(const Tensor& other, OperationType op);

    // Helper to initialize backend based on hardware
    void init_backend(std::size_t size);
    static std::size_t checked_numel(const std::vector<std::size_t>& shape);
    static std::vector<std::size_t> checked_contiguous_strides(
        const std::vector<std::size_t>& shape);
    static void validate_allocation_size(std::size_t size,
                                         QuantizationMode mode);
};

} // namespace labneura
