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
    MUL
};

// Forward declaration for backend implementation
class TensorBackend;

// Main Tensor class: handles mixed int/float input with architecture-aware storage
class Tensor {
public:
    // Constructor with array/scalar and optional quantization
    // Supports int, float, or std::vector<int/float>
    Tensor();
    explicit Tensor(const std::vector<float>& data, QuantizationMode mode = QuantizationMode::FP32);
    explicit Tensor(const std::vector<int>& data, QuantizationMode mode = QuantizationMode::FP32);
    explicit Tensor(float scalar, QuantizationMode mode = QuantizationMode::FP32);
    explicit Tensor(int scalar, QuantizationMode mode = QuantizationMode::FP32);

    // Destructor
    ~Tensor();

    // Copy and move semantics
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Shape and size queries
    std::size_t size() const;
    std::size_t numel() const { return size(); }
    QuantizationMode quantization_mode() const { return quantization_mode_; }

    // Get raw data pointer (for internal use and operations)
    float* data_fp32();
    const float* data_fp32() const;
    int8_t* data_int8();
    const int8_t* data_int8() const;

    // Operations
    // Addition: result = this + other
    Tensor add(const Tensor& other) const;
    
    // In-place addition: this += other
    void add_inplace(const Tensor& other);
    // In-place multiplication: this = this . other
    void mul_inplace(const Tensor& other);
    // In-place subtraction: this -= other
    void sub_inplace(const Tensor& other);

private:
    QuantizationMode quantization_mode_;
    std::unique_ptr<TensorBackend> backend_;

    // Helper to initialize backend based on hardware
    void init_backend(std::size_t size);
};

} // namespace labneura
