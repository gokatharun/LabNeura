#include "labneura/tensor.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <omp.h>

// Hardware detection
#if defined(__aarch64__) && defined(__APPLE__)
#define LABNEURA_HAVE_M1 1
#include <arm_neon.h>
#else
#define LABNEURA_HAVE_M1 0
#endif

namespace labneura {

// Abstract backend interface
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
};

// Generic fallback backend (CPU registers + memory)
class GenericBackend : public TensorBackend {
public:
    explicit GenericBackend(std::size_t size, QuantizationMode mode)
        : size_(size), quantization_mode_(mode) {
        if (mode == QuantizationMode::FP32) {
            data_fp32_ = std::make_unique<float[]>(size);
        } else if (mode == QuantizationMode::INT8) {
            data_int8_ = std::make_unique<int8_t[]>(size);
        }
    }

    std::size_t size() const override { return size_; }
    
    QuantizationMode quantization_mode() const override { 
        return quantization_mode_; 
    }

    float* data_fp32() override {
        if (quantization_mode_ != QuantizationMode::FP32) {
            throw std::runtime_error("Tensor is not in FP32 mode");
        }
        return data_fp32_.get();
    }

    const float* data_fp32() const override {
        if (quantization_mode_ != QuantizationMode::FP32) {
            throw std::runtime_error("Tensor is not in FP32 mode");
        }
        return data_fp32_.get();
    }

    int8_t* data_int8() override {
        if (quantization_mode_ != QuantizationMode::INT8) {
            throw std::runtime_error("Tensor is not in INT8 mode");
        }
        return data_int8_.get();
    }

    const int8_t* data_int8() const override {
        if (quantization_mode_ != QuantizationMode::INT8) {
            throw std::runtime_error("Tensor is not in INT8 mode");
        }
        return data_int8_.get();
    }

    std::unique_ptr<TensorBackend> clone() const override {
        auto backend = std::make_unique<GenericBackend>(size_, quantization_mode_);
        if (quantization_mode_ == QuantizationMode::FP32) {
            std::copy(data_fp32_.get(), data_fp32_.get() + size_, 
                     backend->data_fp32());
        } else if (quantization_mode_ == QuantizationMode::INT8) {
            std::copy(data_int8_.get(), data_int8_.get() + size_, 
                     backend->data_int8());
        }
        return backend;
    }

    void add_inplace(const TensorBackend& other) override {
        if (size_ != other.size()) {
            throw std::runtime_error("Tensor size mismatch in addition");
        }

        if (quantization_mode_ == QuantizationMode::FP32) {
            const float* other_data = other.data_fp32();
            float* this_data = data_fp32();
            #pragma omp parallel for
            for (std::size_t i = 0; i < size_; ++i) {
                this_data[i] += other_data[i];
            }
        } else if (quantization_mode_ == QuantizationMode::INT8) {
            const int8_t* other_data = other.data_int8();
            int8_t* this_data = data_int8();
            #pragma omp parallel for
            for (std::size_t i = 0; i < size_; ++i) {
                int16_t sum = static_cast<int16_t>(this_data[i]) + 
                             static_cast<int16_t>(other_data[i]);
                this_data[i] = static_cast<int8_t>(
                    std::max(-128, std::min(127, static_cast<int>(sum)))
                );
            }
        }
    }

private:
    std::size_t size_;
    QuantizationMode quantization_mode_;
    std::unique_ptr<float[]> data_fp32_;
    std::unique_ptr<int8_t[]> data_int8_;
};

// M1/NEON optimized backend (Apple Silicon)
#if LABNEURA_HAVE_M1
class NEONBackend : public TensorBackend {
public:
    explicit NEONBackend(std::size_t size, QuantizationMode mode)
        : size_(size), quantization_mode_(mode) {
        // Align to 16 bytes for NEON operations
        aligned_size_ = ((size + 3) / 4) * 4;
        
        if (mode == QuantizationMode::FP32) {
            data_fp32_ = std::make_unique<float[]>(aligned_size_);
            // Initialize padding with zeros
            std::fill(data_fp32_.get() + size, data_fp32_.get() + aligned_size_, 0.0f);
        } else if (mode == QuantizationMode::INT8) {
            data_int8_ = std::make_unique<int8_t[]>(aligned_size_);
            std::fill(data_int8_.get() + size, data_int8_.get() + aligned_size_, 0);
        }
    }

    std::size_t size() const override { return size_; }
    
    QuantizationMode quantization_mode() const override { 
        return quantization_mode_; 
    }

    float* data_fp32() override {
        if (quantization_mode_ != QuantizationMode::FP32) {
            throw std::runtime_error("Tensor is not in FP32 mode");
        }
        return data_fp32_.get();
    }

    const float* data_fp32() const override {
        if (quantization_mode_ != QuantizationMode::FP32) {
            throw std::runtime_error("Tensor is not in FP32 mode");
        }
        return data_fp32_.get();
    }

    int8_t* data_int8() override {
        if (quantization_mode_ != QuantizationMode::INT8) {
            throw std::runtime_error("Tensor is not in INT8 mode");
        }
        return data_int8_.get();
    }

    const int8_t* data_int8() const override {
        if (quantization_mode_ != QuantizationMode::INT8) {
            throw std::runtime_error("Tensor is not in INT8 mode");
        }
        return data_int8_.get();
    }

    std::unique_ptr<TensorBackend> clone() const override {
        auto backend = std::make_unique<NEONBackend>(size_, quantization_mode_);
        if (quantization_mode_ == QuantizationMode::FP32) {
            std::copy(data_fp32_.get(), data_fp32_.get() + size_, 
                     backend->data_fp32());
        } else if (quantization_mode_ == QuantizationMode::INT8) {
            std::copy(data_int8_.get(), data_int8_.get() + size_, 
                     backend->data_int8());
        }
        return backend;
    }

    void add_inplace(const TensorBackend& other) override {
        if (size_ != other.size()) {
            throw std::runtime_error("Tensor size mismatch in addition");
        }

        if (quantization_mode_ == QuantizationMode::FP32) {
            add_inplace_fp32(other);
        } else if (quantization_mode_ == QuantizationMode::INT8) {
            add_inplace_int8(other);
        }
    }

private:
    std::size_t size_;
    std::size_t aligned_size_;
    QuantizationMode quantization_mode_;
    std::unique_ptr<float[]> data_fp32_;
    std::unique_ptr<int8_t[]> data_int8_;

    void add_inplace_fp32(const TensorBackend& other) {
        const float* other_data = other.data_fp32();
        float* this_data = data_fp32();

        // NEON loop: process 32 floats per iteration (8 NEON operations)
        std::size_t len32 = (size_ / 32) * 32;
        for (std::size_t i = 0; i < len32; i += 32) {
            // Load all 8 groups
            float32x4_t a0 = vld1q_f32(&this_data[i]);
            float32x4_t b0 = vld1q_f32(&other_data[i]);
            float32x4_t a1 = vld1q_f32(&this_data[i + 4]);
            float32x4_t b1 = vld1q_f32(&other_data[i + 4]);
            float32x4_t a2 = vld1q_f32(&this_data[i + 8]);
            float32x4_t b2 = vld1q_f32(&other_data[i + 8]);
            float32x4_t a3 = vld1q_f32(&this_data[i + 12]);
            float32x4_t b3 = vld1q_f32(&other_data[i + 12]);
            float32x4_t a4 = vld1q_f32(&this_data[i + 16]);
            float32x4_t b4 = vld1q_f32(&other_data[i + 16]);
            float32x4_t a5 = vld1q_f32(&this_data[i + 20]);
            float32x4_t b5 = vld1q_f32(&other_data[i + 20]);
            float32x4_t a6 = vld1q_f32(&this_data[i + 24]);
            float32x4_t b6 = vld1q_f32(&other_data[i + 24]);
            float32x4_t a7 = vld1q_f32(&this_data[i + 28]);
            float32x4_t b7 = vld1q_f32(&other_data[i + 28]);
            
            // Compute all
            float32x4_t r0 = vaddq_f32(a0, b0);
            float32x4_t r1 = vaddq_f32(a1, b1);
            float32x4_t r2 = vaddq_f32(a2, b2);
            float32x4_t r3 = vaddq_f32(a3, b3);
            float32x4_t r4 = vaddq_f32(a4, b4);
            float32x4_t r5 = vaddq_f32(a5, b5);
            float32x4_t r6 = vaddq_f32(a6, b6);
            float32x4_t r7 = vaddq_f32(a7, b7);
            
            // Store all
            vst1q_f32(&this_data[i], r0);
            vst1q_f32(&this_data[i + 4], r1);
            vst1q_f32(&this_data[i + 8], r2);
            vst1q_f32(&this_data[i + 12], r3);
            vst1q_f32(&this_data[i + 16], r4);
            vst1q_f32(&this_data[i + 20], r5);
            vst1q_f32(&this_data[i + 24], r6);
            vst1q_f32(&this_data[i + 28], r7);
        }

        // Handle remaining multiples of 16
        std::size_t len16 = ((size_ - len32) / 16) * 16 + len32;
        for (std::size_t i = len32; i < len16; i += 16) {
            float32x4_t va0 = vld1q_f32(&this_data[i]);
            float32x4_t vb0 = vld1q_f32(&other_data[i]);
            float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            float32x4_t va2 = vld1q_f32(&this_data[i + 8]);
            float32x4_t vb2 = vld1q_f32(&other_data[i + 8]);
            float32x4_t va3 = vld1q_f32(&this_data[i + 12]);
            float32x4_t vb3 = vld1q_f32(&other_data[i + 12]);
            
            float32x4_t r0 = vaddq_f32(va0, vb0);
            float32x4_t r1 = vaddq_f32(va1, vb1);
            float32x4_t r2 = vaddq_f32(va2, vb2);
            float32x4_t r3 = vaddq_f32(va3, vb3);
            
            vst1q_f32(&this_data[i], r0);
            vst1q_f32(&this_data[i + 4], r1);
            vst1q_f32(&this_data[i + 8], r2);
            vst1q_f32(&this_data[i + 12], r3);
        }

        // Handle remaining multiples of 8
        std::size_t len8 = ((size_ - len16) / 8) * 8 + len16;
        for (std::size_t i = len16; i < len8; i += 8) {
            float32x4_t va0 = vld1q_f32(&this_data[i]);
            float32x4_t vb0 = vld1q_f32(&other_data[i]);
            float32x4_t va1 = vld1q_f32(&this_data[i + 4]);
            float32x4_t vb1 = vld1q_f32(&other_data[i + 4]);
            
            float32x4_t r0 = vaddq_f32(va0, vb0);
            float32x4_t r1 = vaddq_f32(va1, vb1);
            
            vst1q_f32(&this_data[i], r0);
            vst1q_f32(&this_data[i + 4], r1);
        }

        // Handle remaining multiples of 4
        std::size_t len4 = ((size_ - len8) / 4) * 4 + len8;
        for (std::size_t i = len8; i < len4; i += 4) {
            float32x4_t va = vld1q_f32(&this_data[i]);
            float32x4_t vb = vld1q_f32(&other_data[i]);
            float32x4_t result = vaddq_f32(va, vb);
            vst1q_f32(&this_data[i], result);
        }

        // Scalar remainder
        for (std::size_t i = len4; i < size_; ++i) {
            this_data[i] += other_data[i];
        }
    }

    void add_inplace_int8(const TensorBackend& other) {
        const int8_t* other_data = other.data_int8();
        int8_t* this_data = data_int8();

        // NEON loop: process 16 int8s at a time (128-bit register)
        std::size_t neon_len = (size_ / 16) * 16;
        for (std::size_t i = 0; i < neon_len; i += 16) {
            int8x16_t va = vld1q_s8(&this_data[i]);
            int8x16_t vb = vld1q_s8(&other_data[i]);
            int8x16_t result = vaddq_s8(va, vb);
            vst1q_s8(&this_data[i], result);
        }

        // Handle remainder with scalar saturation
        for (std::size_t i = neon_len; i < size_; ++i) {
            int16_t sum = static_cast<int16_t>(this_data[i]) + 
                         static_cast<int16_t>(other_data[i]);
            this_data[i] = static_cast<int8_t>(
                std::max(-128, std::min(127, static_cast<int>(sum)))
            );
        }
    }
};
#endif

// Tensor implementation
Tensor::Tensor() : quantization_mode_(QuantizationMode::FP32) {
    init_backend(0);
}

Tensor::Tensor(const std::vector<float>& data, QuantizationMode mode)
    : quantization_mode_(mode) {
    init_backend(data.size());
    if (!data.empty()) {
        float* fp32_data = data_fp32();
        #pragma omp parallel for
        for (std::size_t i = 0; i < data.size(); ++i) {
            fp32_data[i] = data[i];
        }
    }
}

Tensor::Tensor(const std::vector<int>& data, QuantizationMode mode)
    : quantization_mode_(mode) {
    init_backend(data.size());
    if (mode == QuantizationMode::FP32) {
        float* fp32_data = data_fp32();
        #pragma omp parallel for
        for (std::size_t i = 0; i < data.size(); ++i) {
            fp32_data[i] = static_cast<float>(data[i]);
        }
    } else if (mode == QuantizationMode::INT8) {
        int8_t* int8_data = data_int8();
        #pragma omp parallel for
        for (std::size_t i = 0; i < data.size(); ++i) {
            int8_data[i] = static_cast<int8_t>(
                std::max(-128, std::min(127, data[i]))
            );
        }
    }
}

Tensor::Tensor(float scalar, QuantizationMode mode)
    : quantization_mode_(mode) {
    init_backend(1);
    data_fp32()[0] = scalar;
}

Tensor::Tensor(int scalar, QuantizationMode mode)
    : quantization_mode_(mode) {
    init_backend(1);
    if (mode == QuantizationMode::FP32) {
        data_fp32()[0] = static_cast<float>(scalar);
    } else if (mode == QuantizationMode::INT8) {
        data_int8()[0] = static_cast<int8_t>(
            std::max(-128, std::min(127, scalar))
        );
    }
}

Tensor::~Tensor() = default;

Tensor::Tensor(const Tensor& other)
    : quantization_mode_(other.quantization_mode_),
      backend_(other.backend_ ? other.backend_->clone() : nullptr) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        quantization_mode_ = other.quantization_mode_;
        backend_ = other.backend_ ? other.backend_->clone() : nullptr;
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : quantization_mode_(other.quantization_mode_),
      backend_(std::move(other.backend_)) {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        quantization_mode_ = other.quantization_mode_;
        backend_ = std::move(other.backend_);
    }
    return *this;
}

std::size_t Tensor::size() const {
    return backend_ ? backend_->size() : 0;
}

float* Tensor::data_fp32() {
    return backend_ ? backend_->data_fp32() : nullptr;
}

const float* Tensor::data_fp32() const {
    return backend_ ? backend_->data_fp32() : nullptr;
}

int8_t* Tensor::data_int8() {
    return backend_ ? backend_->data_int8() : nullptr;
}

const int8_t* Tensor::data_int8() const {
    return backend_ ? backend_->data_int8() : nullptr;
}

Tensor Tensor::add(const Tensor& other) const {
    Tensor result(*this);
    result.add_inplace(other);
    return result;
}

void Tensor::add_inplace(const Tensor& other) {
    if (!backend_) {
        throw std::runtime_error("Tensor backend not initialized");
    }
    backend_->add_inplace(*other.backend_);
}

void Tensor::init_backend(std::size_t size) {
#if LABNEURA_HAVE_M1
    backend_ = std::make_unique<NEONBackend>(size, quantization_mode_);
#else
    backend_ = std::make_unique<GenericBackend>(size, quantization_mode_);
#endif
}

} // namespace labneura
