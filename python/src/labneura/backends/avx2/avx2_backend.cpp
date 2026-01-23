#if defined(__AVX2__)

#include "labneura/backends/avx2.h"
#include "avx2_internal.h"
#include "labneura/utils/alignment.hpp"


// =======================
// Register detection utility
// =======================

// Detect number of available AVX2 (YMM) registers using CPUID
// Returns 32 if AVX-512 is supported (extends YMM registers)
// Returns 16 for standard x86_64 (default)
static inline int detect_avx2_registers() {
    int num_registers = 16;  // Default: standard x86_64

    try {
        #if defined(__GNUC__) || defined(__clang__)
        unsigned int eax, ebx, ecx, edx;
        
        // Check CPUID support
        if (__get_cpuid_max(0, nullptr) >= 7) {
            // CPUID leaf 7, subleaf 0: Extended Features
            if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
                // Check EBX bit 16: AVX-512F (Foundation)
                // AVX-512 extends YMM registers from 16 to 32
                if (ebx & (1 << 16)) {
                    num_registers = 32;
                }
            }
        }
        #elif defined(_MSC_VER)
        int cpuInfo[4];
        __cpuidex(cpuInfo, 7, 0);
        // Check EBX bit 16: AVX-512F
        if (cpuInfo[1] & (1 << 16)) {
            num_registers = 32;
        }
        #endif
    } catch (...) {
        // If any error occurs, use safe default
        num_registers = 16;
    }

    return num_registers;
}

// Get optimized unroll factor: half of available registers
static inline int get_avx2_unroll_factor() {
    return detect_avx2_registers() / 2;  // 8 registers per operand
}

// FP32: 8 elements per YMM, unroll_factor=8 → 64 floats per iteration
static inline std::size_t get_avx2_fp32_chunk() {
    return get_avx2_unroll_factor() * 8;
}

// Use alignment utility from labneura::util

// =======================
// Constructor
// =======================

AVX2Backend::AVX2Backend(std::size_t size, QuantizationMode mode)
    : size_(size), quantization_mode_(mode) {

    // AVX2: 256-bit registers → 8 FP32 elements
    aligned_size_ = ((size + 7) / 8) * 8;

    if (mode == QuantizationMode::FP32) {
        float* p = labneura::util::allocate_aligned<float>(aligned_size_, labneura::util::AVX2_ALIGN);
        data_fp32_ = std::unique_ptr<float, void(*)(float*)>(p, labneura::util::free_aligned_float);
        std::fill(data_fp32_.get() + size_, data_fp32_.get() + aligned_size_, 0.0f);
    } else if (mode == QuantizationMode::FP16) {
        int16_t* p = labneura::util::allocate_aligned<int16_t>(aligned_size_, labneura::util::AVX2_ALIGN);
        auto fp16_deleter = [](int16_t* ptr){ labneura::util::aligned_free(ptr); };
        data_fp16_ = std::unique_ptr<int16_t, void(*)(int16_t*)>(p, fp16_deleter);
        std::fill(data_fp16_.get() + size_, data_fp16_.get() + aligned_size_, static_cast<int16_t>(0));
    } else if (mode == QuantizationMode::INT16) {
        int16_t* p = labneura::util::allocate_aligned<int16_t>(aligned_size_, labneura::util::AVX2_ALIGN);
        auto int16_deleter = [](int16_t* ptr){ labneura::util::aligned_free(ptr); };
        data_int16_ = std::unique_ptr<int16_t, void(*)(int16_t*)>(p, int16_deleter);
        std::fill(data_int16_.get() + size_, data_int16_.get() + aligned_size_, static_cast<int16_t>(0));
    } else {
        int8_t* p = labneura::util::allocate_aligned<int8_t>(aligned_size_, labneura::util::AVX2_ALIGN);
        data_int8_ = std::unique_ptr<int8_t, void(*)(int8_t*)>(p, labneura::util::free_aligned_int8);
        std::fill(data_int8_.get() + size_, data_int8_.get() + aligned_size_, 0);
    }
}

// =======================
// Metadata
// =======================

std::size_t AVX2Backend::size() const {
    return size_;
}

QuantizationMode AVX2Backend::quantization_mode() const {
    return quantization_mode_;
}

// =======================
// Data access
// =======================

float* AVX2Backend::data_fp32() {
    if (quantization_mode_ != QuantizationMode::FP32) {
        throw std::runtime_error("Tensor is not in FP32 mode");
    }
    return data_fp32_.get();
}

const float* AVX2Backend::data_fp32() const {
    if (quantization_mode_ != QuantizationMode::FP32) {
        throw std::runtime_error("Tensor is not in FP32 mode");
    }
    return data_fp32_.get();
}

int16_t* AVX2Backend::data_fp16() {
    if (quantization_mode_ != QuantizationMode::FP16) {
        throw std::runtime_error("Tensor is not in FP16 mode");
    }
    return data_fp16_.get();
}

const int16_t* AVX2Backend::data_fp16() const {
    if (quantization_mode_ != QuantizationMode::FP16) {
        throw std::runtime_error("Tensor is not in FP16 mode");
    }
    return data_fp16_.get();
}

int16_t* AVX2Backend::data_int16() {
    if (quantization_mode_ != QuantizationMode::INT16) {
        throw std::runtime_error("Tensor is not in INT16 mode");
    }
    return data_int16_.get();
}

const int16_t* AVX2Backend::data_int16() const {
    if (quantization_mode_ != QuantizationMode::INT16) {
        throw std::runtime_error("Tensor is not in INT16 mode");
    }
    return data_int16_.get();
}

int8_t* AVX2Backend::data_int8() {
    if (quantization_mode_ != QuantizationMode::INT8) {
        throw std::runtime_error("Tensor is not in INT8 mode");
    }
    return data_int8_.get();
}

const int8_t* AVX2Backend::data_int8() const {
    if (quantization_mode_ != QuantizationMode::INT8) {
        throw std::runtime_error("Tensor is not in INT8 mode");
    }
    return data_int8_.get();
}

// =======================
// Clone
// =======================

std::unique_ptr<TensorBackend> AVX2Backend::clone() const {
    auto backend = std::make_unique<AVX2Backend>(size_, quantization_mode_);
    if (quantization_mode_ == QuantizationMode::FP32) {
        std::copy(data_fp32_.get(), data_fp32_.get() + size_, backend->data_fp32());
    } else if (quantization_mode_ == QuantizationMode::FP16) {
        std::copy(data_fp16_.get(), data_fp16_.get() + size_, backend->data_fp16());
    } else if (quantization_mode_ == QuantizationMode::INT16) {
        std::copy(data_int16_.get(), data_int16_.get() + size_, backend->data_int16());
    } else {
        std::copy(data_int8_.get(), data_int8_.get() + size_, backend->data_int8());
    }
    return backend;
}

} // namespace labneura

#endif // __AVX2__