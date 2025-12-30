#include <iostream>
#include <chrono>
#include "labneura/types.hpp"
#include "labneura/tensor.h"

// Detect hardware at runtime
void print_hardware_info() {
#if defined(__aarch64__) && defined(__APPLE__)
    std::cout << "✓ Apple Silicon M1 detected - NEON SIMD optimizations ENABLED" << std::endl;
#else
    std::cout << "✗ Generic CPU - using scalar fallback (no SIMD)" << std::endl;
#endif
}

int main() {
    std::cout << "=== Hardware Detection ===" << std::endl;
    print_hardware_info();
    
    // Example 1: Quantize/Dequantize
    std::cout << "\n=== Quantization Example ===" << std::endl;
    float x = 0.5f;
    float scale = 0.1f;
    int8_t q = labneura::quantize_int8(x, scale);
    float y = labneura::dequantize_int8(q, scale);
    std::cout << "Original: " << x << ", Quantized: " << int(q) 
              << ", Dequantized: " << y << std::endl;

    // Example 2: Tensor with float array
    std::cout << "\n=== Tensor Float Addition ===" << std::endl;
    std::vector<float> arr1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> arr2 = {0.5f, 0.5f, 0.5f, 0.5f};
    
    labneura::Tensor t1(arr1, labneura::QuantizationMode::FP32);
    labneura::Tensor t2(arr2, labneura::QuantizationMode::FP32);
    
    std::cout << "t1 size: " << t1.size() << std::endl;
    std::cout << "t2 size: " << t2.size() << std::endl;
    
    // In-place addition
    t1.add_inplace(t2);
    std::cout << "After t1.add_inplace(t2): ";
    for (std::size_t i = 0; i < t1.size(); ++i) {
        std::cout << t1.data_fp32()[i] << " ";
    }
    std::cout << std::endl;

    // Example 3: Tensor with int array (converted to FP32)
    std::cout << "\n=== Tensor Int to FP32 Addition ===" << std::endl;
    std::vector<int> int_arr = {5, 10, 15};
    labneura::Tensor t3(int_arr, labneura::QuantizationMode::FP32);
    
    std::cout << "t3 (from ints) data: ";
    for (std::size_t i = 0; i < t3.size(); ++i) {
        std::cout << t3.data_fp32()[i] << " ";
    }
    std::cout << std::endl;

    // Example 4: Tensor with int8 quantization
    std::cout << "\n=== Tensor INT8 Addition ===" << std::endl;
    std::vector<int> int_arr2 = {10, 20, 30};
    labneura::Tensor t4(int_arr2, labneura::QuantizationMode::INT8);
    labneura::Tensor t5(int_arr2, labneura::QuantizationMode::INT8);
    
    std::cout << "t4 (INT8) before: ";
    for (std::size_t i = 0; i < t4.size(); ++i) {
        std::cout << int(t4.data_int8()[i]) << " ";
    }
    std::cout << std::endl;
    
    t4.add_inplace(t5);
    
    std::cout << "t4 (INT8) after add: ";
    for (std::size_t i = 0; i < t4.size(); ++i) {
        std::cout << int(t4.data_int8()[i]) << " ";
    }
    std::cout << std::endl;

    // Example 5: Scalar tensor
    std::cout << "\n=== Scalar Tensor ===" << std::endl;
    labneura::Tensor scalar1(5.5f);
    labneura::Tensor scalar2(2.5f);
    scalar1.add_inplace(scalar2);
    std::cout << "5.5 + 2.5 = " << scalar1.data_fp32()[0] << std::endl;

    // Benchmark: FP32 addition performance
    std::cout << "\n=== Performance Benchmark ===" << std::endl;
    const std::size_t bench_size = 1000000;
    std::vector<float> bench_data(bench_size, 1.5f);
    labneura::Tensor bench_t1(bench_data, labneura::QuantizationMode::FP32);
    labneura::Tensor bench_t2(bench_data, labneura::QuantizationMode::FP32);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 100; ++iter) {
        bench_t1.add_inplace(bench_t2);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Added " << bench_size << " elements, 100 iterations in " 
              << duration.count() << " ms" << std::endl;
    std::cout << "Throughput: " << (bench_size * 100.0 / duration.count() / 1e6) 
              << " M elements/sec" << std::endl;

#if defined(__aarch64__) && defined(__APPLE__)
    std::cout << "(NEON SIMD: processes 4 floats per instruction)" << std::endl;
#else
    std::cout << "(Scalar: processes 1 float per iteration)" << std::endl;
#endif

    return 0;
}
