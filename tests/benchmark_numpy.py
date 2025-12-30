#!/usr/bin/env python3
"""
Benchmark LabNeura Tensor backend vs NumPy
Demonstrates performance advantage of NEON on Apple M1
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/Users/tharungoka/LabNeura/python')
import labneura

def benchmark_labneura(size, iterations):
    """Benchmark LabNeura Tensor addition"""
    data = [1.5] * size
    t1 = labneura.Tensor(data, labneura.QuantizationMode.FP32)
    t2 = labneura.Tensor(data, labneura.QuantizationMode.FP32)
    
    start = time.perf_counter()
    for _ in range(iterations):
        t1.add_inplace(t2)
    elapsed = time.perf_counter() - start
    
    return elapsed

def benchmark_numpy(size, iterations):
    """Benchmark NumPy array addition"""
    a = np.full(size, 1.5, dtype=np.float32)
    b = np.full(size, 1.5, dtype=np.float32)
    
    start = time.perf_counter()
    for _ in range(iterations):
        a += b
    elapsed = time.perf_counter() - start
    
    return elapsed

def benchmark_python_list(size, iterations):
    """Benchmark Python list addition (baseline)"""
    a = [1.5] * size
    b = [1.5] * size
    
    start = time.perf_counter()
    for _ in range(iterations):
        a = [x + y for x, y in zip(a, b)]
    elapsed = time.perf_counter() - start
    
    return elapsed

def main():
    print("=" * 70)
    print("LabNeura vs NumPy Performance Benchmark")
    print("=" * 70)
    print()
    
    # Hardware detection
    print("=== Hardware Detection ===")
    import platform
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    
    try:
        import subprocess
        sysctl_output = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
        print(f"CPU: {sysctl_output}")
    except:
        pass
    
    # Check if NEON is being used (compile-time detection)
    print("\nLabNeura backend: NEON (M1) or Generic (x86)")
    print("(Detected at C++ compile-time)")
    print()
    
    # Test correctness first
    print("=== Correctness Test ===")
    t1 = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP32)
    t2 = labneura.Tensor([1.0, 1.0, 1.0], labneura.QuantizationMode.FP32)
    t1.add_inplace(t2)
    result = t1.data_fp32()
    print(f"[1, 2, 3] + [1, 1, 1] = {result}")
    assert result == [2.0, 3.0, 4.0], "Correctness test failed!"
    print("✓ Correctness test passed\n")
    
    # Benchmarks
    print("=== Performance Benchmarks ===\n")
    
    test_sizes = [10_000, 100_000, 1_000_000]
    iterations = 100
    
    for size in test_sizes:
        print(f"Array size: {size:,} elements, {iterations} iterations")
        print("-" * 70)
        
        # LabNeura
        t_labneura = benchmark_labneura(size, iterations)
        ops_labneura = (size * iterations) / (t_labneura * 1e6)
        
        # NumPy
        t_numpy = benchmark_numpy(size, iterations)
        ops_numpy = (size * iterations) / (t_numpy * 1e6)
        
        # Python list (baseline)
        if size <= 100_000:  # Only test for smaller sizes (too slow for 1M)
            t_python = benchmark_python_list(size, iterations)
            ops_python = (size * iterations) / (t_python * 1e6)
        else:
            t_python = float('inf')
            ops_python = 0
        
        # Print results
        print(f"{'Backend':<20} {'Time (ms)':>15} {'Throughput (M ops/s)':>20}")
        print("-" * 70)
        print(f"{'LabNeura Tensor':<20} {t_labneura*1000:>15.3f} {ops_labneura:>20.2f}")
        print(f"{'NumPy':<20} {t_numpy*1000:>15.3f} {ops_numpy:>20.2f}")
        if ops_python > 0:
            print(f"{'Python list':<20} {t_python*1000:>15.3f} {ops_python:>20.2f}")
        
        # Calculate speedup
        speedup = t_numpy / t_labneura
        print()
        print(f"LabNeura speedup vs NumPy: {speedup:.2f}x")
        print()
        
        if t_python < float('inf'):
            speedup_vs_python = t_python / t_labneura
            print(f"LabNeura speedup vs Python: {speedup_vs_python:.2f}x")
            print()
    
    # INT8 quantization benchmark
    print("=" * 70)
    print("=== INT8 Quantization Benchmark ===")
    print()
    
    size = 1_000_000
    iterations = 100
    
    # LabNeura INT8
    int_data = [10] * size
    t1 = labneura.Tensor(int_data, labneura.QuantizationMode.INT8)
    t2 = labneura.Tensor(int_data, labneura.QuantizationMode.INT8)
    
    start = time.perf_counter()
    for _ in range(iterations):
        t1.add_inplace(t2)
    t_int8 = time.perf_counter() - start
    
    # NumPy int8
    a_int8 = np.full(size, 10, dtype=np.int8)
    b_int8 = np.full(size, 10, dtype=np.int8)
    
    start = time.perf_counter()
    for _ in range(iterations):
        a_int8 = np.clip(a_int8.astype(np.int16) + b_int8.astype(np.int16), -128, 127).astype(np.int8)
    t_numpy_int8 = time.perf_counter() - start
    
    ops_int8 = (size * iterations) / (t_int8 * 1e6)
    ops_numpy_int8 = (size * iterations) / (t_numpy_int8 * 1e6)
    
    print(f"Array size: {size:,} elements, {iterations} iterations (INT8)")
    print("-" * 70)
    print(f"{'Backend':<20} {'Time (ms)':>15} {'Throughput (M ops/s)':>20}")
    print("-" * 70)
    print(f"{'LabNeura INT8':<20} {t_int8*1000:>15.3f} {ops_int8:>20.2f}")
    print(f"{'NumPy INT8':<20} {t_numpy_int8*1000:>15.3f} {ops_numpy_int8:>20.2f}")
    print()
    speedup_int8 = t_numpy_int8 / t_int8
    print(f"LabNeura INT8 speedup vs NumPy INT8: {speedup_int8:.2f}x")
    print()
    
    # Memory comparison
    print("=" * 70)
    print("=== Memory Usage Comparison ===")
    print()
    
    size = 1_000_000
    
    # FP32: 4 bytes per element
    fp32_bytes = size * 4 / (1024**2)
    
    # INT8: 1 byte per element
    int8_bytes = size * 1 / (1024**2)
    
    print(f"Array size: {size:,} elements")
    print(f"FP32 memory: {fp32_bytes:.2f} MB")
    print(f"INT8 memory: {int8_bytes:.2f} MB")
    print(f"Memory savings with INT8: {(1 - int8_bytes/fp32_bytes)*100:.1f}%")
    print()
    
    # Performance per byte
    print("=== Efficiency (throughput per MB/s) ===")
    print()
    
    # Reuse previous benchmark results
    fp32_speed = ops_labneura  # Use last FP32 result (1M elements)
    int8_speed = ops_int8
    
    fp32_bytes_per_sec = fp32_speed * 1e6 * 4 / (1024**2)  # 4 bytes per float
    int8_bytes_per_sec = int8_speed * 1e6 * 1 / (1024**2)  # 1 byte per int8
    
    print(f"FP32 bandwidth: {fp32_bytes_per_sec:.0f} MB/s")
    print(f"INT8 bandwidth: {int8_bytes_per_sec:.0f} MB/s")
    print()
    
    print("=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print()
    print("1. LabNeura with NEON (M1) uses vectorized operations")
    print("   → 4 floats or 16 int8s added per instruction")
    print()
    print("2. INT8 quantization reduces memory by 4x")
    print("   → More data fits in CPU cache")
    print("   → Faster operations on M1 NEON")
    print()
    print("3. LabNeura is competitive or faster than NumPy")
    print("   → Especially for int8 quantized operations")
    print("   → Direct CPU register/SIMD usage")
    print()

if __name__ == "__main__":
    main()
