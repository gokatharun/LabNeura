#!/usr/bin/env python3
"""
Benchmark INT16 and FP16 operations: LabNeura vs NumPy vs PyTorch
Measures add, sub, mul operations for comparison.
"""
import sys
import time
import json
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, skipping torch benchmarks")

sys.path.insert(0, '/Users/tharungoka/LabNeura')
import labneura


def time_operation(func, iterations=100):
    """Time a function over multiple iterations."""
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    end = time.perf_counter()
    return (end - start) / iterations


def benchmark_add_int16(size, iterations=100):
    """Benchmark ADD operation for INT16."""
    results = {}
    
    # LabNeura INT16
    a_ln = labneura.Tensor(np.random.randint(-100, 100, size, dtype=np.int32).tolist(), 
                           labneura.QuantizationMode.INT16)
    b_ln = labneura.Tensor(np.random.randint(-100, 100, size, dtype=np.int32).tolist(), 
                           labneura.QuantizationMode.INT16)
    
    def labneura_add():
        a_ln.add_inplace(b_ln)
    
    results['labneura_int16'] = time_operation(labneura_add, iterations)
    
    # NumPy int16
    a_np = np.random.randint(-100, 100, size, dtype=np.int16)
    b_np = np.random.randint(-100, 100, size, dtype=np.int16)
    
    def numpy_add():
        np.add(a_np, b_np, out=a_np, casting='unsafe')
    
    results['numpy_int16'] = time_operation(numpy_add, iterations)
    
    # PyTorch int16 if available
    if TORCH_AVAILABLE:
        a_torch = torch.randint(-100, 100, (size,), dtype=torch.int16)
        b_torch = torch.randint(-100, 100, (size,), dtype=torch.int16)
        
        def torch_add():
            torch.add(a_torch, b_torch, out=a_torch)
        
        results['torch_int16'] = time_operation(torch_add, iterations)
    
    return results


def benchmark_sub_int16(size, iterations=100):
    """Benchmark SUB operation for INT16."""
    results = {}
    
    # LabNeura INT16
    a_ln = labneura.Tensor(np.random.randint(-100, 100, size, dtype=np.int32).tolist(), 
                           labneura.QuantizationMode.INT16)
    b_ln = labneura.Tensor(np.random.randint(-100, 100, size, dtype=np.int32).tolist(), 
                           labneura.QuantizationMode.INT16)
    
    def labneura_sub():
        a_ln.sub_inplace(b_ln)
    
    results['labneura_int16'] = time_operation(labneura_sub, iterations)
    
    # NumPy int16
    a_np = np.random.randint(-100, 100, size, dtype=np.int16)
    b_np = np.random.randint(-100, 100, size, dtype=np.int16)
    
    def numpy_sub():
        np.subtract(a_np, b_np, out=a_np, casting='unsafe')
    
    results['numpy_int16'] = time_operation(numpy_sub, iterations)
    
    # PyTorch int16
    if TORCH_AVAILABLE:
        a_torch = torch.randint(-100, 100, (size,), dtype=torch.int16)
        b_torch = torch.randint(-100, 100, (size,), dtype=torch.int16)
        
        def torch_sub():
            torch.sub(a_torch, b_torch, out=a_torch)
        
        results['torch_int16'] = time_operation(torch_sub, iterations)
    
    return results


def benchmark_mul_int16(size, iterations=100):
    """Benchmark MUL operation for INT16."""
    results = {}
    
    # LabNeura INT16
    a_ln = labneura.Tensor(np.random.randint(-100, 100, size, dtype=np.int32).tolist(), 
                           labneura.QuantizationMode.INT16)
    b_ln = labneura.Tensor(np.random.randint(-100, 100, size, dtype=np.int32).tolist(), 
                           labneura.QuantizationMode.INT16)
    
    def labneura_mul():
        a_ln.mul_inplace(b_ln)
    
    results['labneura_int16'] = time_operation(labneura_mul, iterations)
    
    # NumPy int16
    a_np = np.random.randint(-100, 100, size, dtype=np.int16)
    b_np = np.random.randint(-100, 100, size, dtype=np.int16)
    
    def numpy_mul():
        np.multiply(a_np, b_np, out=a_np, casting='unsafe')
    
    results['numpy_int16'] = time_operation(numpy_mul, iterations)
    
    # PyTorch int16
    if TORCH_AVAILABLE:
        a_torch = torch.randint(-100, 100, (size,), dtype=torch.int16)
        b_torch = torch.randint(-100, 100, (size,), dtype=torch.int16)
        
        def torch_mul():
            torch.mul(a_torch, b_torch, out=a_torch)
        
        results['torch_int16'] = time_operation(torch_mul, iterations)
    
    return results


def benchmark_add_fp16(size, iterations=100):
    """Benchmark ADD operation for FP16."""
    results = {}
    
    # LabNeura FP16
    a_ln = labneura.Tensor(np.random.uniform(-1, 1, size).astype(np.float32).tolist(), 
                           labneura.QuantizationMode.FP16)
    b_ln = labneura.Tensor(np.random.uniform(-1, 1, size).astype(np.float32).tolist(), 
                           labneura.QuantizationMode.FP16)
    
    def labneura_add():
        a_ln.add_inplace(b_ln)
    
    results['labneura_fp16'] = time_operation(labneura_add, iterations)
    
    # NumPy float16
    a_np = np.random.uniform(-1, 1, size).astype(np.float16)
    b_np = np.random.uniform(-1, 1, size).astype(np.float16)
    
    def numpy_add():
        np.add(a_np, b_np, out=a_np)
    
    results['numpy_fp16'] = time_operation(numpy_add, iterations)
    
    # PyTorch float16
    if TORCH_AVAILABLE:
        a_torch = torch.randn(size, dtype=torch.float16)
        b_torch = torch.randn(size, dtype=torch.float16)
        
        def torch_add():
            torch.add(a_torch, b_torch, out=a_torch)
        
        results['torch_fp16'] = time_operation(torch_add, iterations)
    
    return results


def benchmark_sub_fp16(size, iterations=100):
    """Benchmark SUB operation for FP16."""
    results = {}
    
    # LabNeura FP16
    a_ln = labneura.Tensor(np.random.uniform(-1, 1, size).astype(np.float32).tolist(), 
                           labneura.QuantizationMode.FP16)
    b_ln = labneura.Tensor(np.random.uniform(-1, 1, size).astype(np.float32).tolist(), 
                           labneura.QuantizationMode.FP16)
    
    def labneura_sub():
        a_ln.sub_inplace(b_ln)
    
    results['labneura_fp16'] = time_operation(labneura_sub, iterations)
    
    # NumPy float16
    a_np = np.random.uniform(-1, 1, size).astype(np.float16)
    b_np = np.random.uniform(-1, 1, size).astype(np.float16)
    
    def numpy_sub():
        np.subtract(a_np, b_np, out=a_np)
    
    results['numpy_fp16'] = time_operation(numpy_sub, iterations)
    
    # PyTorch float16
    if TORCH_AVAILABLE:
        a_torch = torch.randn(size, dtype=torch.float16)
        b_torch = torch.randn(size, dtype=torch.float16)
        
        def torch_sub():
            torch.sub(a_torch, b_torch, out=a_torch)
        
        results['torch_fp16'] = time_operation(torch_sub, iterations)
    
    return results


def benchmark_mul_fp16(size, iterations=100):
    """Benchmark MUL operation for FP16."""
    results = {}
    
    # LabNeura FP16
    a_ln = labneura.Tensor(np.random.uniform(-1, 1, size).astype(np.float32).tolist(), 
                           labneura.QuantizationMode.FP16)
    b_ln = labneura.Tensor(np.random.uniform(-1, 1, size).astype(np.float32).tolist(), 
                           labneura.QuantizationMode.FP16)
    
    def labneura_mul():
        a_ln.mul_inplace(b_ln)
    
    results['labneura_fp16'] = time_operation(labneura_mul, iterations)
    
    # NumPy float16
    a_np = np.random.uniform(-1, 1, size).astype(np.float16)
    b_np = np.random.uniform(-1, 1, size).astype(np.float16)
    
    def numpy_mul():
        np.multiply(a_np, b_np, out=a_np)
    
    results['numpy_fp16'] = time_operation(numpy_mul, iterations)
    
    # PyTorch float16
    if TORCH_AVAILABLE:
        a_torch = torch.randn(size, dtype=torch.float16)
        b_torch = torch.randn(size, dtype=torch.float16)
        
        def torch_mul():
            torch.mul(a_torch, b_torch, out=a_torch)
        
        results['torch_fp16'] = time_operation(torch_mul, iterations)
    
    return results


def compute_speedups(baseline_time, results):
    """Compute speedup ratios relative to LabNeura."""
    speedups = {}
    for lib, time_val in results.items():
        if lib != 'labneura_int16' and lib != 'labneura_fp16':
            speedup = baseline_time / time_val
            speedups[lib] = speedup
    return speedups


def main():
    """Run all benchmarks."""
    sizes = [1000, 10000, 100000, 1000000]
    all_results = {}
    
    print("=" * 80)
    print("LABNEURA INT16/FP16 BENCHMARK vs NumPy & PyTorch")
    print("=" * 80)
    
    for size in sizes:
        print(f"\n{'=' * 80}")
        print(f"Tensor Size: {size:,} elements")
        print(f"{'=' * 80}")
        
        # INT16 Benchmarks
        print("\n--- INT16 ADD ---")
        int16_add = benchmark_add_int16(size, iterations=50)
        for lib, time_val in int16_add.items():
            print(f"  {lib:20s}: {time_val*1e6:10.2f} μs")
        speedups = compute_speedups(int16_add.get('labneura_int16', 1), int16_add)
        for lib, speedup in speedups.items():
            print(f"    {lib} speedup: {speedup:.2f}x")
        
        print("\n--- INT16 SUB ---")
        int16_sub = benchmark_sub_int16(size, iterations=50)
        for lib, time_val in int16_sub.items():
            print(f"  {lib:20s}: {time_val*1e6:10.2f} μs")
        speedups = compute_speedups(int16_sub.get('labneura_int16', 1), int16_sub)
        for lib, speedup in speedups.items():
            print(f"    {lib} speedup: {speedup:.2f}x")
        
        print("\n--- INT16 MUL ---")
        int16_mul = benchmark_mul_int16(size, iterations=50)
        for lib, time_val in int16_mul.items():
            print(f"  {lib:20s}: {time_val*1e6:10.2f} μs")
        speedups = compute_speedups(int16_mul.get('labneura_int16', 1), int16_mul)
        for lib, speedup in speedups.items():
            print(f"    {lib} speedup: {speedup:.2f}x")
        
        # FP16 Benchmarks
        print("\n--- FP16 ADD ---")
        fp16_add = benchmark_add_fp16(size, iterations=50)
        for lib, time_val in fp16_add.items():
            print(f"  {lib:20s}: {time_val*1e6:10.2f} μs")
        speedups = compute_speedups(fp16_add.get('labneura_fp16', 1), fp16_add)
        for lib, speedup in speedups.items():
            print(f"    {lib} speedup: {speedup:.2f}x")
        
        print("\n--- FP16 SUB ---")
        fp16_sub = benchmark_sub_fp16(size, iterations=50)
        for lib, time_val in fp16_sub.items():
            print(f"  {lib:20s}: {time_val*1e6:10.2f} μs")
        speedups = compute_speedups(fp16_sub.get('labneura_fp16', 1), fp16_sub)
        for lib, speedup in speedups.items():
            print(f"    {lib} speedup: {speedup:.2f}x")
        
        print("\n--- FP16 MUL ---")
        fp16_mul = benchmark_mul_fp16(size, iterations=50)
        for lib, time_val in fp16_mul.items():
            print(f"  {lib:20s}: {time_val*1e6:10.2f} μs")
        speedups = compute_speedups(fp16_mul.get('labneura_fp16', 1), fp16_mul)
        for lib, speedup in speedups.items():
            print(f"    {lib} speedup: {speedup:.2f}x")
        
        all_results[size] = {
            'int16_add': int16_add,
            'int16_sub': int16_sub,
            'int16_mul': int16_mul,
            'fp16_add': fp16_add,
            'fp16_sub': fp16_sub,
            'fp16_mul': fp16_mul,
        }
    
    # Save results
    with open('benchmark_int16_fp16_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to benchmark_int16_fp16_results.json")


if __name__ == '__main__':
    main()
