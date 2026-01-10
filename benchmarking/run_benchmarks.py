#!/usr/bin/env python3
"""
Main Benchmarking Script

Runs comprehensive benchmarks and generates reports.
Usage: python run_benchmarks.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.benchmark_runner import BenchmarkRunner
from benchmarking.metrics_calculator import MetricsCalculator


def run_comprehensive_benchmarks():
    """Run comprehensive benchmarks across all sizes and operations."""
    
    print("=" * 80)
    print("LabNeura Comprehensive Benchmarking Suite")
    print("=" * 80)
    print()
    
    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
    operations = ['addition', 'multiplication', 'subtraction']
    
    runner = BenchmarkRunner(warmup_iterations=5, benchmark_iterations=100)
    all_results = {}
    
    print("Running benchmarks...")
    print()
    
    for size in sizes:
        print(f"\n{'='*80}")
        print(f"Size: {size:,} elements")
        print(f"{'='*80}")
        
        all_results[size] = {}
        
        # Addition
        print(f"\nAddition Benchmark...")
        add_results = runner.benchmark_addition(size)
        all_results[size]['addition'] = add_results
        
        print(f"  LabNeura: {add_results['labneura'].flops_str} | "
              f"Time: {add_results['labneura'].total_time*1000:.2f}ms")
        print(f"  NumPy:    {add_results['numpy'].flops_str} | "
              f"Time: {add_results['numpy'].total_time*1000:.2f}ms")
        print(f"  PyTorch:  {add_results['pytorch'].flops_str} | "
              f"Time: {add_results['pytorch'].total_time*1000:.2f}ms")
        
        speedups = runner.get_comparison_speedup(add_results, baseline='numpy')
        print(f"  Speedup vs NumPy:")
        print(f"    LabNeura: {speedups['labneura']:.2f}x")
        print(f"    PyTorch:  {speedups['pytorch']:.2f}x")
        
        # Multiplication
        print(f"\nMultiplication Benchmark...")
        mul_results = runner.benchmark_multiplication(size)
        all_results[size]['multiplication'] = mul_results
        
        print(f"  LabNeura: {mul_results['labneura'].flops_str} | "
              f"Time: {mul_results['labneura'].total_time*1000:.2f}ms")
        print(f"  NumPy:    {mul_results['numpy'].flops_str} | "
              f"Time: {mul_results['numpy'].total_time*1000:.2f}ms")
        print(f"  PyTorch:  {mul_results['pytorch'].flops_str} | "
              f"Time: {mul_results['pytorch'].total_time*1000:.2f}ms")
        
        speedups = runner.get_comparison_speedup(mul_results, baseline='numpy')
        print(f"  Speedup vs NumPy:")
        print(f"    LabNeura: {speedups['labneura']:.2f}x")
        print(f"    PyTorch:  {speedups['pytorch']:.2f}x")
        
        # Subtraction
        print(f"\nSubtraction Benchmark...")
        sub_results = runner.benchmark_subtraction(size)
        all_results[size]['subtraction'] = sub_results
        
        print(f"  LabNeura: {sub_results['labneura'].flops_str} | "
              f"Time: {sub_results['labneura'].total_time*1000:.2f}ms")
        print(f"  NumPy:    {sub_results['numpy'].flops_str} | "
              f"Time: {sub_results['numpy'].total_time*1000:.2f}ms")
        print(f"  PyTorch:  {sub_results['pytorch'].flops_str} | "
              f"Time: {sub_results['pytorch'].total_time*1000:.2f}ms")
        
        speedups = runner.get_comparison_speedup(sub_results, baseline='numpy')
        print(f"  Speedup vs NumPy:")
        print(f"    LabNeura: {speedups['labneura']:.2f}x")
        print(f"    PyTorch:  {speedups['pytorch']:.2f}x")
        
        # INT8 Operations
        print(f"\nINT8 Addition Benchmark...")
        int8_results = runner.benchmark_int8_operations(size)
        all_results[size]['int8_addition'] = int8_results['add']
        
        print(f"  LabNeura: {int8_results['add']['labneura'].flops_str} | "
              f"Time: {int8_results['add']['labneura'].total_time*1000:.2f}ms")
        print(f"  NumPy:    {int8_results['add']['numpy'].flops_str} | "
              f"Time: {int8_results['add']['numpy'].total_time*1000:.2f}ms")
        print(f"  PyTorch:  {int8_results['add']['pytorch'].flops_str} | "
              f"Time: {int8_results['add']['pytorch'].total_time*1000:.2f}ms")
        
        speedups = runner.get_comparison_speedup(int8_results['add'], baseline='numpy')
        print(f"  Speedup vs NumPy:")
        print(f"    LabNeura: {speedups['labneura']:.2f}x")
        print(f"    PyTorch:  {speedups['pytorch']:.2f}x")
        
        # INT16 Operations
        print(f"\nINT16 Addition Benchmark...")
        int16_results = runner.benchmark_int16_operations(size)
        all_results[size]['int16_addition'] = int16_results['add']
        
        print(f"  LabNeura: {int16_results['add']['labneura'].flops_str} | "
              f"Time: {int16_results['add']['labneura'].total_time*1000:.2f}ms")
        print(f"  NumPy:    {int16_results['add']['numpy'].flops_str} | "
              f"Time: {int16_results['add']['numpy'].total_time*1000:.2f}ms")
        print(f"  PyTorch:  {int16_results['add']['pytorch'].flops_str} | "
              f"Time: {int16_results['add']['pytorch'].total_time*1000:.2f}ms")
        
        speedups = runner.get_comparison_speedup(int16_results['add'], baseline='numpy')
        print(f"  Speedup vs NumPy:")
        print(f"    LabNeura: {speedups['labneura']:.2f}x")
        print(f"    PyTorch:  {speedups['pytorch']:.2f}x")
        
        # FP16 Operations
        print(f"\nFP16 Addition Benchmark...")
        fp16_results = runner.benchmark_fp16_operations(size)
        all_results[size]['fp16_addition'] = fp16_results['add']
        
        print(f"  LabNeura: {fp16_results['add']['labneura'].flops_str} | "
              f"Time: {fp16_results['add']['labneura'].total_time*1000:.2f}ms")
        print(f"  NumPy:    {fp16_results['add']['numpy'].flops_str} | "
              f"Time: {fp16_results['add']['numpy'].total_time*1000:.2f}ms")
        print(f"  PyTorch:  {fp16_results['add']['pytorch'].flops_str} | "
              f"Time: {fp16_results['add']['pytorch'].total_time*1000:.2f}ms")
        
        speedups = runner.get_comparison_speedup(fp16_results['add'], baseline='numpy')
        print(f"  Speedup vs NumPy:")
        print(f"    LabNeura: {speedups['labneura']:.2f}x")
        print(f"    PyTorch:  {speedups['pytorch']:.2f}x")
    
    print(f"\n\n{'='*80}")
    print("Benchmark Complete!")
    print(f"{'='*80}\n")
    
    return all_results


def export_results(results: dict, output_file: str = 'benchmark_results.json'):
    """Export results to JSON file."""
    
    # Convert results to serializable format
    serializable_results = {}
    
    for size, ops in results.items():
        serializable_results[size] = {}
        for op, libs in ops.items():
            serializable_results[size][op] = {}
            for lib, metrics in libs.items():
                serializable_results[size][op][lib] = {
                    'flops': metrics.flops,
                    'flops_str': metrics.flops_str,
                    'total_time_ms': metrics.total_time * 1000,
                    'latency_stats': metrics.latency_stats,
                    'throughput': metrics.throughput,
                }
    
    output_path = Path(__file__).parent / output_file
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results exported to: {output_path}")


if __name__ == '__main__':
    results = run_comprehensive_benchmarks()
    export_results(results)
