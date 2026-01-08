#!/usr/bin/env python3
"""
Benchmark Visualization and Summary

Generates visual summaries of benchmark results.
"""

import json
from pathlib import Path
from typing import Dict
import numpy as np


def load_benchmark_results(results_file: str = 'benchmark_results.json') -> Dict:
    """Load benchmark results from JSON file."""
    results_path = Path(__file__).parent / results_file
    with open(results_path, 'r') as f:
        return json.load(f)


def print_summary_table():
    """Print summary comparison table."""
    results = load_benchmark_results()
    
    print("\n" + "="*100)
    print("LabNeura vs NumPy vs PyTorch - Performance Summary (CPU Only)")
    print("="*100 + "\n")
    
    operations = ['addition', 'multiplication', 'subtraction', 'int8_addition']
    sizes = [100, 1000, 10000, 100000, 1000000]
    
    for op in operations:
        print(f"\n{op.upper()} BENCHMARK")
        print("-" * 100)
        print(f"{'Size':<12} {'LabNeura':<20} {'NumPy':<20} {'PyTorch':<20} {'Winner':<15}")
        print("-" * 100)
        
        for size in sizes:
            size_key = str(size)
            if size_key in results and op in results[size_key]:
                ln_flops = results[size_key][op]['labneura']['flops']
                np_flops = results[size_key][op]['numpy']['flops']
                pt_flops = results[size_key][op]['pytorch']['flops']
                
                ln_str = format_flops(ln_flops)
                np_str = format_flops(np_flops)
                pt_str = format_flops(pt_flops)
                
                winner = get_winner(ln_flops, np_flops, pt_flops)
                
                print(f"{size:<12,} {ln_str:<20} {np_str:<20} {pt_str:<20} {winner:<15}")


def format_flops(flops: float) -> str:
    """Format FLOPS into human-readable format."""
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPS"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPS"
    else:
        return f"{flops:.2f} FLOPS"


def get_winner(labneura: float, numpy: float, pytorch: float) -> str:
    """Determine which library performed best."""
    values = {'LabNeura': labneura, 'NumPy': numpy, 'PyTorch': pytorch}
    winner = max(values, key=values.get)
    return f"🏆 {winner}"


def print_speedup_summary():
    """Print speedup comparison vs NumPy."""
    results = load_benchmark_results()
    
    print("\n\n" + "="*100)
    print("SPEEDUP vs NumPy (Baseline = 1.0x)")
    print("="*100 + "\n")
    
    operations = ['addition', 'multiplication', 'subtraction', 'int8_addition']
    sizes = [100, 1000, 10000, 100000, 1000000]
    
    speedup_matrix = {}
    
    for op in operations:
        print(f"\n{op.upper()}")
        print("-" * 80)
        print(f"{'Size':<12} {'LabNeura':<15} {'PyTorch':<15} {'Avg Speedup':<15}")
        print("-" * 80)
        
        speeds = []
        for size in sizes:
            size_key = str(size)
            if size_key in results and op in results[size_key]:
                ln_flops = results[size_key][op]['labneura']['flops']
                np_flops = results[size_key][op]['numpy']['flops']
                pt_flops = results[size_key][op]['pytorch']['flops']
                
                ln_speedup = ln_flops / np_flops if np_flops > 0 else 0
                pt_speedup = pt_flops / np_flops if np_flops > 0 else 0
                
                speeds.append(ln_speedup)
                
                print(f"{size:<12,} {ln_speedup:<15.2f}x {pt_speedup:<15.2f}x")
        
        avg_speedup = np.mean(speeds) if speeds else 0
        print(f"{'Average':<12} {avg_speedup:<15.2f}x")
        speedup_matrix[op] = avg_speedup


def print_latency_analysis():
    """Print latency statistics."""
    results = load_benchmark_results()
    
    print("\n\n" + "="*100)
    print("LATENCY ANALYSIS (Total Time for 100 Iterations)")
    print("="*100 + "\n")
    
    operations = ['addition', 'multiplication', 'subtraction', 'int8_addition']
    sizes = [100000, 1000000]
    
    for op in operations:
        print(f"\n{op.upper()}")
        print("-" * 90)
        print(f"{'Size':<12} {'LabNeura':<20} {'NumPy':<20} {'PyTorch':<20}")
        print("-" * 90)
        
        for size in sizes:
            size_key = str(size)
            if size_key in results and op in results[size_key]:
                ln_time = results[size_key][op]['labneura']['total_time_ms']
                np_time = results[size_key][op]['numpy']['total_time_ms']
                pt_time = results[size_key][op]['pytorch']['total_time_ms']
                
                print(f"{size:<12,} {ln_time:<20.2f}ms {np_time:<20.2f}ms {pt_time:<20.2f}ms")


def print_throughput_analysis():
    """Print throughput analysis."""
    results = load_benchmark_results()
    
    print("\n\n" + "="*100)
    print("THROUGHPUT ANALYSIS (Elements per Second)")
    print("="*100 + "\n")
    
    operations = ['addition', 'multiplication', 'subtraction', 'int8_addition']
    sizes = [100000, 1000000]
    
    for op in operations:
        print(f"\n{op.upper()}")
        print("-" * 100)
        print(f"{'Size':<12} {'LabNeura':<25} {'NumPy':<25} {'PyTorch':<25}")
        print("-" * 100)
        
        for size in sizes:
            size_key = str(size)
            if size_key in results and op in results[size_key]:
                ln_time = results[size_key][op]['labneura']['total_time_ms'] / 1000
                np_time = results[size_key][op]['numpy']['total_time_ms'] / 1000
                pt_time = results[size_key][op]['pytorch']['total_time_ms'] / 1000
                
                ln_tput = (size * 100) / ln_time if ln_time > 0 else 0
                np_tput = (size * 100) / np_time if np_time > 0 else 0
                pt_tput = (size * 100) / pt_time if pt_time > 0 else 0
                
                ln_str = f"{ln_tput/1e9:.2f}B els/s"
                np_str = f"{np_tput/1e9:.2f}B els/s"
                pt_str = f"{pt_tput/1e9:.2f}B els/s"
                
                print(f"{size:<12,} {ln_str:<25} {np_str:<25} {pt_str:<25}")


if __name__ == '__main__':
    print_summary_table()
    print_speedup_summary()
    print_latency_analysis()
    print_throughput_analysis()
