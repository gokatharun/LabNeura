#!/usr/bin/env python3
"""
Benchmark Visualization and Summary

Generates visual summaries of benchmark results.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np


def load_benchmark_results(results_file: str = 'benchmark_results.json') -> Dict:
    """Load benchmark results from JSON file."""
    results_path = Path(__file__).parent / results_file
    with open(results_path, 'r') as f:
        return json.load(f)


def print_summary_table():
    """Print summary comparison table with relative performance columns."""
    results = load_benchmark_results()
    
    print("\n" + "="*130)
    print("LabNeura vs NumPy vs PyTorch - Performance Summary (CPU Only)")
    print("="*130 + "\n")
    
    operations = ['addition', 'multiplication', 'subtraction', 'division', 'int8_addition', 'int16_addition', 'fp16_addition']
    sizes = [100, 1000, 10000, 100000, 1000000]
    
    for op in operations:
        print(f"\n{op.upper()} BENCHMARK")
        print("-" * 142)
        print(
            f"{'Size':<12} {'LabNeura':<16} {'NumPy':<16} {'PyTorch':<16} "
            f"{'LN/NP x':<10} {'LN/NP %':<10} {'LN/PT x':<10} {'LN/PT %':<10} {'Winner':<15}"
        )
        print("-" * 142)
        ln_vals, np_vals, pt_vals = [], [], []
        
        for size in sizes:
            size_key = str(size)
            if size_key in results and op in results[size_key]:
                ln_flops = results[size_key][op]['labneura']['flops']
                np_flops = results[size_key][op]['numpy']['flops']
                pt_flops = results[size_key][op]['pytorch']['flops']
                ln_vals.append(ln_flops)
                np_vals.append(np_flops)
                pt_vals.append(pt_flops)
                
                ln_str = format_flops(ln_flops)
                np_str = format_flops(np_flops)
                pt_str = format_flops(pt_flops)
                
                ln_np_x, ln_np_pct = format_ratio_columns(ln_flops, np_flops, higher_is_better=True)
                ln_pt_x, ln_pt_pct = format_ratio_columns(ln_flops, pt_flops, higher_is_better=True)
                
                winner = get_winner(ln_flops, np_flops, pt_flops)
                
                print(
                    f"{size:<12,} {ln_str:<16} {np_str:<16} {pt_str:<16} "
                    f"{ln_np_x:<10} {ln_np_pct:<10} {ln_pt_x:<10} {ln_pt_pct:<10} {winner:<15}"
                )

        if ln_vals and np_vals and pt_vals:
            avg_ln = float(np.mean(ln_vals))
            avg_np = float(np.mean(np_vals))
            avg_pt = float(np.mean(pt_vals))
            avg_ln_str = format_flops(avg_ln)
            avg_np_str = format_flops(avg_np)
            avg_pt_str = format_flops(avg_pt)
            avg_ln_np_x, avg_ln_np_pct = format_ratio_columns(avg_ln, avg_np, higher_is_better=True)
            avg_ln_pt_x, avg_ln_pt_pct = format_ratio_columns(avg_ln, avg_pt, higher_is_better=True)
            avg_winner = get_winner(avg_ln, avg_np, avg_pt)
            print("-" * 142)
            print(
                f"{'Average':<12} {avg_ln_str:<16} {avg_np_str:<16} {avg_pt_str:<16} "
                f"{avg_ln_np_x:<10} {avg_ln_np_pct:<10} {avg_ln_pt_x:<10} {avg_ln_pt_pct:<10} {avg_winner:<15}"
            )


def format_flops(flops: float) -> str:
    """Format FLOPS into human-readable format."""
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPS"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPS"
    return f"{flops:.2f} FLOPS"


def get_winner(labneura: float, numpy: float, pytorch: float) -> str:
    """Determine which library performed best."""
    values = {'LabNeura': labneura, 'NumPy': numpy, 'PyTorch': pytorch}
    winner = max(values, key=values.get)
    return f"🏆 {winner}"


def ratio_values(lab_value: float, other_value: float, higher_is_better: bool = True) -> Optional[Tuple[float, float]]:
    """Return numeric ratio and percentage delta."""
    if other_value <= 0 or lab_value <= 0:
        return None
    ratio = (lab_value / other_value) if higher_is_better else (other_value / lab_value)
    pct = (ratio - 1.0) * 100.0
    return ratio, pct


def format_ratio_columns(lab_value: float, other_value: float, higher_is_better: bool = True) -> Tuple[str, str]:
    """Return (x, % ) strings with consistent formatting."""
    ratio_result = ratio_values(lab_value, other_value, higher_is_better)
    if ratio_result is None:
        return "N/A", "N/A"
    ratio, pct = ratio_result
    return f"{ratio:.2f}x", f"{pct:+.0f}%"


def print_speedup_summary():
    """Print speedup comparison vs NumPy and PyTorch (with percentages)."""
    results = load_benchmark_results()
    
    print("\n\n" + "="*120)
    print("SPEEDUP vs NumPy (Baseline = 1.0x) and vs PyTorch")
    print("="*120 + "\n")
    
    operations = ['addition', 'multiplication', 'subtraction', 'division', 'int8_addition', 'int16_addition', 'fp16_addition']
    sizes = [100, 1000, 10000, 100000, 1000000]
    
    speedup_matrix = {}
    
    for op in operations:
        print(f"\n{op.upper()}")
        print("-" * 142)
        print(
            f"{'Size':<12} "
            f"{'LN/NP x':<10} {'LN/NP %':<10} "
            f"{'LN/PT x':<10} {'LN/PT %':<10} "
            f"{'PT/NP x':<10} {'PT/NP %':<10} "
            f"{'Avg LN/NP x':<12} {'Avg LN/NP %':<12}"
        )
        print("-" * 142)
        
        ln_np_list: list[float] = []
        ln_pt_list: list[float] = []
        pt_np_list: list[float] = []
        for size in sizes:
            size_key = str(size)
            if size_key in results and op in results[size_key]:
                ln_flops = results[size_key][op]['labneura']['flops']
                np_flops = results[size_key][op]['numpy']['flops']
                pt_flops = results[size_key][op]['pytorch']['flops']
                
                ln_speedup = ln_flops / np_flops if np_flops > 0 else 0
                pt_speedup_np = pt_flops / np_flops if np_flops > 0 else 0
                ln_speedup_pt = ln_flops / pt_flops if pt_flops > 0 else 0
                
                ln_np_list.append(ln_speedup)
                ln_pt_list.append(ln_speedup_pt)
                pt_np_list.append(pt_speedup_np)
                
                ln_np_x, ln_np_pct = format_ratio_columns(ln_flops, np_flops, higher_is_better=True)
                ln_pt_x, ln_pt_pct = format_ratio_columns(ln_flops, pt_flops, higher_is_better=True)
                pt_np_x, pt_np_pct = format_ratio_columns(pt_flops, np_flops, higher_is_better=True)
                
                print(
                    f"{size:<12,} "
                    f"{ln_np_x:<10} {ln_np_pct:<10} "
                    f"{ln_pt_x:<10} {ln_pt_pct:<10} "
                    f"{pt_np_x:<10} {pt_np_pct:<10} "
                    f"{'':<12} {'':<12}"
                )
        
        avg_ln_np = np.mean(ln_np_list) if ln_np_list else 0
        avg_ln_pt = np.mean(ln_pt_list) if ln_pt_list else 0
        avg_pt_np = np.mean(pt_np_list) if pt_np_list else 0
        avg_ln_np_pct = (avg_ln_np - 1.0) * 100.0 if avg_ln_np else 0.0
        print(
            f"{'Average':<12} "
            f"{avg_ln_np:<10.2f}x {avg_ln_np_pct:<10.0f}% "
            f"{avg_ln_pt:<10.2f}x {(avg_ln_pt - 1.0) * 100.0 if avg_ln_pt else 0.0:<10.0f}% "
            f"{avg_pt_np:<10.2f}x {(avg_pt_np - 1.0) * 100.0 if avg_pt_np else 0.0:<10.0f}% "
            f"{avg_ln_np:<12.2f}x {avg_ln_np_pct:<12.0f}%"
        )
        speedup_matrix[op] = avg_ln_np


def print_latency_analysis():
    """Print latency statistics."""
    results = load_benchmark_results()
    
    print("\n\n" + "="*130)
    print("LATENCY ANALYSIS (Total Time for 100 Iterations)")
    print("="*130 + "\n")
    
    operations = ['addition', 'multiplication', 'subtraction', 'division', 'int8_addition', 'int16_addition', 'fp16_addition']
    sizes = [100000, 1000000]
    
    for op in operations:
        print(f"\n{op.upper()}")
        print("-" * 142)
        print(
            f"{'Size':<12} {'LabNeura':<18} {'NumPy':<18} {'PyTorch':<18} "
            f"{'LN/NP x':<10} {'LN/NP %':<10} {'LN/PT x':<10} {'LN/PT %':<10}"
        )
        print("-" * 142)
        
        for size in sizes:
            size_key = str(size)
            if size_key in results and op in results[size_key]:
                ln_time = results[size_key][op]['labneura']['total_time_ms']
                np_time = results[size_key][op]['numpy']['total_time_ms']
                pt_time = results[size_key][op]['pytorch']['total_time_ms']
                ln_np_x, ln_np_pct = format_ratio_columns(ln_time, np_time, higher_is_better=False)
                ln_pt_x, ln_pt_pct = format_ratio_columns(ln_time, pt_time, higher_is_better=False)
                
                print(
                    f"{size:<12,} {ln_time:<18.2f}ms {np_time:<18.2f}ms {pt_time:<18.2f}ms "
                    f"{ln_np_x:<10} {ln_np_pct:<10} {ln_pt_x:<10} {ln_pt_pct:<10}"
                )


def print_throughput_analysis():
    """Print throughput analysis."""
    results = load_benchmark_results()
    
    print("\n\n" + "="*130)
    print("THROUGHPUT ANALYSIS (Elements per Second)")
    print("="*130 + "\n")
    
    operations = ['addition', 'multiplication', 'subtraction', 'division', 'int8_addition']
    sizes = [100000, 1000000]
    
    for op in operations:
        print(f"\n{op.upper()}")
        print("-" * 154)
        print(
            f"{'Size':<12} {'LabNeura':<22} {'NumPy':<22} {'PyTorch':<22} "
            f"{'LN/NP x':<12} {'LN/NP %':<12} {'LN/PT x':<12} {'LN/PT %':<12}"
        )
        print("-" * 154)
        
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
                ln_np_x, ln_np_pct = format_ratio_columns(ln_tput, np_tput, higher_is_better=True)
                ln_pt_x, ln_pt_pct = format_ratio_columns(ln_tput, pt_tput, higher_is_better=True)
                
                print(
                    f"{size:<12,} {ln_str:<22} {np_str:<22} {pt_str:<22} "
                    f"{ln_np_x:<12} {ln_np_pct:<12} {ln_pt_x:<12} {ln_pt_pct:<12}"
                )


if __name__ == '__main__':
    print_summary_table()
    print_speedup_summary()
    print_latency_analysis()
    print_throughput_analysis()
