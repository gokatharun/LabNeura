#!/usr/bin/env python3
"""
Main Benchmarking Script

Runs comprehensive benchmarks and exports results for visualization/CSV.
Usage: python benchmarking/run_benchmarks.py
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import sys
# Add parent directory to path so `benchmarking` package is importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.benchmark_runner import BenchmarkRunner, PerformanceMetrics
from benchmarking.metrics_calculator import MetricsCalculator


def metrics_to_dict(metrics: PerformanceMetrics, size: int, iterations: int) -> Dict:
    """Convert PerformanceMetrics to a JSON-serializable dictionary."""
    throughput = MetricsCalculator.calculate_throughput(size, iterations, metrics.total_time)
    return {
        "flops": metrics.flops,
        "flops_str": metrics.flops_str,
        "total_time": metrics.total_time,
        "total_time_ms": metrics.total_time * 1000.0,
        "latency_stats": metrics.latency_stats,
        "throughput": throughput,
        "iterations": iterations,
    }


def print_op_summary(op_name: str, results: Dict[str, PerformanceMetrics], runner: BenchmarkRunner):
    """Pretty-print per-operation benchmark summary."""
    print(f"\n{op_name} Benchmark...")
    for lib in ("labneura", "numpy", "pytorch"):
        metrics = results[lib]
        print(
            f"  {lib.title():<8}: {metrics.flops_str} | "
            f"Time: {metrics.total_time * 1000:.2f}ms"
        )
    speedups = runner.get_comparison_speedup(results, baseline="numpy")
    print("  Speedup vs NumPy:")
    print(f"    LabNeura: {speedups['labneura']:.2f}x")
    print(f"    PyTorch:  {speedups['pytorch']:.2f}x")


def run_comprehensive_benchmarks() -> Dict[str, Dict]:
    """Run benchmarks across all tensor sizes and operations."""
    print("=" * 80)
    print("LabNeura Comprehensive Benchmarking Suite")
    print("=" * 80)
    print()

    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
    runner = BenchmarkRunner(warmup_iterations=5, benchmark_iterations=100)
    all_results: Dict[str, Dict] = {}

    print("Running benchmarks...")
    print()

    operations: Tuple[Tuple[str, callable], ...] = (
        ("addition", runner.benchmark_addition),
        ("multiplication", runner.benchmark_multiplication),
        ("subtraction", runner.benchmark_subtraction),
        ("division", runner.benchmark_division),
    )

    for size in sizes:
        size_key = str(size)
        print(f"\n{'='*80}")
        print(f"Size: {size:,} elements")
        print(f"{'='*80}")

        all_results[size_key] = {}

        for op_name, op_func in operations:
            op_results = op_func(size)
            print_op_summary(op_name.title(), op_results, runner)
            all_results[size_key][op_name] = {
                lib: metrics_to_dict(metrics, size, runner.benchmark_iterations)
                for lib, metrics in op_results.items()
            }

        # INT8
        int8_results = runner.benchmark_int8_operations(size)["add"]
        print_op_summary("INT8 Addition", int8_results, runner)
        all_results[size_key]["int8_addition"] = {
            lib: metrics_to_dict(metrics, size, runner.benchmark_iterations)
            for lib, metrics in int8_results.items()
        }

        # INT16
        int16_results = runner.benchmark_int16_operations(size)["add"]
        print_op_summary("INT16 Addition", int16_results, runner)
        all_results[size_key]["int16_addition"] = {
            lib: metrics_to_dict(metrics, size, runner.benchmark_iterations)
            for lib, metrics in int16_results.items()
        }

        # FP16
        fp16_results = runner.benchmark_fp16_operations(size)["add"]
        print_op_summary("FP16 Addition", fp16_results, runner)
        all_results[size_key]["fp16_addition"] = {
            lib: metrics_to_dict(metrics, size, runner.benchmark_iterations)
            for lib, metrics in fp16_results.items()
        }

    print(f"\n\n{'='*80}")
    print("Benchmark Complete!")
    print(f"{'='*80}\n")

    return all_results


def export_results(results: Dict[str, Dict], output_file: str = "benchmark_results.json") -> Path:
    """Write results to a JSON file in the benchmarking directory."""
    output_path = Path(__file__).parent / output_file
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Results exported to {output_path}")
    return output_path


if __name__ == "__main__":
    results = run_comprehensive_benchmarks()
    export_results(results)
