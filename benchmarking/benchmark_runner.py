"""
Benchmark Runner

Runs comprehensive benchmarks comparing LabNeura, NumPy, and PyTorch.
"""

import time
import numpy as np
import torch
import labneura
from typing import Dict, List, Callable, Any, Tuple
from .metrics_calculator import MetricsCalculator, PerformanceMetrics


class BenchmarkRunner:
    """Run and manage benchmarks."""
    
    def __init__(self, warmup_iterations: int = 5, benchmark_iterations: int = 100):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.results = {}
    
    def run_operation_benchmark(
        self,
        operation_name: str,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> PerformanceMetrics:
        """
        Run a benchmark for an operation.
        
        Args:
            operation_name: Name of the operation
            operation_func: Function to benchmark
            *args: Arguments to pass to operation
            **kwargs: Keyword arguments
            
        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()
        
        # Warmup
        for _ in range(self.warmup_iterations):
            operation_func(*args, **kwargs)
        
        # Benchmark
        times = []
        start = time.perf_counter()
        
        for _ in range(self.benchmark_iterations):
            iter_start = time.perf_counter()
            operation_func(*args, **kwargs)
            iter_end = time.perf_counter()
            times.append(iter_end - iter_start)
        
        end = time.perf_counter()
        total_time = end - start
        
        metrics.total_time = total_time
        metrics.latency_stats = MetricsCalculator.calculate_latency_stats(times)
        
        return metrics
    
    def benchmark_addition(self, size: int) -> Dict[str, PerformanceMetrics]:
        """Benchmark addition operation."""
        results = {}
        
        # LabNeura
        a_ln = labneura.Tensor([1.0] * size)
        b_ln = labneura.Tensor([1.0] * size)
        
        def ln_add():
            a_ln.add_inplace(b_ln)
        
        results['labneura'] = self.run_operation_benchmark('add', ln_add)
        num_ops = size  # One addition per element
        results['labneura'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['labneura'].total_time
        )
        results['labneura'].flops_str = MetricsCalculator.format_flops(results['labneura'].flops)
        
        # NumPy
        a_np = np.ones(size, dtype=np.float32)
        b_np = np.ones(size, dtype=np.float32)
        
        def np_add():
            return a_np + b_np
        
        results['numpy'] = self.run_operation_benchmark('add', np_add)
        results['numpy'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['numpy'].total_time
        )
        results['numpy'].flops_str = MetricsCalculator.format_flops(results['numpy'].flops)
        
        # PyTorch
        a_pt = torch.ones(size, dtype=torch.float32)
        b_pt = torch.ones(size, dtype=torch.float32)
        
        def pt_add():
            return a_pt + b_pt
        
        results['pytorch'] = self.run_operation_benchmark('add', pt_add)
        results['pytorch'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['pytorch'].total_time
        )
        results['pytorch'].flops_str = MetricsCalculator.format_flops(results['pytorch'].flops)
        
        return results
    
    def benchmark_multiplication(self, size: int) -> Dict[str, PerformanceMetrics]:
        """Benchmark multiplication operation."""
        results = {}
        
        # LabNeura
        a_ln = labneura.Tensor([2.0] * size)
        b_ln = labneura.Tensor([3.0] * size)
        
        def ln_mul():
            a_ln.mul_inplace(b_ln)
        
        results['labneura'] = self.run_operation_benchmark('mul', ln_mul)
        num_ops = size
        results['labneura'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['labneura'].total_time
        )
        results['labneura'].flops_str = MetricsCalculator.format_flops(results['labneura'].flops)
        
        # NumPy
        a_np = np.full(size, 2.0, dtype=np.float32)
        b_np = np.full(size, 3.0, dtype=np.float32)
        
        def np_mul():
            return a_np * b_np
        
        results['numpy'] = self.run_operation_benchmark('mul', np_mul)
        results['numpy'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['numpy'].total_time
        )
        results['numpy'].flops_str = MetricsCalculator.format_flops(results['numpy'].flops)
        
        # PyTorch
        a_pt = torch.full((size,), 2.0, dtype=torch.float32)
        b_pt = torch.full((size,), 3.0, dtype=torch.float32)
        
        def pt_mul():
            return a_pt * b_pt
        
        results['pytorch'] = self.run_operation_benchmark('mul', pt_mul)
        results['pytorch'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['pytorch'].total_time
        )
        results['pytorch'].flops_str = MetricsCalculator.format_flops(results['pytorch'].flops)
        
        return results
    
    def benchmark_subtraction(self, size: int) -> Dict[str, PerformanceMetrics]:
        """Benchmark subtraction operation."""
        results = {}
        
        # LabNeura
        a_ln = labneura.Tensor([10.0] * size)
        b_ln = labneura.Tensor([3.0] * size)
        
        def ln_sub():
            a_ln.sub_inplace(b_ln)
        
        results['labneura'] = self.run_operation_benchmark('sub', ln_sub)
        num_ops = size
        results['labneura'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['labneura'].total_time
        )
        results['labneura'].flops_str = MetricsCalculator.format_flops(results['labneura'].flops)
        
        # NumPy
        a_np = np.full(size, 10.0, dtype=np.float32)
        b_np = np.full(size, 3.0, dtype=np.float32)
        
        def np_sub():
            return a_np - b_np
        
        results['numpy'] = self.run_operation_benchmark('sub', np_sub)
        results['numpy'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['numpy'].total_time
        )
        results['numpy'].flops_str = MetricsCalculator.format_flops(results['numpy'].flops)
        
        # PyTorch
        a_pt = torch.full((size,), 10.0, dtype=torch.float32)
        b_pt = torch.full((size,), 3.0, dtype=torch.float32)
        
        def pt_sub():
            return a_pt - b_pt
        
        results['pytorch'] = self.run_operation_benchmark('sub', pt_sub)
        results['pytorch'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['pytorch'].total_time
        )
        results['pytorch'].flops_str = MetricsCalculator.format_flops(results['pytorch'].flops)
        
        return results

    def benchmark_division(self, size: int) -> Dict[str, PerformanceMetrics]:
        """Benchmark division operation."""
        results = {}
        
        # LabNeura (in-place for fair comparison with add/mul/sub)
        a_ln = labneura.Tensor([10.0] * size)
        b_ln = labneura.Tensor([2.0] * size)
        def ln_div():
            a_ln.div_inplace(b_ln)
        results['labneura'] = self.run_operation_benchmark('div', ln_div)
        num_ops = size
        results['labneura'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['labneura'].total_time
        )
        results['labneura'].flops_str = MetricsCalculator.format_flops(results['labneura'].flops)
        
        # NumPy
        a_np = np.full(size, 10.0, dtype=np.float32)
        b_np = np.full(size, 2.0, dtype=np.float32)
        def np_div():
            return a_np / b_np
        results['numpy'] = self.run_operation_benchmark('div', np_div)
        results['numpy'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['numpy'].total_time
        )
        results['numpy'].flops_str = MetricsCalculator.format_flops(results['numpy'].flops)
        
        # PyTorch
        a_pt = torch.full((size,), 10.0, dtype=torch.float32)
        b_pt = torch.full((size,), 2.0, dtype=torch.float32)
        def pt_div():
            return a_pt / b_pt
        results['pytorch'] = self.run_operation_benchmark('div', pt_div)
        results['pytorch'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['pytorch'].total_time
        )
        results['pytorch'].flops_str = MetricsCalculator.format_flops(results['pytorch'].flops)
        
        return results
    
    def benchmark_int8_operations(self, size: int) -> Dict[str, Dict[str, PerformanceMetrics]]:
        """Benchmark INT8 operations with proper saturation."""
        results = {}
        
        # LabNeura INT8 ADD (in-place with saturation)
        a_ln = labneura.Tensor([10] * size, labneura.QuantizationMode.INT8)
        b_ln = labneura.Tensor([20] * size, labneura.QuantizationMode.INT8)
        
        def ln_int8_add():
            a_ln.add_inplace(b_ln)
        
        results['add'] = {}
        results['add']['labneura'] = self.run_operation_benchmark('int8_add', ln_int8_add)
        num_ops = size
        results['add']['labneura'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['add']['labneura'].total_time
        )
        results['add']['labneura'].flops_str = MetricsCalculator.format_flops(
            results['add']['labneura'].flops
        )
        
        # NumPy INT8 ADD (with saturation for fair comparison)
        a_np = np.full(size, 10, dtype=np.int8)
        b_np = np.full(size, 20, dtype=np.int8)
        
        def np_int8_add():
            # Proper INT8 saturation arithmetic
            return np.clip(a_np.astype(np.int16) + b_np.astype(np.int16), -128, 127).astype(np.int8)
        
        results['add']['numpy'] = self.run_operation_benchmark('int8_add', np_int8_add)
        results['add']['numpy'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['add']['numpy'].total_time
        )
        results['add']['numpy'].flops_str = MetricsCalculator.format_flops(results['add']['numpy'].flops)
        
        # PyTorch INT8 ADD (with saturation for fair comparison)
        a_pt = torch.full((size,), 10, dtype=torch.int8)
        b_pt = torch.full((size,), 20, dtype=torch.int8)
        
        def pt_int8_add():
            # Proper INT8 saturation arithmetic
            return torch.clamp(a_pt.to(torch.int32) + b_pt.to(torch.int32), -128, 127).to(torch.int8)
        
        results['add']['pytorch'] = self.run_operation_benchmark('int8_add', pt_int8_add)
        results['add']['pytorch'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['add']['pytorch'].total_time
        )
        results['add']['pytorch'].flops_str = MetricsCalculator.format_flops(results['add']['pytorch'].flops)
        
        return results
    
    def benchmark_int16_operations(self, size: int) -> Dict[str, Dict[str, PerformanceMetrics]]:
        """Benchmark INT16 operations with proper saturation."""
        results = {}
        
        # LabNeura INT16 ADD
        a_ln = labneura.Tensor([100] * size, labneura.QuantizationMode.INT16)
        b_ln = labneura.Tensor([200] * size, labneura.QuantizationMode.INT16)
        
        def ln_int16_add():
            a_ln.add_inplace(b_ln)
        
        results['add'] = {}
        results['add']['labneura'] = self.run_operation_benchmark('int16_add', ln_int16_add)
        num_ops = size
        results['add']['labneura'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['add']['labneura'].total_time
        )
        results['add']['labneura'].flops_str = MetricsCalculator.format_flops(
            results['add']['labneura'].flops
        )
        
        # NumPy INT16 ADD (with saturation)
        a_np = np.full(size, 100, dtype=np.int16)
        b_np = np.full(size, 200, dtype=np.int16)
        
        def np_int16_add():
            return np.clip(a_np.astype(np.int32) + b_np.astype(np.int32), -32768, 32767).astype(np.int16)
        
        results['add']['numpy'] = self.run_operation_benchmark('int16_add', np_int16_add)
        results['add']['numpy'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['add']['numpy'].total_time
        )
        results['add']['numpy'].flops_str = MetricsCalculator.format_flops(results['add']['numpy'].flops)
        
        # PyTorch INT16 ADD (with saturation)
        a_pt = torch.full((size,), 100, dtype=torch.int16)
        b_pt = torch.full((size,), 200, dtype=torch.int16)
        
        def pt_int16_add():
            return torch.clamp(a_pt.to(torch.int32) + b_pt.to(torch.int32), -32768, 32767).to(torch.int16)
        
        results['add']['pytorch'] = self.run_operation_benchmark('int16_add', pt_int16_add)
        results['add']['pytorch'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['add']['pytorch'].total_time
        )
        results['add']['pytorch'].flops_str = MetricsCalculator.format_flops(results['add']['pytorch'].flops)
        
        return results
    
    def benchmark_fp16_operations(self, size: int) -> Dict[str, Dict[str, PerformanceMetrics]]:
        """Benchmark FP16 operations."""
        results = {}
        
        # LabNeura FP16 ADD
        a_ln = labneura.Tensor([1.5] * size, labneura.QuantizationMode.FP16)
        b_ln = labneura.Tensor([2.5] * size, labneura.QuantizationMode.FP16)
        
        def ln_fp16_add():
            a_ln.add_inplace(b_ln)
        
        results['add'] = {}
        results['add']['labneura'] = self.run_operation_benchmark('fp16_add', ln_fp16_add)
        num_ops = size
        results['add']['labneura'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['add']['labneura'].total_time
        )
        results['add']['labneura'].flops_str = MetricsCalculator.format_flops(
            results['add']['labneura'].flops
        )
        
        # NumPy FP16 ADD
        a_np = np.full(size, 1.5, dtype=np.float16)
        b_np = np.full(size, 2.5, dtype=np.float16)
        
        def np_fp16_add():
            return a_np + b_np
        
        results['add']['numpy'] = self.run_operation_benchmark('fp16_add', np_fp16_add)
        results['add']['numpy'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['add']['numpy'].total_time
        )
        results['add']['numpy'].flops_str = MetricsCalculator.format_flops(results['add']['numpy'].flops)
        
        # PyTorch FP16 ADD (using torch.float16 / torch.half)
        a_pt = torch.full((size,), 1.5, dtype=torch.float16)
        b_pt = torch.full((size,), 2.5, dtype=torch.float16)
        
        def pt_fp16_add():
            return a_pt + b_pt
        
        results['add']['pytorch'] = self.run_operation_benchmark('fp16_add', pt_fp16_add)
        results['add']['pytorch'].flops = MetricsCalculator.calculate_flops(
            num_ops * self.benchmark_iterations,
            results['add']['pytorch'].total_time
        )
        results['add']['pytorch'].flops_str = MetricsCalculator.format_flops(results['add']['pytorch'].flops)
        
        return results

    def get_comparison_speedup(
        self,
        results: Dict[str, PerformanceMetrics],
        baseline: str = 'numpy'
    ) -> Dict[str, float]:
        """
        Calculate speedup relative to baseline.
        
        Args:
            results: Dictionary of results
            baseline: Baseline library name
            
        Returns:
            Dictionary of speedup factors
        """
        speedups = {}
        baseline_time = results[baseline].total_time
        
        for lib, metrics in results.items():
            speedups[lib] = MetricsCalculator.calculate_speedup(
                baseline_time,
                metrics.total_time
            )
        
        return speedups
