"""
Metrics Calculator for Benchmarking

Calculates FLOPS, memory usage, and other performance metrics.
"""

import time
import psutil
import numpy as np
from typing import Dict, Tuple


class MetricsCalculator:
    """Calculate performance metrics from benchmarks."""
    
    @staticmethod
    def calculate_flops(num_operations: int, time_seconds: float) -> float:
        """
        Calculate floating-point operations per second (FLOPS).
        
        Args:
            num_operations: Number of floating-point operations
            time_seconds: Time taken in seconds
            
        Returns:
            FLOPS value
        """
        if time_seconds <= 0:
            return 0
        return num_operations / time_seconds
    
    @staticmethod
    def format_flops(flops: float) -> str:
        """Format FLOPS into human-readable format."""
        if flops >= 1e9:
            return f"{flops / 1e9:.2f} GFLOPS"
        elif flops >= 1e6:
            return f"{flops / 1e6:.2f} MFLOPS"
        else:
            return f"{flops:.2f} FLOPS"
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        Get current memory usage in MB.
        
        Returns:
            Dictionary with memory metrics
        """
        process = psutil.Process()
        mem_info = process.memory_info()
        
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        }
    
    @staticmethod
    def calculate_memory_bandwidth(
        data_size_mb: float, 
        time_seconds: float
    ) -> Tuple[float, str]:
        """
        Calculate memory bandwidth in GB/s.
        
        Args:
            data_size_mb: Data size in MB
            time_seconds: Time taken in seconds
            
        Returns:
            (bandwidth in GB/s, formatted string)
        """
        if time_seconds <= 0:
            return 0, "0 GB/s"
        
        bandwidth = (data_size_mb / 1024) / time_seconds  # GB/s
        return bandwidth, f"{bandwidth:.2f} GB/s"
    
    @staticmethod
    def calculate_latency_stats(times: list) -> Dict[str, float]:
        """
        Calculate latency statistics from list of times.
        
        Args:
            times: List of execution times in seconds
            
        Returns:
            Dictionary with latency stats
        """
        times_ms = [t * 1000 for t in times]  # Convert to ms
        
        return {
            'min_ms': np.min(times_ms),
            'max_ms': np.max(times_ms),
            'mean_ms': np.mean(times_ms),
            'median_ms': np.median(times_ms),
            'std_ms': np.std(times_ms),
            'p95_ms': np.percentile(times_ms, 95),
            'p99_ms': np.percentile(times_ms, 99),
        }
    
    @staticmethod
    def calculate_throughput(
        num_samples: int,
        num_iterations: int,
        time_seconds: float
    ) -> float:
        """
        Calculate throughput in samples/second.
        
        Args:
            num_samples: Number of samples processed
            num_iterations: Number of iterations
            time_seconds: Time taken in seconds
            
        Returns:
            Throughput in samples/second
        """
        if time_seconds <= 0:
            return 0
        return (num_samples * num_iterations) / time_seconds
    
    @staticmethod
    def calculate_speedup(baseline_time: float, new_time: float) -> float:
        """
        Calculate speedup relative to baseline.
        
        Args:
            baseline_time: Baseline time
            new_time: New time
            
        Returns:
            Speedup factor
        """
        if new_time <= 0:
            return 0
        return baseline_time / new_time
    
    @staticmethod
    def calculate_efficiency(
        theoretical_peak_flops: float,
        actual_flops: float
    ) -> float:
        """
        Calculate computational efficiency as percentage.
        
        Args:
            theoretical_peak_flops: Peak FLOPS capability
            actual_flops: Actual FLOPS achieved
            
        Returns:
            Efficiency percentage (0-100)
        """
        if theoretical_peak_flops <= 0:
            return 0
        return (actual_flops / theoretical_peak_flops) * 100


class PerformanceMetrics:
    """Container for performance metrics."""
    
    def __init__(self):
        self.flops = 0
        self.flops_str = "0 FLOPS"
        self.memory_mb = 0
        self.memory_bandwidth_gbs = 0
        self.latency_stats = {}
        self.throughput = 0
        self.total_time = 0
    
    def __repr__(self):
        return (
            f"PerformanceMetrics(\n"
            f"  FLOPS: {self.flops_str}\n"
            f"  Memory: {self.memory_mb:.2f} MB\n"
            f"  Bandwidth: {self.memory_bandwidth_gbs:.2f} GB/s\n"
            f"  Throughput: {self.throughput:.2f} samples/s\n"
            f"  Time: {self.total_time*1000:.2f} ms\n"
            f")"
        )
