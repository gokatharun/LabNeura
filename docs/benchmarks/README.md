# LabNeura Benchmarking Documentation

This directory contains comprehensive performance benchmarks comparing LabNeura quantization operations against NumPy and PyTorch.

## � Quick Start

**New to these benchmarks?** → **[START HERE](START_HERE.md)** ← 30-second overview + role-based navigation

---

## �📊 Benchmark Reports

### Latest: INT16 & FP16 Quantization
- **File:** [INT16_FP16_BENCHMARK_REPORT.md](INT16_FP16_BENCHMARK_REPORT.md)
- **Focus:** Comparing LabNeura's NEON-optimized INT16/FP16 ops vs NumPy and PyTorch
- **Key Finding:** LabNeura is **5-6x faster** for INT16 and **60-135x faster** for FP16 than NumPy at scale
- **Platform:** ARM64 (Apple Silicon)
- **Operations:** ADD, SUB, MUL at sizes 1K to 1M elements

### INT8 Quantization (Legacy)
- **File:** [BENCHMARKING_REPORT_INT8_OPTIMIZED.md](BENCHMARKING_REPORT_INT8_OPTIMIZED.md)
- **Focus:** INT8 quantization performance with hierarchical SIMD vectorization
- **Achievement:** 95% throughput of reference implementation on low-power mobile devices
- **Details:** [INT8_ACHIEVEMENT_SUMMARY.md](INT8_ACHIEVEMENT_SUMMARY.md)

### Detailed Results & Analysis
- [INT8_BENCHMARK_RESULTS.md](INT8_BENCHMARK_RESULTS.md) - Raw INT8 data
- [PERFORMANCE_SUMMARY.md](PERFORMANCE_SUMMARY.md) - Executive summary of all benchmarks
- [PERFORMANCE_COMPARISON_DUAL_BASELINE.md](PERFORMANCE_COMPARISON_DUAL_BASELINE.md) - Two-baseline comparison framework

---

## 🎯 Quick Performance Summary

### INT16 Operations (1M elements)
| Operation | LabNeura | NumPy | Speedup |
|-----------|----------|-------|---------|
| ADD | 67.98 μs | 414.38 μs | **6.1x** |
| SUB | 68.61 μs | 412.95 μs | **6.0x** |
| MUL | 71.49 μs | 417.14 μs | **5.8x** |

### FP16 Operations (1M elements)
| Operation | LabNeura | NumPy | Speedup |
|-----------|----------|-------|---------|
| ADD | 67.94 μs | 4102.61 μs | **60.4x** |
| SUB | 69.85 μs | 4192.53 μs | **60.0x** |
| MUL | 67.94 μs | 9155.19 μs | **134.8x** |

---

## 🔧 Benchmark Methodology

All benchmarks follow this standardized approach:

1. **Size Range:** 1K, 10K, 100K, 1M elements
2. **Iterations:** 50 per configuration (cold-start, no warmup)
3. **Timing Precision:** nanosecond-level (`time.perf_counter()`)
4. **Libraries Tested:**
   - LabNeura (custom NEON backend)
   - NumPy (1.26.x)
   - PyTorch (2.x)

### Important Notes
- ⚠️ **Speedup ratios <1.0 indicate lower latency for the comparator** (Python/tensor overhead masks NEON gains at small sizes)
- Platform: macOS with Apple Silicon (ARM64 NEON)
- Results are **latency-focused** (not throughput or batch processing)

---

## 📈 Key Insights

### When LabNeura Excels
✅ **Embedded/Mobile Deployment** - No GPU; consistent low-power inference  
✅ **Integer Quantization** - 5-6x faster INT16 than NumPy  
✅ **Real-time Systems** - Predictable latency, SIMD-guaranteed execution  
✅ **Edge Inference** - FP16 handling 60-135x faster than NumPy  

### When Comparators Excel
⚠️ **GPU Acceleration** - PyTorch Metal/CUDA (1.18-1.24x faster FP16)  
⚠️ **FP32/FP64 Computing** - NumPy's primary strength  
⚠️ **Dynamic Graphs** - PyTorch flexibility for recurrent ops  

---

## 🚀 Running Benchmarks

### INT16 & FP16 Benchmarks
```bash
cd /Users/tharungoka/LabNeura
python benchmarking/benchmark_int16_fp16.py
```

Output saved to:
- `benchmarking/benchmark_int16_fp16_output.txt` (human-readable)
- `benchmarking/benchmark_int16_fp16_results.json` (machine-parseable)

---

## 📚 Architecture Details

### NEON Hierarchical Processing (INT16/FP16)
- **64-element chunks:** 8x 8-element SIMD registers
- **Tail sizes:** 32→16→8→4→scalar (hierarchical chaining)
- **Intrinsics:** `vqaddq_s16`, `vqsubq_s16`, `vmull_s16 + vqmovn_s32`
- **Prefetch:** 128-element distance for cache optimization

### Quantization Strategy
- **Range:** [-32768, 32767] for INT16
- **Saturation:** Hardware-enforced (no branch stalls)
- **Zero-point support:** Configurable per tensor

---

## 🔗 Related Files

- **Source Code:**
  - `include/labneura/types.hpp` - Quantize/dequantize helpers
  - `python/src/labneura/backends/neon.cpp` - NEON kernel implementation
  - `python/src/labneura/tensor.cpp` - Tensor operations

- **Tests:**
  - `tests/test_tensor_int16_ops.py` - 18 INT16 operation tests
  - `tests/test_tensor.py` - Core tensor tests

- **Benchmark Scripts:**
  - `benchmarking/benchmark_int16_fp16.py` - INT16/FP16 vs NumPy/PyTorch
  - `benchmarking/benchmark_runner.py` - Generic benchmark framework
  - `benchmarking/metrics_calculator.py` - Performance metric utilities

---

## 📝 License

All benchmark data and documentation released under the [LabNeura License](../../LICENSE).

---

**Last Updated:** January 10, 2026  
**Maintainer:** LabNeura Development Team
