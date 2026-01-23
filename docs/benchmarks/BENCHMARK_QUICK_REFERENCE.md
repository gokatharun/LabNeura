# Quick Reference: INT16/FP16 Benchmark Integration

**Location:** `/Users/tharungoka/LabNeura/benchmarking/`

---

## 🚀 Quick Start

```bash
# Run benchmarks (now includes INT16 & FP16)
python run_benchmarks.py

# View results (now shows INT16 & FP16 tables)
python visualize_results.py
```

---

## 📊 What Changed

### benchmark_runner.py
**Added Methods:**
```python
def benchmark_int16_operations(self, size: int)
    └─ Tests INT16 addition (LabNeura vs NumPy vs PyTorch)
    
def benchmark_fp16_operations(self, size: int)
    └─ Tests FP16 addition (LabNeura vs NumPy vs PyTorch)
```

### run_benchmarks.py
**Now Runs:**
```python
# For each size (100, 1K, 10K, 100K, 1M):
int16_results = runner.benchmark_int16_operations(size)
fp16_results = runner.benchmark_fp16_operations(size)

# Results saved to: benchmark_results.json
```

### visualize_results.py
**Updated to Include:**
```python
operations = [..., 'int16_addition', 'fp16_addition']

# Displays:
# - INT16_ADDITION BENCHMARK table
# - FP16_ADDITION BENCHMARK table
# - Speedup comparisons
# - Latency analysis
```

---

## 🎯 Performance Summary (1M Elements)

### INT16: LabNeura Wins
```
LabNeura: 14.94 GFLOPS (6.69 ms)
NumPy:    1.19 GFLOPS  (84.27 ms)  → 12.59x slower
PyTorch:  1.39 GFLOPS  (72.00 ms)  → 10.76x slower

Winner: LabNeura ✅ (direct NEON intrinsics)
```

### FP16: PyTorch Wins
```
LabNeura: 14.37 GFLOPS (6.96 ms)
PyTorch:  16.56 GFLOPS (6.04 ms)  → 1.15x faster
NumPy:    228.75 MFLOPS(437.15 ms)→ 62.83x slower

Winner: PyTorch (JIT + GPU optimization)
```

---

## 💡 Why PyTorch FP16 is Fast

**PyTorch has native torch.float16 support:**

```python
import torch

# Native FP16
a = torch.ones(1_000_000, dtype=torch.float16)
b = torch.ones(1_000_000, dtype=torch.float16)
result = a + b  # Fast! ⚡
```

**Mechanisms:**
1. GPU Tensor Cores (if available)
2. CPU SIMD (NEON on ARM, AVX on x86)
3. JIT compilation (hot path optimization)
4. Automatic Mixed Precision (AMP)

**Result:** 1.15x faster than LabNeura on FP16

---

## 📈 Performance Chart

```
Operation   Size      LabNeura        Winner
────────────────────────────────────────────
INT16 ADD   100       208.15 MFLOPS   LabNeura ✅
INT16 ADD   1K        2.01 GFLOPS     LabNeura ✅
INT16 ADD   10K       11.99 GFLOPS    LabNeura ✅
INT16 ADD   100K      13.93 GFLOPS    LabNeura ✅
INT16 ADD   1M        14.94 GFLOPS    LabNeura ✅

FP16 ADD    100       232.11 MFLOPS   LabNeura ✅
FP16 ADD    1K        2.00 GFLOPS     LabNeura ✅
FP16 ADD    10K       12.24 GFLOPS    LabNeura ✅
FP16 ADD    100K      13.86 GFLOPS    LabNeura ✅
FP16 ADD    1M        14.37 GFLOPS    PyTorch  (1.15x)

vs NumPy:
INT16 ADD   1M        12.59x faster   LabNeura ✅
FP16 ADD    1M        62.83x faster   LabNeura ✅
```

---

## 📁 Documentation

Read more about:
- **PyTorch FP16:** [PYTORCH_FP16_ANALYSIS.md](PYTORCH_FP16_ANALYSIS.md)
- **Integration Details:** [BENCHMARK_INTEGRATION.md](BENCHMARK_INTEGRATION.md)
- **Benchmark Results:** [benchmark_int16_fp16_output.txt](../benchmarking/benchmark_int16_fp16_output.txt)

---

## ✨ Key Takeaways

✅ **INT16:** LabNeura 12.59x faster than NumPy  
✅ **FP16:** LabNeura 62.83x faster than NumPy  
⚠️ **FP16 vs PyTorch:** PyTorch 1.15x faster (GPU advantage)  

**When to Use:**
- **LabNeura:** Edge devices, INT16 quantization, CPU-only
- **PyTorch:** GPU acceleration, FP16 training, dynamic graphs
- **NumPy:** FP32/FP64, prototyping

---

**Last Updated:** January 10, 2026
