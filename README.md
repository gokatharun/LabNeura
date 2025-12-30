# LabNeura

High-performance C++ tensor library with Python bindings, optimized for Apple Silicon (M1/M2/M3) and modern CPUs. Provides SIMD-accelerated operations with optional INT8 quantization support.

## Features

- **Architecture-aware optimization**: Automatic NEON vectorization on Apple Silicon, generic fallback for other platforms
- **Pure SIMD acceleration**: High-performance vector operations without multi-threading overhead
- **Mixed-precision support**: FP32 (32-bit float) and INT8 (8-bit quantized) modes
- **Python integration**: Easy-to-use Python bindings via pybind11
- **Header-only C++ library**: Simple integration into C++ projects

## Quick Start

### As a C++ Header-Only Library

Include headers from `include/` in your project:

```cpp
#include "labneura/tensor.h"

labneura::Tensor t1(std::vector<float>{1.0, 2.0, 3.0});
labneura::Tensor t2(std::vector<float>{1.0, 1.0, 1.0});
t1.add_inplace(t2);
```

### Build & Install

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
cmake --install .
```

### Python Extension

```bash
cd python
pip install .
```

```python
import labneura

# FP32 operations
t1 = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP32)
t2 = labneura.Tensor([1.0, 1.0, 1.0], labneura.QuantizationMode.FP32)
t1.add_inplace(t2)
data = t1.data_fp32()

# INT8 quantization
t3 = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT8)
```

## Performance

LabNeura delivers **SIMD-accelerated performance** exceeding NumPy on M1 Macs through pure vectorization:

- **NEON vectorization**: 8x float (2 instructions) or 16x int8 operations per iteration
- **Memory efficiency**: INT8 mode uses 4x less memory, improving cache utilization
- **No threading overhead**: Single-threaded SIMD approach maximizes per-core performance

Run benchmarks:

```bash
python benchmark_numpy.py
```

Expected results on M1/M2:
- FP32 addition: 2000-3000 M ops/sec
- INT8 addition: 4000-6000 M ops/sec (4x throughput due to wider vectors)
- Competitive or faster than NumPy across array sizes

## Implementation Details

### NEON Optimization (Apple Silicon)
- Detects M1/M2/M3 at compile-time via `__aarch64__` and `__APPLE__`
- Uses ARM NEON intrinsics for 128-bit vector operations
- Processes 8 floats (2 instructions) or 16 int8 values per iteration
- Single-threaded design maximizes instruction-level parallelism

### Quantization Support
- **FP32 mode**: Full precision for scientific computing
- **INT8 mode**: 4x memory reduction, suitable for inference workloads

## Repository Layout

- `include/labneura/` — C++ headers (tensor.h, types.hpp)
- `src/labneura/` — Implementation (tensor.cpp)
- `examples/` — C++ usage examples
- `python/` — Python bindings (pybind11)
- `CMakeLists.txt` — Build configuration
- `benchmark_numpy.py` — Performance comparison with NumPy
- `PERFORMANCE_ANALYSIS.md` — Detailed performance analysis

## License

See LICENSE file for details.
