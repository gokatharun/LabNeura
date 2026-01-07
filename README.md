# LabNeura

[![PyPI version](https://badge.fury.io/py/labneura2.svg)](https://badge.fury.io/py/labneura2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance C++ tensor library with Python bindings, featuring **hardware-accelerated SIMD operations** optimized for modern CPUs. Supports AVX2 (x86_64), NEON (ARM64/Apple Silicon), and provides mixed-precision computation with optional INT8 quantization.

## 🚀 Features

### Hardware Acceleration
- **Multi-backend Architecture**: Runtime detection and selection of optimal SIMD backend
  - **AVX2 Backend**: 8-wide FP32 or 32-wide INT8 vectorization on Intel/AMD x86_64 CPUs
  - **NEON Backend**: 4-wide FP32 or 16-wide INT8 vectorization on Apple Silicon (M1/M2/M3/M4) and ARM64
  - **Generic Backend**: Portable fallback for all architectures
- **Automatic Backend Selection**: Detects CPU capabilities at runtime using `detect_backend()`
- **Compile-time Optimization**: CMake detects hardware and enables appropriate compiler flags

### Tensor Operations
- **Mixed-Precision Support**: FP32 (32-bit float), INT8 (8-bit quantized), with future support for FP64, FP16, INT32, INT16
- **In-place Operations**: `add_inplace()`, `mul_inplace()`, `sub_inplace()` for memory efficiency
- **Immutable Operations**: `add()` returns new tensors without modifying originals
- **Quantization**: Built-in INT8 quantization with 4x memory reduction

### Python Integration
- **Easy Installation**: `pip install labneura2` (PyPI package name, import as `labneura`)
- **pybind11 Bindings**: Zero-copy data access between C++ and Python
- **NumPy Compatible**: Seamless integration with NumPy workflows

## 📦 Installation

### Python Package (Recommended)

```bash
pip install labneura2
```

**Note**: The PyPI package is named `labneura2`, but you import it as `labneura`:
```python
import labneura  # Import statement
```

### Build from Source

#### Prerequisites
- CMake >= 3.14
- C++17 compatible compiler (Clang, GCC, MSVC)
- Python 3.9+ (for Python bindings)
- pybind11 >= 3.0.1

#### C++ Library
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cmake --install .
```

#### Python Bindings
```bash
cd python
pip install .
```

For development with coverage:
```bash
LABNEURA_COVERAGE=1 pip install -e .
```

## 🎯 Quick Start

### Python Usage

```python
import labneura

# Detect available backend
print(f"Using backend: {labneura.detect_backend()}")  # "AVX2", "NEON", or "GENERIC"

# FP32 operations
t1 = labneura.Tensor([1.0, 2.0, 3.0, 4.0], labneura.QuantizationMode.FP32)
t2 = labneura.Tensor([0.5, 0.5, 0.5, 0.5], labneura.QuantizationMode.FP32)

# In-place addition (modifies t1)
t1.add_inplace(t2)
print(t1.data_fp32())  # [1.5, 2.5, 3.5, 4.5]

# Non-destructive addition (returns new tensor)
t3 = t1.add(t2)

# Other operations
t1.mul_inplace(t2)  # Element-wise multiplication
t1.sub_inplace(t2)  # Element-wise subtraction

# INT8 quantization for memory efficiency
t_int8 = labneura.Tensor([10, 20, 30, 40], labneura.QuantizationMode.INT8)
print(t_int8.data_int8())  # [10, 20, 30, 40]
```

### C++ Usage

```cpp
#include "labneura/tensor.h"
#include "labneura/backends/backend_factory.h"
#include <iostream>

int main() {
    // Detect backend at runtime
    std::cout << "Backend: " << labneura::detect_backend() << std::endl;
    
    // Create tensors
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data2 = {0.5f, 0.5f, 0.5f, 0.5f};
    
    labneura::Tensor t1(data1, labneura::QuantizationMode::FP32);
    labneura::Tensor t2(data2, labneura::QuantizationMode::FP32);
    
    // In-place operations
    t1.add_inplace(t2);
    
    // Access data
    const float* result = t1.data_fp32();
    for (size_t i = 0; i < t1.size(); ++i) {
        std::cout << result[i] << " ";
    }
    
    return 0;
}
```

Compile:
```bash
g++ -std=c++17 -O3 -mavx2 main.cpp -I/path/to/labneura/include -L/path/to/labneura/lib -llabneura
```

## ⚡ Performance

LabNeura delivers **SIMD-accelerated performance** competitive with or exceeding NumPy through hardware-specific optimizations:

### Benchmarks (Apple M1)
```
Backend: NEON
Size: 1,000,000 elements, 100 iterations
FP32 Addition: 2500 M ops/sec
INT8 Addition: 5000 M ops/sec (4x throughput due to wider vectors)
```

### Run Benchmarks
```bash
python examples/benchmark_numpy.py
```

### Performance Characteristics
- **NEON (ARM64)**: Processes 4 FP32 or 16 INT8 elements per instruction
- **AVX2 (x86_64)**: Processes 8 FP32 or 32 INT8 elements per instruction
- **Memory Efficiency**: INT8 mode uses 4x less memory, improving cache utilization
- **Zero Threading Overhead**: Single-threaded SIMD maximizes per-core performance

## 🏗️ Architecture

### Backend Selection Hierarchy
```
Runtime Detection → Best Available Backend
1. Check AVX2 support (x86_64) → AVX2Backend
2. Check NEON support (ARM64) → NEONBackend
3. Fallback → GenericBackend
```

### Directory Structure
```
LabNeura/
├── include/labneura/          # Public C++ headers
│   ├── tensor.h               # Main Tensor API
│   ├── types.hpp              # Type definitions
│   └── backends/              # Backend interfaces
│       ├── base.h             # TensorBackend abstract base
│       ├── avx2.h             # AVX2 SIMD backend
│       ├── neon.h             # NEON SIMD backend
│       ├── generic.h          # Portable fallback
│       ├── backend_factory.h  # Factory + detect_backend()
│       └── cpu_features.h     # Runtime CPU feature detection
├── src/labneura/              # Implementation
│   ├── tensor.cpp             # Tensor class implementation
│   └── backends/*.cpp         # Backend implementations
├── python/                    # Python bindings
│   ├── labneura_py.cpp        # pybind11 bindings
│   └── setup.py               # Python package config
├── examples/                  # Usage examples
│   ├── main.cpp               # C++ example
│   ├── test_tensor.py         # Python example
│   └── benchmark_numpy.py     # Performance comparison
├── tests/                     # Test suite
├── CMakeLists.txt             # CMake build configuration
└── Makefile                   # Build automation
```

## 🧪 Testing

### Run Tests
```bash
# Python tests
pytest tests/

# Specific test files
pytest tests/test_tensor_ops.py
pytest tests/test_quantization.py

# With coverage
pytest --cov=labneura tests/
```

### Run Examples
```bash
# C++ example
./build/labneura_example

# Python examples
python examples/test_tensor.py
python examples/benchmark_numpy.py
```

## 🛠️ Development

### Build Commands
```bash
# Clean build
make clean

# Build Python package
make build

# Install locally
make install

# Run tests
make test

# Build distribution packages
make build-dist

# Publish to TestPyPI
TEST_PYPI_TOKEN=pypi-... make publish-testpypi

# Publish to PyPI
PYPI_TOKEN=pypi-... make publish
```

### Build Flags
```bash
# Enable AVX2 (x86_64)
cmake .. -DCMAKE_CXX_FLAGS="-mavx2"

# Enable LLVM coverage
LABNEURA_COVERAGE=1 cmake ..

# Release build with optimizations
cmake .. -DCMAKE_BUILD_TYPE=Release
```

## 📊 Implementation Details

### SIMD Optimization

#### AVX2 Backend (x86_64)
- **Register Width**: 256 bits
- **FP32**: Processes 8 floats per instruction (`_mm256_add_ps`, `_mm256_mul_ps`)
- **INT8**: Processes 32 bytes per instruction (`_mm256_add_epi8`)
- **Alignment**: Pads arrays to 8-element boundaries

#### NEON Backend (ARM64/Apple Silicon)
- **Register Width**: 128 bits
- **FP32**: Processes 4 floats per instruction (`vaddq_f32`, `vmulq_f32`)
- **INT8**: Processes 16 bytes per instruction (`vaddq_s8`)
- **Alignment**: Pads arrays to 4-element boundaries
- **Optimization**: Single-threaded design maximizes instruction-level parallelism

### Quantization Strategy
- **INT8 Mode**: Stores data as 8-bit signed integers (-128 to 127)
- **Memory Reduction**: 4x smaller than FP32 (1 byte vs 4 bytes per element)
- **Saturation Arithmetic**: Prevents overflow by clamping to [-128, 127]
- **Use Cases**: Neural network inference, mobile deployment

### Backend Detection
```cpp
// Runtime detection in backend_factory.cpp
std::string detect_backend() {
    if (cpu_supports_avx2()) return "AVX2";
    if (cpu_supports_neon()) return "NEON";
    return "GENERIC";
}
```

CPU feature detection uses:
- **x86_64**: `__builtin_cpu_supports("avx2")`
- **ARM64**: Compile-time `__ARM_NEON` macro (NEON is mandatory on ARMv8)

## 🌐 Publishing

### Package Names
- **PyPI Name**: `labneura2` (install with `pip install labneura2`)
- **Python Module**: `labneura` (import with `import labneura`)
- **GitHub**: https://github.com/gokatharun/LabNeura

### CI/CD Pipeline
GitHub Actions workflow automates:
1. Build distribution packages
2. Run test suite
3. Publish to TestPyPI (manual trigger)
4. Publish to PyPI (on release tags)

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📮 Support

- **Issues**: https://github.com/gokatharun/LabNeura/issues
- **Discussions**: https://github.com/gokatharun/LabNeura/discussions

## 🙏 Acknowledgments

Built with:
- [pybind11](https://github.com/pybind/pybind11) - Seamless Python/C++ bindings
- [CMake](https://cmake.org/) - Cross-platform build system
- SIMD intrinsics: Intel AVX2, ARM NEON

---

**Author**: LabNeura Contributors  
**Version**: 0.1.2  
**Python**: 3.9+ | **C++**: 17+ | **Platforms**: macOS (Apple Silicon & Intel), Linux (x86_64, ARM64)
