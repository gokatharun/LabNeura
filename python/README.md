# LabNeura

[![PyPI version](https://badge.fury.io/py/labneura.svg)](https://badge.fury.io/py/labneura)
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
- **Mixed-Precision Support**: FP64, FP32, FP16, INT32, INT16, INT8 (all fully supported)
- **In-place Operations**: `add_inplace()`, `mul_inplace()`, `sub_inplace()`, `div_inplace()` for memory efficiency
- **Immutable Operations**: `add()`, `mul()`, `sub()`, `div()` return new tensors without modifying originals
- **Quantization**: Built-in INT8/INT16 quantization with up to 4x memory reduction
- **Advanced Math**: Floor division, element-wise operations with SIMD acceleration

### Python Integration
- **Easy Installation**: `pip install labneura`
- **pybind11 Bindings**: Zero-copy data access between C++ and Python
- **NumPy Compatible**: Seamless integration with NumPy workflows

## 📦 Installation

### Python Package (Recommended)

```bash
pip install labneura
```

```python
import labneura  # Import statement
```

### Build from Source

#### Prerequisites
- CMake >= 3.20
- C++20 compatible compiler
- Python >= 3.8 (for Python bindings)
- Network access to fetch the pinned slang frontend

#### C++ Library
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cmake --install .
```

#### Python Bindings
```bash
pip install .
```

The root `pyproject.toml` drives both wheel and source builds with CMake.

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
t1.div_inplace(t2)  # Element-wise division

# Multi-precision support
t_int8 = labneura.Tensor([10, 20, 30, 40], labneura.QuantizationMode.INT8)
print(t_int8.data_int8())  # [10, 20, 30, 40]

t_int16 = labneura.Tensor([100, 200, 300], labneura.QuantizationMode.INT16)
t_fp16 = labneura.Tensor([1.5, 2.5, 3.5], labneura.QuantizationMode.FP16)
```

### EDA Netlist Parsing

LabNeura's EDA submodule parses individual structural examples into netlist
objects and reads complete Verilog projects with slang's Verilog-2005 parser
and elaborator:

```python
import labneura.eda as eda

module = eda.parse_verilog("logic_example.v")
print(module.name)
for assignment in module.assignments:
    print(assignment.lhs, "=", assignment.rhs)
for instance in module.instances:
    print(instance.name, instance.cell_type)
    for connection in instance.connections:
        print(connection.pin, "->", connection.net)

eda.write_json(module, "logic_example.json")
module.dump_graph("logic_example.dot")  # Graphviz: dot -Tsvg logic_example.dot
```

`parse_verilog_project` recursively reads `.v` files, resolves module
hierarchy, and exposes an elaborated AST as JSON. Project summary counts are
available alongside module ports, wires, instances, and each instance's
formal-port-to-net connections:

```python
project = eda.parse_verilog_project("examples/verilog")
print(project.module_count, project.total_port_count, project.total_wire_count)
for module in project.modules:
    print(module.name, module.port_count, module.wire_count)
print(project.ast_json)
```

Verilog handling is split into three explicit stages:

```python
rtl = eda.parse_verilog("logic_example.v")
generic = eda.generic_map(rtl)
library = eda.parse_liberty("assets/sky130_fd_sc_hd__tt_025C_1v80.lib")
mapped = eda.tech_map(generic, library)
```

`parse_verilog` returns the parsed module, including its continuous
assignments; it does not lower those assignments into cells.
`generic_map` lowers the supported scalar Boolean assignments into
technology-independent cells. It preserves associative multi-input operators,
such as an XOR3, in the generic netlist. `tech_map` takes both that generic
netlist and an explicit `StandardCellLibrary`, resolves generic cells to
available library cells, and attaches their Liberty definitions to instances.
Modules with `always` blocks are flagged by `parse_verilog` (`is_behavioral`)
and lowered by `generic_map` (flops, latches, muxes, arithmetic, function/task
inlining). Both `generic_map` and `tech_map` also accept a whole project and a
top module, flattening the hierarchy below it:

```python
project = eda.parse_verilog_project("examples/verilog/picorv32")
mapped = eda.tech_map(project, library, top="picorv32",
                      parameters={"ENABLE_MUL": "1"})
```

For FPGAs, `fpga_map` replaces `tech_map`: it maps the generic netlist onto
K-input LUTs (`lut_size` 2-6, default 4) with a depth-optimal, area-recovering
cut mapper. Flops pass through; each LUT is a `LUT<n>` cell with pins `I0..`/`O`
and a hex `INIT` truth table (`instance.parameters["INIT"]`). Like `tech_map`, each
instance carries `instance.cell` with area, pins, capacitances and timing arcs
(LUT delay per input, flop clock-to-Q and setup). They come from the built-in
illustrative model `eda.default_fpga_library()` (area in config bits, ns, pF, no
routing delay) or from your own Liberty file via `fpga_map(..., library=lib)`. `write_blif`
exports the result for Yosys, ABC or VPR; async flops are written as `.subckt`.

```python
luts = eda.fpga_map(project, top="picorv32", lut_size=6)
eda.write_blif(luts, "picorv32.blif")
```

Unsupported constructs (e.g. `initial` blocks, inout ports) fail explicitly rather than
being silently dropped. The bundled SKY130 file is available as a convenient
library input; the educational library remains at `assets/cells.lib`.

`parse_verilog_source` remains a separate lightweight parser for source strings
and supports scalar ports/wires and `and`/`or`/`xor` primitives. Verilog parsing
requires CMake 3.20+, a C++20 compiler, and the pinned slang frontend, which is
fetched during the build.

EDA implementation sources are grouped by pipeline stage: Verilog and Liberty
parsers live under `src/labneura/eda/parser/`, the shared netlist code under
`netlist/`, and generic and technology mapping under `mapping/`. The
optimization, placement, routing, and FPGA-mapping directories are reserved
for future stage implementations.

Liberty standard-cell libraries can be read with `parse_liberty`. The reader
exposes cell area, pin direction/function/capacitance, timing arcs, and
lookup-table indices and values:

```python
library = eda.parse_liberty("cells.lib")
print(library.name, library.time_unit, library.cell_count)
cell = library.cells[0]
for pin in cell.pins:
    print(pin.name, pin.direction, pin.function)
    for arc in pin.timing_arcs:
        print(arc.related_pin, arc.timing_type)
        if arc.cell_rise is not None:
            print(arc.cell_rise.index_1, arc.cell_rise.values)
```

Liberty backslash-newline continuations are supported. The bundled default
comes from `assets/sky130_fd_sc_hd__tt_025C_1v80.lib`. It can also be loaded
directly for inspection with
`eda.parse_liberty("assets/sky130_fd_sc_hd__tt_025C_1v80.lib")`.
See [the included library notice](assets/SKY130-LIBRARY-NOTICE.md) for its
source, license, and technology/corner details.

The reader extracts common cell, pin, and delay-table data; it is not a full
Liberty semantic validator.

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

LabNeura delivers **SIMD-accelerated performance** that significantly outperforms NumPy and is competitive with PyTorch through hardware-specific optimizations optimized for quantized inference on edge devices.

### Key Results (1M Elements, Apple Silicon)

| Operation | LabNeura | vs NumPy | vs PyTorch |
|-----------|----------|----------|-----------|
| **INT16 ADD** | 67.98 μs | **6.1x faster** | 2.9x faster |
| **INT16 MUL** | 71.49 μs | **5.8x faster** | 1.96x faster |
| **FP16 ADD** | 67.94 μs | **60.4x faster** | 1.18x slower* |
| **FP16 MUL** | 67.94 μs | **134.8x faster** | 1.23x slower* |

*PyTorch benefits from GPU pipeline optimization for large batches; LabNeura is superior for CPU-only deployment.

### Complete Benchmarks

📊 **[View Full Benchmark Reports](docs/benchmarks/)**
- **[INT16 & FP16 Benchmark Report](docs/benchmarks/INT16_FP16_BENCHMARK_REPORT.md)**: Detailed comparison vs NumPy and PyTorch
- **[INT8 Performance Results](docs/benchmarks/INT8_BENCHMARK_RESULTS.md)**: Optimized quantized operations
- **[Quick Reference](docs/benchmarks/QUICK_REFERENCE.md)**: At-a-glance performance tables
- **[Start Here](docs/benchmarks/START_HERE.md)**: Benchmark documentation index
- Multiple tensor sizes (1K to 1M elements) with comprehensive analysis

### Run Benchmarks Locally

```bash
# INT16/FP16 comparison
python benchmarking/benchmark_int16_fp16.py

# View results
cat benchmarking/benchmark_int16_fp16_results.json
```

### Performance Characteristics
- **NEON (ARM64)**: Processes 4 FP32, 8 FP16, 8 INT16, or 16 INT8 elements per instruction
- **AVX2 (x86_64)**: Processes 8 FP32, 16 FP16, 16 INT16, or 32 INT8 elements per instruction
- **Memory Efficiency**: INT8 uses 4x less memory, INT16/FP16 use 2x less memory than FP32
- **Zero Threading Overhead**: Single-threaded SIMD maximizes per-core performance
- **Competitive with PyTorch**: 6.1x faster than NumPy on INT16, competitive with PyTorch on CPU

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
│   └── eda_bindings.cpp       # EDA Python bindings
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
- **PyPI Name**: `labneura` (install with `pip install labneura`)
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

## 📚 Documentation

- **[Backend Architecture](docs/BACKEND_ARCHITECTURE.md)**: Detailed SIMD implementation guide
- **[Publishing Checklist](docs/PUBLISHING_CHECKLIST.md)**: Package deployment workflow
- **[Benchmarks](docs/benchmarks/)**: Comprehensive performance analysis
- **[Testing Guide](docs/testing/TESTING_GUIDE.md)**: Test suite documentation
- **[Operations Roadmap](OPERATIONS_ROADMAP.md)**: Planned features and API design

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
**Version**: 0.0.13  
**Python**: 3.9+ | **C++**: 17+ | **Platforms**: macOS (Apple Silicon & Intel), Linux (x86_64, ARM64)
