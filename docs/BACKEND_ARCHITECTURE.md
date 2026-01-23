# LabNeura Backend Architecture

## Modular Backend Organization

The LabNeura backends have been modularized for better maintainability, faster compilation, and easier development.

### Directory Structure

```
src/labneura/backends/
├── base.cpp                        # Base class implementation (65 lines)
├── backend_factory.cpp             # Factory for backend selection (75 lines)
├── cpu_features.cpp                # CPU feature detection (62 lines)
│
├── avx2/                          # AVX2 backend (x86-64 SIMD)
│   ├── avx2_backend.cpp           # Main backend class (183 lines)
│   ├── avx2_ops_fp32.cpp          # FP32 operations (607 lines)
│   ├── avx2_ops_int8.cpp          # INT8 operations (138 lines)
│   ├── avx2_ops_int16.cpp         # INT16 operations (400 lines)
│   ├── avx2_ops_fp16.cpp          # FP16 operations (367 lines)
│   └── avx2_internal.h            # Internal helpers (66 lines)
│
├── generic/                       # Generic scalar backend (fallback)
│   ├── generic_backend.cpp        # Main backend class (75 lines)
│   ├── generic_ops_fp32.cpp       # FP32 operations (25 lines)
│   ├── generic_ops_int8.cpp       # INT8 operations (43 lines)
│   └── generic_internal.h         # Internal helpers (5 lines)
│
└── neon/                          # NEON backend (ARM64 SIMD)
    ├── neon_backend.cpp           # Main backend class (142 lines)
    ├── neon_ops_fp32.cpp          # FP32 operations (768 lines)
    ├── neon_ops_int8.cpp          # INT8 operations (376 lines)
    ├── neon_ops_int16.cpp         # INT16 operations (382 lines)
    └── neon_internal.h            # Internal helpers (10 lines)
```

## Before vs After

### Before Modularization
- **avx2.cpp**: 1,666 lines (monolithic)
- **generic.cpp**: 129 lines
- **neon.cpp**: 1,628 lines (monolithic)

### After Modularization
- **Largest file**: 768 lines (neon_ops_fp32.cpp)
- **Average file size**: ~250 lines
- **Total files**: 18 modular files vs 3 monolithic files

## Benefits

1. **Faster Compilation**
   - Changing INT8 operations only recompiles that specific file
   - Parallel compilation across different operation types
   - CMake incremental builds are much faster

2. **Better Code Navigation**
   - Easy to find specific operation implementations
   - Clear separation of concerns by data type
   - Reduced cognitive load when working on specific features

3. **Parallel Development**
   - Multiple developers can work on different data types simultaneously
   - Fewer merge conflicts in separate files
   - Clear ownership boundaries

4. **Maintainability**
   - Each file has single responsibility
   - Easier to understand and test individual components
   - Simpler to add new operations or data types

## Build Configuration

### CMakeLists.txt
All modular files are included in the build:

```cmake
add_library(labneura
  src/labneura/tensor.cpp
  src/labneura/backends/base.cpp
  src/labneura/backends/cpu_features.cpp
  src/labneura/backends/backend_factory.cpp
  # Generic backend (modular)
  src/labneura/backends/generic/generic_backend.cpp
  src/labneura/backends/generic/generic_ops_fp32.cpp
  src/labneura/backends/generic/generic_ops_int8.cpp
  # NEON backend (modular)
  src/labneura/backends/neon/neon_backend.cpp
  src/labneura/backends/neon/neon_ops_fp32.cpp
  src/labneura/backends/neon/neon_ops_int8.cpp
  src/labneura/backends/neon/neon_ops_int16.cpp
  # AVX2 backend (modular)
  src/labneura/backends/avx2/avx2_backend.cpp
  src/labneura/backends/avx2/avx2_ops_fp32.cpp
  src/labneura/backends/avx2/avx2_ops_int8.cpp
  src/labneura/backends/avx2/avx2_ops_fp16.cpp
  src/labneura/backends/avx2/avx2_ops_int16.cpp
)
```

### Python setup.py
Python bindings reference all modular files similarly.

## Adding New Operations

To add a new operation (e.g., `EXP`):

1. Add to relevant operation files by data type:
   - `avx2/avx2_ops_fp32.cpp` for AVX2 FP32 implementation
   - `neon/neon_ops_fp32.cpp` for NEON FP32 implementation
   - `generic/generic_ops_fp32.cpp` for fallback implementation

2. No need to modify other data type files if operation only applies to FP32

3. Compile only the affected files - much faster iteration!

## Testing

All 557 tests pass with the modular architecture:
```bash
$ pytest tests/ -v
=============== 557 passed, 27 skipped, 3 warnings in 14.09s ===============
```

The modularization is transparent to users - all APIs remain unchanged.

## Future Improvements

- Add more granular operation files (e.g., separate ADD/MUL/DIV)
- Create shared SIMD utility headers for common patterns
- Add backend-specific optimization flags per file
- Consider compile-time polymorphism for operation dispatch
