#!/usr/bin/env python3
"""
Diagnostic script to check if OpenMP is enabled and test threading
"""

import sys
import subprocess
import os

sys.path.insert(0, '/Users/tharungoka/LabNeura/python')

print("=" * 70)
print("LabNeura OpenMP/Threading Diagnostics")
print("=" * 70)
print()

# 1. Check build system
print("=== Build Configuration ===")
print()

# Check CMakeLists.txt
with open('/Users/tharungoka/LabNeura/CMakeLists.txt', 'r') as f:
    cmake_content = f.read()
    if '-fopenmp' in cmake_content or 'Xclang -fopenmp' in cmake_content:
        print("✓ CMakeLists.txt: OpenMP flags detected")
    else:
        print("✗ CMakeLists.txt: No OpenMP flags found")

# Check if libomp is installed
print()
print("=== libomp Installation ===")
print()
libomp_paths = [
    '/opt/homebrew/opt/libomp',
    '/usr/local/opt/libomp'
]

for path in libomp_paths:
    if os.path.exists(path):
        print(f"✓ Found libomp at: {path}")
        include_path = os.path.join(path, 'include', 'omp.h')
        lib_path = os.path.join(path, 'lib', 'libomp.dylib')
        print(f"  - omp.h: {include_path} {'✓' if os.path.exists(include_path) else '✗'}")
        print(f"  - libomp.dylib: {lib_path} {'✓' if os.path.exists(lib_path) else '✗'}")
    else:
        print(f"✗ libomp not found at: {path}")

# 2. Check tensor.cpp for OpenMP pragmas
print()
print("=== Tensor Implementation ===")
print()

with open('/Users/tharungoka/LabNeura/src/labneura/tensor.cpp', 'r') as f:
    tensor_content = f.read()
    
    pragma_count = tensor_content.count('#pragma omp')
    if pragma_count > 0:
        print(f"✓ Found {pragma_count} OpenMP pragmas in tensor.cpp")
        print("  - parallel for constructs present")
    else:
        print("✗ No OpenMP pragmas found in tensor.cpp")
    
    if '#include <omp.h>' in tensor_content:
        print("✓ omp.h header included")
    else:
        print("✗ omp.h header NOT included")

# 3. Test actual threading at runtime
print()
print("=== Runtime Threading Test ===")
print()

try:
    import labneura
    
    # Create a test tensor
    t1 = labneura.Tensor([1.0] * 1000, labneura.QuantizationMode.FP32)
    t2 = labneura.Tensor([1.0] * 1000, labneura.QuantizationMode.FP32)
    
    print("✓ LabNeura module loaded successfully")
    
    # Try to get OpenMP info (if available)
    try:
        import ctypes
        
        # Try to get number of threads
        # This is a bit hacky but works on macOS
        omp_lib = None
        for libname in ['libomp.dylib', 'libgomp.so.1', 'libgomp.so']:
            try:
                omp_lib = ctypes.CDLL(libname)
                print(f"✓ Loaded OpenMP library: {libname}")
                
                # Get max threads
                omp_get_max_threads = omp_lib.omp_get_max_threads
                omp_get_max_threads.restype = ctypes.c_int
                max_threads = omp_get_max_threads()
                print(f"✓ Max OpenMP threads available: {max_threads}")
                break
            except:
                pass
        
        if not omp_lib:
            print("⚠ Could not load OpenMP library (OpenMP may not be available)")
    except Exception as e:
        print(f"⚠ Could not query OpenMP info: {e}")
    
    # Run quick benchmark with timing
    print()
    print("=== Quick Performance Test ===")
    print()
    
    import time
    
    # Small array - should be mostly overhead
    t1_small = labneura.Tensor([1.5] * 1000)
    t2_small = labneura.Tensor([1.5] * 1000)
    
    start = time.perf_counter()
    for _ in range(100):
        t1_small.add_inplace(t2_small)
    elapsed_small = time.perf_counter() - start
    
    print(f"1K elements × 100 iterations: {elapsed_small*1000:.3f}ms")
    
    # Large array - threading should help
    t1_large = labneura.Tensor([1.5] * 1_000_000)
    t2_large = labneura.Tensor([1.5] * 1_000_000)
    
    start = time.perf_counter()
    for _ in range(10):
        t1_large.add_inplace(t2_large)
    elapsed_large = time.perf_counter() - start
    
    throughput = (1_000_000 * 10) / (elapsed_large * 1e6)
    print(f"1M elements × 10 iterations: {elapsed_large*1000:.3f}ms")
    print(f"Throughput: {throughput:.2f} M ops/sec")
    
    if throughput > 500:
        print("✓ Performance suggests multi-threading is active (>500 M ops/sec)")
    elif throughput > 100:
        print("⚠ Performance suggests single-threaded (100-500 M ops/sec)")
    else:
        print("✗ Performance is very low (<100 M ops/sec) - check compilation")
    
except ImportError as e:
    print(f"✗ Could not import labneura: {e}")
    print("  Make sure to rebuild: cd /Users/tharungoka/LabNeura/python && python3 setup.py build_ext --inplace")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("Recommendations:")
print("=" * 70)
print()
print("1. If OpenMP is NOT detected:")
print("   brew install libomp")
print()
print("2. Rebuild LabNeura:")
print("   cd /Users/tharungoka/LabNeura/build")
print("   rm -rf *")
print("   cmake ..")
print("   make -j")
print()
print("3. Rebuild Python extension:")
print("   cd /Users/tharungoka/LabNeura/python")
print("   rm -rf build *.so *.pyc")
print("   python3 setup.py build_ext --inplace")
print()
print("4. Test with:")
print("   python3 benchmark_numpy.py")
print()
