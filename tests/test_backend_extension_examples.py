"""
Backend Extension Examples

This file demonstrates how the testing framework can be extended
to support additional backends (NEON, Generic, etc.) if they become
available or need explicit testing.

Examples are provided but not executed as separate tests.
"""

import labneura
import numpy as np


# ============================================================================
# Example 1: How to test Generic Backend explicitly
# ============================================================================

def example_generic_backend_testing():
    """
    To explicitly test the Generic (scalar) backend:
    
    1. Set environment variable to force generic backend:
       LABNEURA_FORCE_GENERIC=1 pytest tests/test_backend_*.py
    
    2. Or modify backend_factory.cpp to add a runtime flag:
       if (std::getenv("LABNEURA_FORCE_GENERIC")) return GenericBackend;
    
    3. Then run tests to verify generic backend correctness
    """
    
    # This would be tested with generic backend if env var set
    a = labneura.Tensor([1.0, 2.0, 3.0])
    b = labneura.Tensor([1.0, 1.0, 1.0])
    
    # Generic backend would process these one element at a time
    for i in range(a.size()):
        # Conceptually: a[i] = a[i] + b[i]
        # In generic backend: scalar loop, one operation per iteration
        pass


# ============================================================================
# Example 2: How to test SSE4.1 Backend explicitly
# ============================================================================

def example_sse41_backend_testing():
    """
    To explicitly test SSE4.1 backend on x86_64:
    
    1. Add to CMakeLists.txt for forced SSE4.1:
       -msse4.1 -DFORCE_SSE41_BACKEND
    
    2. In backend_factory.cpp:
       if (SSE41_AVAILABLE) return SSE41Backend;
    
    3. Tests would verify SSE4.1 processing:
       - 4 float32 per instruction (128-bit register)
       - 16 int8 per instruction (128-bit register)
       - Proper remainder handling
    """
    
    # These tests verify SSE4.1 would work correctly
    sizes = [4, 5, 8, 9, 16, 17, 32]
    
    for size in sizes:
        a = labneura.Tensor([1.0] * size)
        b = labneura.Tensor([2.0] * size)
        
        # SSE4.1 would process in 4-element chunks
        # Chunk 1: 4 elements with one instruction
        # Chunk 2: 4 elements with one instruction
        # ...
        # Remainder: scalar loop for (size % 4) elements
        
        a.add_inplace(b)
        result = a.data_fp32()
        assert all(abs(x - 3.0) < 1e-5 for x in result)


# ============================================================================
# Example 3: How to test AVX2 Backend explicitly
# ============================================================================

def example_avx2_backend_testing():
    """
    To explicitly test AVX2 backend on x86_64:
    
    1. Add to CMakeLists.txt for forced AVX2:
       -mavx2 -DFORCE_AVX2_BACKEND
    
    2. In backend_factory.cpp:
       if (AVX2_AVAILABLE) return AVX2Backend;
    
    3. Tests would verify AVX2 processing:
       - 8 float32 per instruction (256-bit register)
       - 32 int8 per instruction (256-bit register)
       - Proper remainder handling
    """
    
    # These tests verify AVX2 would work correctly
    sizes = [8, 9, 16, 17, 32, 33, 64]
    
    for size in sizes:
        a = labneura.Tensor([1.0] * size)
        b = labneura.Tensor([1.0] * size)
        
        # AVX2 would process in 8-element chunks
        # Chunk 1: 8 elements with one instruction
        # Chunk 2: 8 elements with one instruction
        # ...
        # Remainder: vector or scalar loop for (size % 8) elements
        
        a.add_inplace(b)
        result = a.data_fp32()
        assert all(abs(x - 2.0) < 1e-5 for x in result)


# ============================================================================
# Example 4: Backend-Agnostic Test Pattern
# ============================================================================

def example_backend_agnostic_pattern():
    """
    This pattern works with ANY backend - tests algorithm correctness
    without caring which backend is used.
    """
    
    class BackendAgnosticTests:
        
        @staticmethod
        def test_operation_correctness(operation_name, operand_a, operand_b, expected):
            """
            Generic test that works with any backend.
            
            Args:
                operation_name: 'add', 'sub', 'mul'
                operand_a: list of numbers
                operand_b: list of numbers
                expected: expected result
            """
            if isinstance(operand_a[0], int):
                t1 = labneura.Tensor(operand_a, labneura.QuantizationMode.INT8)
                t2 = labneura.Tensor(operand_b, labneura.QuantizationMode.INT8)
            else:
                t1 = labneura.Tensor(operand_a)
                t2 = labneura.Tensor(operand_b)
            
            if operation_name == 'add':
                t1.add_inplace(t2)
            elif operation_name == 'sub':
                t1.sub_inplace(t2)
            elif operation_name == 'mul':
                t1.mul_inplace(t2)
            else:
                raise ValueError(f"Unknown operation: {operation_name}")
            
            result = t1.data_fp32() if isinstance(operand_a[0], float) else t1.data_int8()
            
            if isinstance(operand_a[0], float):
                np.testing.assert_allclose(result, expected, rtol=1e-6)
            else:
                assert result == expected
        
        @staticmethod
        def run_backend_agnostic_suite():
            """Run a suite of tests that work with any backend."""
            
            test_cases = [
                # (operation, a, b, expected)
                ('add', [1.0, 2.0], [1.0, 2.0], [2.0, 4.0]),
                ('sub', [5.0, 10.0], [1.0, 2.0], [4.0, 8.0]),
                ('mul', [2.0, 3.0], [2.0, 2.0], [4.0, 6.0]),
                ('add', [1, 2], [1, 2], [2, 4]),
                ('mul', [10, 20], [1, 2], [10, 40]),
            ]
            
            for operation, a, b, expected in test_cases:
                BackendAgnosticTests.test_operation_correctness(
                    operation, a, b, expected
                )


# ============================================================================
# Example 5: Performance Comparison Pattern
# ============================================================================

def example_performance_testing_pattern():
    """
    How to structure performance tests across different backends.
    """
    
    import time
    
    def benchmark_operation(size, operation, iterations=100):
        """
        Benchmark an operation across different backend choices.
        """
        results = {}
        
        # Create test data
        a = labneura.Tensor([1.0] * size)
        b = labneura.Tensor([2.0] * size)
        
        # Measure performance
        start = time.time()
        for _ in range(iterations):
            if operation == 'add':
                a.add_inplace(b)
            elif operation == 'mul':
                a.mul_inplace(b)
            elif operation == 'sub':
                a.sub_inplace(b)
        elapsed = time.time() - start
        
        backend = labneura.detect_backend()
        results[backend] = {
            'size': size,
            'iterations': iterations,
            'time_seconds': elapsed,
            'ops_per_second': (size * iterations) / elapsed
        }
        
        return results
    
    # Run benchmarks
    for size in [100, 1000, 10000, 100000]:
        print(f"\nBenchmarking size {size}:")
        
        add_bench = benchmark_operation(size, 'add', iterations=10)
        for backend, stats in add_bench.items():
            print(f"  {backend}: {stats['ops_per_second']:,.0f} ops/sec")


# ============================================================================
# Example 6: Cross-Platform Testing Documentation
# ============================================================================

CROSS_PLATFORM_TEST_MATRIX = """
Platform-Specific Test Coverage Matrix
======================================

                    | ARM M1    | ARM64     | x86_64    | x86_64 (no AVX2)
Operation           | (NEON)    | (NEON)    | (AVX2)    | (SSE4.1)
--------------------|-----------|-----------|-----------|------------------
FP32 Add            | ✓         | ✓         | ✓         | ✓
FP32 Sub            | ✓         | ✓         | ✓         | ✓
FP32 Mul            | ✓         | ✓         | ✓         | ✓
INT8 Add Sat        | ✓         | ✓         | ✓         | ✓
INT8 Sub Sat        | ✓         | ✓         | ✓         | ✓
INT8 Mul Sat        | ✓         | ✓         | ✓         | ✓
CPU Detection       | ✓ (ARM)   | ✓ (ARM)   | ✓ (x86)   | ✓ (x86)
Backend Factory     | ✓         | ✓         | ✓         | ✓
Vectorization       | NEON      | NEON      | AVX2      | SSE4.1
--------------------|-----------|-----------|-----------|------------------
NEON Backend        | ✓         | ✓         | ✗         | ✗
AVX2 Backend        | ✗         | ✗         | ✓         | ✗
SSE4.1 Backend      | ✗         | ✗         | ✓/Test    | ✓
Generic Backend     | Test/Fb   | Test/Fb   | Test/Fb   | Test/Fb
--------------------|-----------|-----------|-----------|------------------

Legend:
✓ = Tested and working
✗ = Not available on platform
Test/Fb = Can test by forcing generic backend (fallback)
Test = Can test backend-specific code
"""

print(CROSS_PLATFORM_TEST_MATRIX)


# ============================================================================
# Example 7: How to Add New Backend Tests
# ============================================================================

def example_adding_new_backend_tests():
    """
    Template for adding tests when new backends are added.
    """
    
    class NewBackendTestTemplate:
        """
        Template for testing a new backend.
        Copy and adapt this for each new backend.
        """
        
        @staticmethod
        def test_new_backend_basic_operations():
            """Test basic operations work with new backend."""
            a = labneura.Tensor([1.0, 2.0, 3.0])
            b = labneura.Tensor([1.0, 1.0, 1.0])
            
            # Test add
            a_copy = labneura.Tensor([1.0, 2.0, 3.0])
            a_copy.add_inplace(b)
            assert len(a_copy.data_fp32()) == 3
            
            # Test sub
            a.sub_inplace(b)
            assert len(a.data_fp32()) == 3
            
            # Test mul
            a.mul_inplace(labneura.Tensor([2.0, 2.0, 2.0]))
            assert len(a.data_fp32()) == 3
        
        @staticmethod
        def test_new_backend_vectorization():
            """
            Test new backend's vectorization characteristics.
            Replace ELEMENT_SIZE with backend's actual vector size.
            """
            ELEMENT_SIZE = 8  # For example, if backend processes 8 elements
            
            for size in range(1, ELEMENT_SIZE * 3 + 1):
                a = labneura.Tensor([1.0] * size)
                b = labneura.Tensor([1.0] * size)
                a.add_inplace(b)
                result = a.data_fp32()
                assert all(abs(x - 2.0) < 1e-5 for x in result)
        
        @staticmethod
        def test_new_backend_data_types():
            """Test new backend with different data types."""
            # FP32
            fp32 = labneura.Tensor([1.0, 2.0, 3.0])
            fp32.add_inplace(labneura.Tensor([1.0, 1.0, 1.0]))
            
            # INT8
            int8 = labneura.Tensor([1, 2, 3], labneura.QuantizationMode.INT8)
            int8.add_inplace(labneura.Tensor([1, 1, 1], labneura.QuantizationMode.INT8))
            
            assert fp32.size() == int8.size()


if __name__ == "__main__":
    # These are examples, not executable tests
    print("This file contains examples of how to extend backend tests.")
    print("See documentation in each function for details.")
