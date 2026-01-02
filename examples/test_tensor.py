#!/usr/bin/env python3
"""
Python test script for LabNeura Tensor class
Tests basic functionality and demonstrates NEON backend usage on Apple M1
"""

import sys
import time
import labneura

def print_hardware_info():
    """Print detected hardware"""
    print("=== Hardware Detection ===")
    # NEON detection happens at C++ compile-time
    print("✓ Testing Tensor with Python bindings")
    print("(NEON backend status: detected at C++ compile-time)\n")

def test_quantization():
    """Test quantization/dequantization functions"""
    print("=== Quantization Functions ===")
    x = 0.5
    scale = 0.1
    
    q = labneura.quantize_int8(x, scale)
    y = labneura.dequantize_int8(q, scale)
    
    print(f"Original: {x}, Quantized: {q}, Dequantized: {y}")
    assert abs(y - x) < 0.01, "Quantization error too large"
    print("✓ Quantization test passed\n")

def test_tensor_creation():
    """Test Tensor creation with different types"""
    print("=== Tensor Creation ===")
    
    # From float vector
    t1 = labneura.Tensor([1.0, 2.0, 3.0, 4.0], labneura.QuantizationMode.FP32)
    print(f"Float tensor size: {t1.size()}")
    print(f"Float tensor data: {t1.data_fp32()}")
    assert t1.size() == 4, "Size mismatch"
    assert t1.data_fp32() == [1.0, 2.0, 3.0, 4.0], "Data mismatch"
    
    # From int vector
    t2 = labneura.Tensor([5, 10, 15], labneura.QuantizationMode.FP32)
    print(f"Int-to-FP32 tensor data: {t2.data_fp32()}")
    assert t2.data_fp32() == [5.0, 10.0, 15.0], "Int conversion failed"
    
    # Scalar
    t3 = labneura.Tensor(3.5)
    print(f"Scalar tensor data: {t3.data_fp32()}")
    assert t3.data_fp32() == [3.5], "Scalar creation failed"
    
    print("✓ Tensor creation test passed\n")

def test_fp32_addition():
    """Test FP32 in-place addition"""
    print("=== FP32 Addition Test ===")
    
    t1 = labneura.Tensor([1.0, 2.0, 3.0, 4.0], labneura.QuantizationMode.FP32)
    t2 = labneura.Tensor([0.5, 0.5, 0.5, 0.5], labneura.QuantizationMode.FP32)
    
    print(f"Before: t1 = {t1.data_fp32()}")
    t1.add_inplace(t2)
    print(f"After:  t1 = {t1.data_fp32()}")
    
    expected = [1.5, 2.5, 3.5, 4.5]
    result = t1.data_fp32()
    
    for i, (r, e) in enumerate(zip(result, expected)):
        assert abs(r - e) < 1e-6, f"Element {i} mismatch: {r} != {e}"
    
    print("✓ FP32 addition test passed\n")

def test_int8_addition():
    """Test INT8 quantized addition"""
    print("=== INT8 Addition Test ===")
    
    t1 = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT8)
    t2 = labneura.Tensor([5, 10, 15], labneura.QuantizationMode.INT8)
    
    print(f"Before: t1 = {t1.data_int8()}")
    t1.add_inplace(t2)
    print(f"After:  t1 = {t1.data_int8()}")
    
    expected = [15, 30, 45]
    result = t1.data_int8()
    assert result == expected, f"INT8 addition mismatch: {result} != {expected}"
    
    print("✓ INT8 addition test passed\n")

def test_int8_saturation():
    """Test INT8 saturation (overflow handling)"""
    print("=== INT8 Saturation Test ===")
    
    t1 = labneura.Tensor([100, 120], labneura.QuantizationMode.INT8)
    t2 = labneura.Tensor([50, 50], labneura.QuantizationMode.INT8)
    
    print(f"Before: t1 = {t1.data_int8()}")
    t1.add_inplace(t2)
    print(f"After:  t1 = {t1.data_int8()}")
    
    # 100 + 50 = 150, clamped to 127 (max int8)
    # 120 + 50 = 170, clamped to 127
    expected = [127, 127]
    result = t1.data_int8()
    assert result == expected, f"Saturation mismatch: {result} != {expected}"
    
    print("✓ INT8 saturation test passed\n")

def test_add_non_inplace():
    """Test non-inplace addition (returns new tensor)"""
    print("=== Non-Inplace Addition Test ===")
    
    t1 = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP32)
    t2 = labneura.Tensor([10.0, 20.0, 30.0], labneura.QuantizationMode.FP32)
    
    print(f"t1 = {t1.data_fp32()}")
    print(f"t2 = {t2.data_fp32()}")
    
    result = t1.add(t2)  # Non-inplace
    
    print(f"t1.add(t2) = {result.data_fp32()}")
    
    # Original tensors unchanged
    assert t1.data_fp32() == [1.0, 2.0, 3.0], "t1 should be unchanged"
    assert t2.data_fp32() == [10.0, 20.0, 30.0], "t2 should be unchanged"
    assert result.data_fp32() == [11.0, 22.0, 33.0], "Result mismatch"
    
    print("✓ Non-inplace addition test passed\n")

def benchmark():
    """Benchmark Tensor operations"""
    print("=== Performance Benchmark ===")
    
    size = 1_000_000
    iterations = 100
    
    # Create large tensors
    data = [1.5] * size
    t1 = labneura.Tensor(data, labneura.QuantizationMode.FP32)
    t2 = labneura.Tensor(data, labneura.QuantizationMode.FP32)
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        t1.add_inplace(t2)
    elapsed = time.time() - start
    
    throughput = (size * iterations) / (elapsed * 1e6)  # M elements/sec
    
    print(f"Added {size:,} elements, {iterations} iterations")
    print(f"Time: {elapsed:.3f}s")
    print(f"Throughput: {throughput:.2f} M elements/sec")
    print("(NEON processes 4 floats per instruction on M1)")
    print()

def main():
    """Run all tests"""
    print("LabNeura Tensor Python Bindings Test Suite\n")
    print("=" * 50)
    print()
    
    try:
        print_hardware_info()
        test_quantization()
        test_tensor_creation()
        test_fp32_addition()
        test_int8_addition()
        test_int8_saturation()
        test_add_non_inplace()
        benchmark()
        
        print("=" * 50)
        print("✓ All tests passed!")
        return 0
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
