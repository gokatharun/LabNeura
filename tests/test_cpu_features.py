"""
Comprehensive tests for cpu_features.cpp functionality.

These tests exercise CPU feature detection and backend selection logic
through the Python bindings.
"""

import pytest
import labneura


def test_detect_backend_returns_valid_string():
    """Test that detect_backend returns one of the valid backend names"""
    backend = labneura.detect_backend()
    assert isinstance(backend, str)
    assert backend in {"AVX2", "NEON", "GENERIC"}


def test_detect_backend_consistent():
    """Test that detect_backend returns the same value on multiple calls"""
    backend1 = labneura.detect_backend()
    backend2 = labneura.detect_backend()
    backend3 = labneura.detect_backend()
    assert backend1 == backend2 == backend3


def test_backend_selection_creates_correct_type():
    """Test that backend selection creates tensors that work correctly"""
    # Create FP32 tensor with various sizes
    for size in [1, 4, 7, 8, 16, 17, 32, 100, 1000]:
        t = labneura.Tensor([1.5] * size, labneura.QuantizationMode.FP32)
        assert t.size() == size
        assert len(t.data_fp32()) == size


def test_backend_selection_creates_int8_correctly():
    """Test that backend selection works for INT8 mode"""
    for size in [1, 4, 8, 16, 32, 100]:
        t = labneura.Tensor([10] * size, labneura.QuantizationMode.INT8)
        assert t.size() == size
        assert len(t.data_int8()) == size


def test_fallback_to_generic_on_small_tensors():
    """Test that small tensors work regardless of available SIMD"""
    # Even if AVX2/NEON not available, generic backend should work
    t = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP32)
    other = labneura.Tensor([1.0, 1.0, 1.0], labneura.QuantizationMode.FP32)
    t.add_inplace(other)
    assert t.data_fp32() == [2.0, 3.0, 4.0]


def test_operations_on_different_sizes():
    """Test that operations work on various tensor sizes (tests different code paths)"""
    for size in [1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 100]:
        a = labneura.Tensor([1.5] * size, labneura.QuantizationMode.FP32)
        b = labneura.Tensor([0.5] * size, labneura.QuantizationMode.FP32)
        a.add_inplace(b)
        result = a.data_fp32()
        assert len(result) == size
        assert all(abs(x - 2.0) < 1e-6 for x in result)


def test_add_operations_various_backends():
    """Test add operation which should use detected backend"""
    sizes = [1, 8, 16, 32, 64, 128, 1000]
    for size in sizes:
        a = labneura.Tensor([2.5] * size, labneura.QuantizationMode.FP32)
        b = labneura.Tensor([1.5] * size, labneura.QuantizationMode.FP32)
        a.add_inplace(b)
        result = a.data_fp32()
        expected = [4.0] * size
        for i, (r, e) in enumerate(zip(result, expected)):
            assert abs(r - e) < 1e-6, f"Mismatch at index {i}: {r} != {e}"


def test_subtract_operations_various_backends():
    """Test sub operation across different sizes"""
    sizes = [1, 8, 16, 32, 64]
    for size in sizes:
        a = labneura.Tensor([5.0] * size, labneura.QuantizationMode.FP32)
        b = labneura.Tensor([2.0] * size, labneura.QuantizationMode.FP32)
        a.sub_inplace(b)
        result = a.data_fp32()
        assert all(abs(x - 3.0) < 1e-6 for x in result)


def test_multiply_operations_various_backends():
    """Test mul operation across different sizes"""
    sizes = [1, 8, 16, 32, 64]
    for size in sizes:
        a = labneura.Tensor([3.0] * size, labneura.QuantizationMode.FP32)
        b = labneura.Tensor([2.0] * size, labneura.QuantizationMode.FP32)
        a.mul_inplace(b)
        result = a.data_fp32()
        assert all(abs(x - 6.0) < 1e-6 for x in result)


def test_int8_operations_across_sizes():
    """Test INT8 operations with different tensor sizes"""
    sizes = [1, 4, 8, 16, 32]
    for size in sizes:
        a = labneura.Tensor([10] * size, labneura.QuantizationMode.INT8)
        b = labneura.Tensor([5] * size, labneura.QuantizationMode.INT8)
        a.add_inplace(b)
        result = a.data_int8()
        assert all(x == 15 for x in result)


def test_mixed_positive_negative_operations():
    """Test operations with mixed positive/negative values"""
    sizes = [8, 16, 32]
    for size in sizes:
        a = labneura.Tensor([1.0, -1.0] * (size // 2), labneura.QuantizationMode.FP32)
        b = labneura.Tensor([-1.0, 1.0] * (size // 2), labneura.QuantizationMode.FP32)
        c = a.add(b)  # Non-inplace
        result = c.data_fp32()
        expected = [0.0] * size
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-6


def test_backend_independence_from_tensor_order():
    """Test that operation order doesn't affect correctness"""
    size = 100
    a1 = labneura.Tensor([2.0] * size, labneura.QuantizationMode.FP32)
    b1 = labneura.Tensor([3.0] * size, labneura.QuantizationMode.FP32)
    c1 = a1.add(b1)

    a2 = labneura.Tensor([2.0] * size, labneura.QuantizationMode.FP32)
    b2 = labneura.Tensor([3.0] * size, labneura.QuantizationMode.FP32)
    c2 = b2.add(a2)

    result1 = c1.data_fp32()
    result2 = c2.data_fp32()
    for r1, r2 in zip(result1, result2):
        assert abs(r1 - r2) < 1e-6


def test_backend_with_zero_values():
    """Test operations with zero values"""
    size = 64
    a = labneura.Tensor([0.0] * size, labneura.QuantizationMode.FP32)
    b = labneura.Tensor([5.5] * size, labneura.QuantizationMode.FP32)
    a.add_inplace(b)
    result = a.data_fp32()
    assert all(abs(x - 5.5) < 1e-6 for x in result)


def test_backend_with_very_small_values():
    """Test operations with very small floating point values"""
    size = 32
    small_val = 1e-6
    a = labneura.Tensor([small_val] * size, labneura.QuantizationMode.FP32)
    b = labneura.Tensor([small_val] * size, labneura.QuantizationMode.FP32)
    a.add_inplace(b)
    result = a.data_fp32()
    expected = 2 * small_val
    for r in result:
        assert abs(r - expected) < 1e-8
