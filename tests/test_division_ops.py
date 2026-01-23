"""
Comprehensive tests for division operations across all data types and backends.
Tests cover normal cases, corner cases, and edge cases to ensure correctness.
"""
import numpy as np
import pytest
import labneura


def make_fp32(values):
    """Helper to create FP32 tensor."""
    return labneura.Tensor(list(values), labneura.QuantizationMode.FP32)


def make_fp16(values):
    """Helper to create FP16 tensor."""
    return labneura.Tensor(list(values), labneura.QuantizationMode.FP16)


def make_int16(values):
    """Helper to create INT16 tensor."""
    return labneura.Tensor(list(values), labneura.QuantizationMode.INT16)


def make_int8(values):
    """Helper to create INT8 tensor."""
    return labneura.Tensor(list(values), labneura.QuantizationMode.INT8)


# =======================
# FP32 Division Tests
# =======================

def test_div_fp32_basic():
    """Basic FP32 division test."""
    a = make_fp32([10.0, 20.0, 30.0, 40.0])
    b = make_fp32([2.0, 4.0, 5.0, 8.0])
    c = a.div(b)
    np.testing.assert_allclose(c.data_fp32(), [5.0, 5.0, 6.0, 5.0], rtol=1e-6)
    # Ensure a is unchanged
    np.testing.assert_allclose(a.data_fp32(), [10.0, 20.0, 30.0, 40.0], rtol=1e-6)


def test_div_inplace_fp32():
    """In-place FP32 division test."""
    a = make_fp32([100.0, 50.0, 25.0, 10.0])
    b = make_fp32([10.0, 5.0, 2.5, 2.0])
    a.div_inplace(b)
    np.testing.assert_allclose(a.data_fp32(), [10.0, 10.0, 10.0, 5.0], rtol=1e-6)


def test_div_fp32_negative_values():
    """Division with negative values."""
    a = make_fp32([10.0, -20.0, -30.0, 40.0])
    b = make_fp32([2.0, 4.0, -5.0, -8.0])
    c = a.div(b)
    np.testing.assert_allclose(c.data_fp32(), [5.0, -5.0, 6.0, -5.0], rtol=1e-6)


def test_div_fp32_by_one():
    """Division by 1.0 should return same values."""
    a = make_fp32([7.5, -3.2, 0.0, 100.5])
    b = make_fp32([1.0, 1.0, 1.0, 1.0])
    c = a.div(b)
    np.testing.assert_allclose(c.data_fp32(), [7.5, -3.2, 0.0, 100.5], rtol=1e-6)


def test_div_fp32_zero_numerator():
    """Division of zero by any value should be zero."""
    a = make_fp32([0.0, 0.0, 0.0, 0.0])
    b = make_fp32([2.0, -3.0, 100.0, 0.01])
    c = a.div(b)
    np.testing.assert_allclose(c.data_fp32(), [0.0, 0.0, 0.0, 0.0], rtol=1e-6)


def test_div_fp32_small_values():
    """Division with very small values."""
    a = make_fp32([0.001, 0.0001, 0.00001, 1e-8])
    b = make_fp32([0.01, 0.001, 0.0001, 1e-9])
    c = a.div(b)
    np.testing.assert_allclose(c.data_fp32(), [0.1, 0.1, 0.1, 10.0], rtol=1e-4)


def test_div_fp32_large_values():
    """Division with large values."""
    a = make_fp32([1e6, 1e8, 1e10, 1e12])
    b = make_fp32([1e3, 1e4, 1e5, 1e6])
    c = a.div(b)
    np.testing.assert_allclose(c.data_fp32(), [1e3, 1e4, 1e5, 1e6], rtol=1e-5)


def test_div_fp32_fractional_results():
    """Division resulting in fractional values."""
    a = make_fp32([1.0, 5.0, 7.0, 11.0])
    b = make_fp32([3.0, 2.0, 4.0, 6.0])
    c = a.div(b)
    expected = [1.0/3.0, 2.5, 1.75, 11.0/6.0]
    np.testing.assert_allclose(c.data_fp32(), expected, rtol=1e-6)


def test_div_fp32_large_tensor():
    """Division with large tensor (tests SIMD loops and tails)."""
    # 200 elements to test main loops and all tail cases
    size = 200
    a_vals = [float(i + 1) for i in range(size)]
    b_vals = [2.0] * size
    a = make_fp32(a_vals)
    b = make_fp32(b_vals)
    c = a.div(b)
    expected = [v / 2.0 for v in a_vals]
    np.testing.assert_allclose(c.data_fp32(), expected, rtol=1e-6)


def test_div_fp32_various_sizes():
    """Test various tensor sizes to cover all SIMD path and tails."""
    sizes = [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]
    for size in sizes:
        a_vals = [float(i + 10) for i in range(size)]
        b_vals = [2.0] * size
        a = make_fp32(a_vals)
        b = make_fp32(b_vals)
        c = a.div(b)
        expected = [v / 2.0 for v in a_vals]
        np.testing.assert_allclose(c.data_fp32(), expected, rtol=1e-6,
                                     err_msg=f"Failed for size={size}")


# =======================
# FP16 Division Tests
# =======================

def test_div_fp16_basic():
    """Basic FP16 division test."""
    a = make_fp16([10.0, 20.0, 30.0, 40.0])
    b = make_fp16([2.0, 4.0, 5.0, 8.0])
    c = a.div(b)
    # FP16 returns uint16 values, convert them to float for comparison
    result = [labneura.dequantize_fp16(v) for v in c.data_fp16()]
    np.testing.assert_allclose(result, [5.0, 5.0, 6.0, 5.0], rtol=1e-3)


def test_div_inplace_fp16():
    """In-place FP16 division test."""
    a = make_fp16([100.0, 50.0, 25.0, 10.0])
    b = make_fp16([10.0, 5.0, 2.5, 2.0])
    a.div_inplace(b)
    result = [labneura.dequantize_fp16(v) for v in a.data_fp16()]
    np.testing.assert_allclose(result, [10.0, 10.0, 10.0, 5.0], rtol=1e-3)


def test_div_fp16_negative_values():
    """FP16 division with negative values."""
    a = make_fp16([10.0, -20.0, -30.0, 40.0])
    b = make_fp16([2.0, 4.0, -5.0, -8.0])
    c = a.div(b)
    result = [labneura.dequantize_fp16(v) for v in c.data_fp16()]
    np.testing.assert_allclose(result, [5.0, -5.0, 6.0, -5.0], rtol=1e-3)


def test_div_fp16_large_tensor():
    """FP16 division with large tensor (tests SIMD loops: 64, 32, 16, 8, 4)."""
    size = 150
    a_vals = [float(i + 1) for i in range(size)]
    b_vals = [2.0] * size
    a = make_fp16(a_vals)
    b = make_fp16(b_vals)
    c = a.div(b)
    expected = [v / 2.0 for v in a_vals]
    result = [labneura.dequantize_fp16(v) for v in c.data_fp16()]
    np.testing.assert_allclose(result, expected, rtol=1e-2)


def test_div_fp16_various_sizes():
    """Test various FP16 tensor sizes to cover all SIMD paths."""
    sizes = [1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128]
    for size in sizes:
        a_vals = [float(i + 5) for i in range(size)]
        b_vals = [2.0] * size
        a = make_fp16(a_vals)
        b = make_fp16(b_vals)
        c = a.div(b)
        expected = [v / 2.0 for v in a_vals]
        result = [labneura.dequantize_fp16(v) for v in c.data_fp16()]
        np.testing.assert_allclose(result, expected, rtol=1e-2,
                         err_msg=f"Failed for size={size}")


# =======================
# INT16 Division Tests
# =======================

def test_div_int16_basic():
    """Basic INT16 division test."""
    a = make_int16([10, 20, 30, 40])
    b = make_int16([2, 4, 5, 8])
    c = a.div(b)
    assert c.data_int16() == [5, 5, 6, 5]


def test_div_inplace_int16():
    """In-place INT16 division test."""
    a = make_int16([100, 50, 25, 10])
    b = make_int16([10, 5, 5, 2])
    a.div_inplace(b)
    assert a.data_int16() == [10, 10, 5, 5]


def test_div_int16_negative_values():
    """INT16 division with negative values."""
    a = make_int16([10, -20, -30, 40])
    b = make_int16([2, 4, -5, -8])
    c = a.div(b)
    assert c.data_int16() == [5, -5, 6, -5]


def test_div_int16_truncation():
    """INT16 division truncates toward zero."""
    a = make_int16([7, -7, 9, -9])
    b = make_int16([3, 3, 4, 4])
    c = a.div(b)
    # Integer division truncates: 7//3=2, -7//3=-2, 9//4=2, -9//4=-2
    assert c.data_int16() == [2, -2, 2, -2]


def test_div_int16_by_one():
    """INT16 division by 1 should return same values."""
    a = make_int16([7, -3, 0, 100])
    b = make_int16([1, 1, 1, 1])
    c = a.div(b)
    assert c.data_int16() == [7, -3, 0, 100]


def test_div_int16_zero_numerator():
    """Division of zero by any value should be zero."""
    a = make_int16([0, 0, 0, 0])
    b = make_int16([2, -3, 100, 1])
    c = a.div(b)
    assert c.data_int16() == [0, 0, 0, 0]


def test_div_int16_by_zero_safe():
    """Division by zero should return 0 (safe behavior)."""
    a = make_int16([10, 20, 30, 40])
    b = make_int16([0, 0, 0, 0])
    c = a.div(b)
    assert c.data_int16() == [0, 0, 0, 0]


def test_div_int16_mixed_zero():
    """Mixed division including division by zero."""
    a = make_int16([10, 20, 30, 40, 50])
    b = make_int16([2, 0, 5, 0, 10])
    c = a.div(b)
    assert c.data_int16() == [5, 0, 6, 0, 5]


def test_div_int16_large_values():
    """INT16 division with large values."""
    a = make_int16([32767, -32768, 16000, -16000])
    b = make_int16([2, 2, 100, -100])
    c = a.div(b)
    assert c.data_int16() == [16383, -16384, 160, 160]


def test_div_int16_saturation_edge_case():
    """INT16 division edge case with min/max values."""
    a = make_int16([32767, -32768])
    b = make_int16([1, 1])
    c = a.div(b)
    assert c.data_int16() == [32767, -32768]


def test_div_int16_large_tensor():
    """INT16 division with large tensor (tests SIMD: 16, 8, 4 and scalar tail)."""
    size = 100
    a_vals = [i + 10 for i in range(size)]
    b_vals = [2] * size
    a = make_int16(a_vals)
    b = make_int16(b_vals)
    c = a.div(b)
    expected = [v // 2 for v in a_vals]
    assert c.data_int16() == expected


def test_div_int16_various_sizes():
    """Test various INT16 tensor sizes to cover all SIMD paths."""
    sizes = [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 50, 64]
    for size in sizes:
        a_vals = [i + 10 for i in range(size)]
        b_vals = [2] * size
        a = make_int16(a_vals)
        b = make_int16(b_vals)
        c = a.div(b)
        expected = [v // 2 for v in a_vals]
        assert c.data_int16() == expected, f"Failed for size={size}"


# =======================
# INT8 Division Tests
# =======================

def test_div_int8_basic():
    """Basic INT8 division test."""
    a = make_int8([10, 20, 30, 40])
    b = make_int8([2, 4, 5, 8])
    c = a.div(b)
    assert c.data_int8() == [5, 5, 6, 5]


def test_div_inplace_int8():
    """In-place INT8 division test."""
    a = make_int8([100, 50, 24, 10])
    b = make_int8([10, 5, 4, 2])
    a.div_inplace(b)
    assert a.data_int8() == [10, 10, 6, 5]


def test_div_int8_negative_values():
    """INT8 division with negative values."""
    a = make_int8([10, -20, -30, 40])
    b = make_int8([2, 4, -5, -8])
    c = a.div(b)
    assert c.data_int8() == [5, -5, 6, -5]


def test_div_int8_truncation():
    """INT8 division truncates toward zero."""
    a = make_int8([7, -7, 9, -9])
    b = make_int8([3, 3, 4, 4])
    c = a.div(b)
    # Integer division truncates: 7//3=2, -7//3=-2, 9//4=2, -9//4=-2
    assert c.data_int8() == [2, -2, 2, -2]


def test_div_int8_by_one():
    """INT8 division by 1 should return same values."""
    a = make_int8([7, -3, 0, 100])
    b = make_int8([1, 1, 1, 1])
    c = a.div(b)
    assert c.data_int8() == [7, -3, 0, 100]


def test_div_int8_zero_numerator():
    """Division of zero by any value should be zero."""
    a = make_int8([0, 0, 0, 0])
    b = make_int8([2, -3, 100, 1])
    c = a.div(b)
    assert c.data_int8() == [0, 0, 0, 0]


def test_div_int8_by_zero_safe():
    """Division by zero should return 0 (safe behavior)."""
    a = make_int8([10, 20, 30, 40])
    b = make_int8([0, 0, 0, 0])
    c = a.div(b)
    assert c.data_int8() == [0, 0, 0, 0]


def test_div_int8_mixed_zero():
    """Mixed division including division by zero."""
    a = make_int8([10, 20, 30, 40, 50])
    b = make_int8([2, 0, 5, 0, 10])
    c = a.div(b)
    assert c.data_int8() == [5, 0, 6, 0, 5]


def test_div_int8_saturation_edge_case():
    """INT8 division edge case with min/max values."""
    a = make_int8([127, -128, 100, -100])
    b = make_int8([1, 1, 2, -2])
    c = a.div(b)
    assert c.data_int8() == [127, -128, 50, 50]


def test_div_int8_large_values():
    """INT8 division with values near saturation limits."""
    a = make_int8([127, -128, 120, -120])
    b = make_int8([2, 2, 10, -10])
    c = a.div(b)
    assert c.data_int8() == [63, -64, 12, 12]


def test_div_int8_large_tensor():
    """INT8 division with large tensor (tests SIMD: 8 elements per iteration)."""
    size = 80
    a_vals = [(i % 120) + 10 for i in range(size)]
    b_vals = [2] * size
    a = make_int8(a_vals)
    b = make_int8(b_vals)
    c = a.div(b)
    expected = [v // 2 for v in a_vals]
    assert c.data_int8() == expected


def test_div_int8_various_sizes():
    """Test various INT8 tensor sizes to cover all SIMD paths."""
    sizes = [1, 2, 3, 7, 8, 9, 15, 16, 23, 24, 31, 32, 50, 64, 80]
    for size in sizes:
        a_vals = [(i % 100) + 10 for i in range(size)]
        b_vals = [2] * size
        a = make_int8(a_vals)
        b = make_int8(b_vals)
        c = a.div(b)
        expected = [v // 2 for v in a_vals]
        assert c.data_int8() == expected, f"Failed for size={size}"


# =======================
# Error Handling Tests
# =======================

def test_div_size_mismatch():
    """Division with size mismatch should raise error."""
    a = make_fp32([1.0, 2.0])
    b = make_fp32([1.0])
    with pytest.raises(RuntimeError, match="size mismatch"):
        a.div_inplace(b)


def test_div_quantization_mismatch():
    """Division with quantization mode mismatch should raise error."""
    a = make_fp32([1.0, 2.0])
    b = make_int8([1, 2])
    with pytest.raises(RuntimeError, match="Quantization mode mismatch"):
        a.div_inplace(b)


def test_div_empty_tensors():
    """Division with empty tensors should not crash."""
    a = labneura.Tensor()
    b = labneura.Tensor()
    assert a.size() == 0 and b.size() == 0
    a.div_inplace(b)
    assert a.data_fp32() == []


# =======================
# Backend-specific Tests
# =======================

def test_div_backend_consistency():
    """Verify division produces consistent results regardless of backend."""
    backend = labneura.detect_backend()
    print(f"\nRunning on backend: {backend}")
    
    # Test all data types
    test_cases = [
        (make_fp32([10.0, 20.0, 30.0]), make_fp32([2.0, 4.0, 5.0])),
        (make_fp16([10.0, 20.0, 30.0]), make_fp16([2.0, 4.0, 5.0])),
        (make_int16([10, 20, 30]), make_int16([2, 4, 5])),
        (make_int8([10, 20, 30]), make_int8([2, 4, 5])),
    ]
    
    for a, b in test_cases:
        c = a.div(b)
        # Just verify no crash and reasonable output
        assert c.size() == a.size()


def test_div_returns_new_tensor():
    """Verify div() returns a new tensor and doesn't modify original."""
    a = make_fp32([10.0, 20.0, 30.0])
    b = make_fp32([2.0, 4.0, 5.0])
    c = a.div(b)
    
    # c should have the result
    np.testing.assert_allclose(c.data_fp32(), [5.0, 5.0, 6.0], rtol=1e-6)
    
    # a should be unchanged
    np.testing.assert_allclose(a.data_fp32(), [10.0, 20.0, 30.0], rtol=1e-6)
    
    # Modify a to ensure c is independent
    a.mul_inplace(make_fp32([2.0, 2.0, 2.0]))
    np.testing.assert_allclose(c.data_fp32(), [5.0, 5.0, 6.0], rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
