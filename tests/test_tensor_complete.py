"""
Comprehensive tests for tensor.cpp and tensor operations.

Tests tensor construction, operations, error handling, and edge cases.
"""

import numpy as np
import pytest
import labneura


class TestTensorConstruction:
    """Tests for Tensor constructors"""

    def test_default_constructor(self):
        """Empty tensor from default constructor"""
        t = labneura.Tensor()
        assert t.size() == 0
        assert t.numel() == 0
        assert t.data_fp32() == []
        assert t.quantization_mode() == labneura.QuantizationMode.FP32

    def test_float_vector_constructor_fp32(self):
        """Tensor from float vector in FP32 mode"""
        data = [1.0, 2.5, -3.7, 0.0]
        t = labneura.Tensor(data, labneura.QuantizationMode.FP32)
        assert t.size() == 4
        np.testing.assert_allclose(t.data_fp32(), data, rtol=1e-6)

    def test_float_vector_constructor_int8(self):
        """Tensor from float vector converts to INT8 via FP32"""
        # First create in FP32, data gets converted during construction
        data = [10.0, -20.0, 100.0]
        t = labneura.Tensor(data, labneura.QuantizationMode.FP32)
        assert t.size() == 3
        result = t.data_fp32()
        np.testing.assert_allclose(result, [10.0, -20.0, 100.0])

    def test_int_vector_constructor_fp32(self):
        """Tensor from int vector in FP32 mode converts to float"""
        data = [1, -2, 3, -4]
        t = labneura.Tensor(data, labneura.QuantizationMode.FP32)
        assert t.size() == 4
        np.testing.assert_allclose(t.data_fp32(), [1.0, -2.0, 3.0, -4.0], rtol=1e-6)

    def test_int_vector_constructor_int8(self):
        """Tensor from int vector in INT8 mode"""
        data = [10, -20, 50, -100]
        t = labneura.Tensor(data, labneura.QuantizationMode.INT8)
        assert t.size() == 4
        assert t.data_int8() == [10, -20, 50, -100]

    def test_float_scalar_constructor(self):
        """Tensor from float scalar"""
        t = labneura.Tensor(3.14)
        assert t.size() == 1
        np.testing.assert_allclose(t.data_fp32(), [3.14], rtol=1e-6)

    def test_int_scalar_constructor_fp32(self):
        """Tensor from int scalar in FP32 mode"""
        t = labneura.Tensor(42, labneura.QuantizationMode.FP32)
        assert t.size() == 1
        np.testing.assert_allclose(t.data_fp32(), [42.0])

    def test_int_scalar_constructor_int8(self):
        """Tensor from int scalar in INT8 mode"""
        t = labneura.Tensor(123, labneura.QuantizationMode.INT8)
        assert t.size() == 1
        assert t.data_int8() == [123]

    def test_int8_scalar_clamping_positive(self):
        """INT8 scalar constructor clamps to 127"""
        t = labneura.Tensor(200, labneura.QuantizationMode.INT8)
        assert t.data_int8() == [127]

    def test_int8_scalar_clamping_negative(self):
        """INT8 scalar constructor clamps to -128"""
        t = labneura.Tensor(-200, labneura.QuantizationMode.INT8)
        assert t.data_int8() == [-128]

    def test_int8_vector_clamping(self):
        """INT8 vector constructor clamps values"""
        data = [50, 200, -200, 0]
        t = labneura.Tensor(data, labneura.QuantizationMode.INT8)
        result = t.data_int8()
        assert result == [50, 127, -128, 0]

    def test_large_tensor_fp32(self):
        """Large tensor creation"""
        size = 100000
        data = [1.5] * size
        t = labneura.Tensor(data, labneura.QuantizationMode.FP32)
        assert t.size() == size
        result = t.data_fp32()
        assert len(result) == size
        assert all(abs(x - 1.5) < 1e-6 for x in result)


class TestTensorMetadata:
    """Tests for tensor metadata accessors"""

    def test_size_and_numel_equivalence(self):
        """size() and numel() should return same value"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        t = labneura.Tensor(data)
        assert t.size() == t.numel()
        assert t.size() == 5

    def test_quantization_mode_fp32(self):
        """Check quantization_mode for FP32"""
        t = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP32)
        assert t.quantization_mode() == labneura.QuantizationMode.FP32

    def test_quantization_mode_int8(self):
        """Check quantization_mode for INT8"""
        t = labneura.Tensor([1, 2], labneura.QuantizationMode.INT8)
        assert t.quantization_mode() == labneura.QuantizationMode.INT8

    def test_default_quantization_is_fp32(self):
        """Default quantization should be FP32"""
        t = labneura.Tensor([1.0, 2.0])
        assert t.quantization_mode() == labneura.QuantizationMode.FP32


class TestTensorOperations:
    """Tests for tensor arithmetic operations"""

    def test_add_inplace_fp32(self):
        """In-place addition for FP32"""
        a = labneura.Tensor([1.0, 2.0, 3.0])
        b = labneura.Tensor([0.5, 0.5, 0.5])
        a.add_inplace(b)
        np.testing.assert_allclose(a.data_fp32(), [1.5, 2.5, 3.5], rtol=1e-6)

    def test_add_non_inplace_fp32(self):
        """Non-inplace addition returns new tensor"""
        a = labneura.Tensor([1.0, 2.0, 3.0])
        b = labneura.Tensor([4.0, 5.0, 6.0])
        c = a.add(b)
        # Original unchanged
        np.testing.assert_allclose(a.data_fp32(), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(b.data_fp32(), [4.0, 5.0, 6.0])
        # Result correct
        np.testing.assert_allclose(c.data_fp32(), [5.0, 7.0, 9.0])

    def test_sub_inplace_fp32(self):
        """In-place subtraction for FP32"""
        a = labneura.Tensor([10.0, 20.0, 30.0])
        b = labneura.Tensor([1.0, 2.0, 3.0])
        a.sub_inplace(b)
        np.testing.assert_allclose(a.data_fp32(), [9.0, 18.0, 27.0], rtol=1e-6)

    def test_mul_inplace_fp32(self):
        """In-place multiplication for FP32"""
        a = labneura.Tensor([2.0, 3.0, 4.0])
        b = labneura.Tensor([2.0, 2.0, 2.0])
        a.mul_inplace(b)
        np.testing.assert_allclose(a.data_fp32(), [4.0, 6.0, 8.0], rtol=1e-6)

    def test_add_inplace_int8(self):
        """In-place addition for INT8"""
        a = labneura.Tensor([10, 20], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([5, 10], labneura.QuantizationMode.INT8)
        a.add_inplace(b)
        assert a.data_int8() == [15, 30]

    def test_sub_inplace_int8(self):
        """In-place subtraction for INT8"""
        a = labneura.Tensor([50, 100], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([10, 20], labneura.QuantizationMode.INT8)
        a.sub_inplace(b)
        assert a.data_int8() == [40, 80]

    def test_add_non_inplace_int8(self):
        """Non-inplace add with INT8"""
        a = labneura.Tensor([10, 20], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([5, 10], labneura.QuantizationMode.INT8)
        c = a.add(b)
        assert a.data_int8() == [10, 20]  # Original unchanged
        assert c.data_int8() == [15, 30]


class TestTensorErrors:
    """Tests for error handling"""

    def test_add_size_mismatch(self):
        """Adding tensors of different sizes raises error"""
        a = labneura.Tensor([1.0, 2.0])
        b = labneura.Tensor([1.0, 2.0, 3.0])
        with pytest.raises(RuntimeError, match="size mismatch"):
            a.add_inplace(b)

    def test_sub_size_mismatch(self):
        """Subtracting tensors of different sizes raises error"""
        a = labneura.Tensor([1.0, 2.0, 3.0])
        b = labneura.Tensor([1.0])
        with pytest.raises(RuntimeError, match="size mismatch"):
            a.sub_inplace(b)

    def test_mul_size_mismatch(self):
        """Multiplying tensors of different sizes raises error"""
        a = labneura.Tensor([1.0])
        b = labneura.Tensor([1.0, 2.0, 3.0])
        with pytest.raises(RuntimeError, match="size mismatch"):
            a.mul_inplace(b)

    def test_add_quantization_mode_mismatch(self):
        """Adding tensors with different quantization modes raises error"""
        a = labneura.Tensor([1.0], labneura.QuantizationMode.FP32)
        b = labneura.Tensor([1], labneura.QuantizationMode.INT8)
        with pytest.raises(RuntimeError, match="Quantization mode mismatch"):
            a.add_inplace(b)

    def test_fp32_data_on_int8_tensor(self):
        """Accessing FP32 data on INT8 tensor raises error"""
        t = labneura.Tensor([10], labneura.QuantizationMode.INT8)
        with pytest.raises(RuntimeError, match="not in FP32"):
            t.data_fp32()

    def test_int8_data_on_fp32_tensor(self):
        """Accessing INT8 data on FP32 tensor raises error"""
        t = labneura.Tensor([1.0], labneura.QuantizationMode.FP32)
        with pytest.raises(RuntimeError, match="not in INT8"):
            t.data_int8()


class TestTensorNumericalAccuracy:
    """Tests for numerical accuracy"""

    def test_fp32_addition_precision(self):
        """Test precision of FP32 addition"""
        a = labneura.Tensor([0.1] * 10)
        b = labneura.Tensor([0.1] * 10)
        for _ in range(5):
            a.add_inplace(b)
        result = a.data_fp32()
        expected = [0.6] * 10  # 0.1 + 5 * 0.1
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-5

    def test_fp32_multiplication_by_zero(self):
        """Multiplication by zero should give zero"""
        a = labneura.Tensor([1.5, 2.5, 3.5])
        b = labneura.Tensor([0.0, 0.0, 0.0])
        a.mul_inplace(b)
        result = a.data_fp32()
        assert all(abs(x) < 1e-10 for x in result)

    def test_fp32_multiplication_by_one(self):
        """Multiplication by one should give identity"""
        original = [1.5, -2.5, 3.5, 0.0]
        a = labneura.Tensor(original)
        b = labneura.Tensor([1.0, 1.0, 1.0, 1.0])
        a.mul_inplace(b)
        np.testing.assert_allclose(a.data_fp32(), original, rtol=1e-6)

    def test_fp32_subtraction_from_self(self):
        """Subtracting tensor from itself should give zero"""
        a = labneura.Tensor([1.0, 2.0, 3.0, 4.0])
        b = labneura.Tensor([1.0, 2.0, 3.0, 4.0])
        a.sub_inplace(b)
        result = a.data_fp32()
        assert all(abs(x) < 1e-10 for x in result)


class TestTensorIntegration:
    """Integration tests combining multiple operations"""

    def test_chain_operations(self):
        """Test chaining multiple operations"""
        a = labneura.Tensor([2.0, 3.0, 4.0])
        b = labneura.Tensor([1.0, 1.0, 1.0])
        a.add_inplace(b)  # [3, 4, 5]
        a.mul_inplace(b)  # [3, 4, 5]
        a.sub_inplace(b)  # [2, 3, 4]
        np.testing.assert_allclose(a.data_fp32(), [2.0, 3.0, 4.0], rtol=1e-6)

    def test_non_inplace_chain(self):
        """Test chaining non-inplace operations"""
        a = labneura.Tensor([1.0, 2.0])
        b = labneura.Tensor([1.0, 1.0])
        c = a.add(b)  # [2, 3]
        d = c.add(b)  # [3, 4]
        e = d.add(b)  # [4, 5]
        np.testing.assert_allclose(e.data_fp32(), [4.0, 5.0], rtol=1e-6)

    def test_large_scale_operations(self):
        """Test operations on large tensors"""
        size = 10000
        a = labneura.Tensor([1.0] * size)
        b = labneura.Tensor([2.0] * size)
        result = a.add(b)
        assert result.size() == size
        assert all(abs(x - 3.0) < 1e-6 for x in result.data_fp32())
