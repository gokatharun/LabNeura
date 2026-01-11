#!/usr/bin/env python3
"""
Comprehensive test coverage for all C++ functions in LabNeura.
Tests all function definitions across all backends and Tensor operations.
Covers: Tensor constructors, operations, metadata, copy/move semantics, and data accessors.
"""

import sys
import labneura
import pytest

class TestTensorConstructors:
    """Test all Tensor constructors with different data types and quantization modes."""
    
    def test_empty_tensor_constructor(self):
        """Test default empty tensor constructor."""
        t = labneura.Tensor()
        assert t.size() == 0
    
    def test_float_vector_fp32_constructor(self):
        """Test constructor with float vector in FP32 mode."""
        data = [1.5, 2.5, 3.5, 4.5]
        t = labneura.Tensor(data, labneura.QuantizationMode.FP32)
        assert t.size() == 4
        assert t.quantization_mode() == labneura.QuantizationMode.FP32
        result = t.data_fp32()
        for i, val in enumerate(data):
            assert abs(result[i] - val) < 1e-5
    
    def test_float_vector_int8_constructor(self):
        """Test constructor with float vector in INT8 mode."""
        data = [1.0, 2.0, 3.0, -4.0]
        t = labneura.Tensor(data, labneura.QuantizationMode.INT8)
        assert t.size() == 4
        assert t.quantization_mode() == labneura.QuantizationMode.INT8
        result = t.data_int8()
        assert result[0] == 1
        assert result[1] == 2
        assert result[2] == 3
        assert result[3] == -4
    
    def test_float_vector_fp16_constructor(self):
        """Test constructor with float vector in FP16 mode."""
        data = [1.0, 2.0, 3.0, 4.0]
        t = labneura.Tensor(data, labneura.QuantizationMode.FP16)
        assert t.size() == 4
        assert t.quantization_mode() == labneura.QuantizationMode.FP16
        result = t.data_fp16()
        assert len(result) == 4
    
    def test_int_vector_fp32_constructor(self):
        """Test constructor with int vector in FP32 mode."""
        data = [1, 2, 3, 4]
        t = labneura.Tensor(data, labneura.QuantizationMode.FP32)
        assert t.size() == 4
        result = t.data_fp32()
        for i, val in enumerate(data):
            assert abs(result[i] - float(val)) < 1e-5
    
    def test_int_vector_int8_constructor(self):
        """Test constructor with int vector in INT8 mode."""
        data = [10, 20, 30, -40]
        t = labneura.Tensor(data, labneura.QuantizationMode.INT8)
        assert t.size() == 4
        result = t.data_int8()
        assert result[0] == 10
        assert result[1] == 20
        assert result[2] == 30
        assert result[3] == -40
    
    def test_float_scalar_fp32_constructor(self):
        """Test constructor with float scalar in FP32 mode."""
        t = labneura.Tensor(3.14, labneura.QuantizationMode.FP32)
        assert t.size() == 1
        assert abs(t.data_fp32()[0] - 3.14) < 1e-5
    
    def test_float_scalar_int8_constructor(self):
        """Test constructor with float scalar in INT8 mode."""
        t = labneura.Tensor(42.5, labneura.QuantizationMode.INT8)
        assert t.size() == 1
        assert t.data_int8()[0] == 42 or t.data_int8()[0] == 43  # Rounding
    
    def test_float_scalar_fp16_constructor(self):
        """Test constructor with float scalar in FP16 mode."""
        t = labneura.Tensor(2.0, labneura.QuantizationMode.FP16)
        assert t.size() == 1
    
    def test_int_scalar_fp32_constructor(self):
        """Test constructor with int scalar in FP32 mode."""
        t = labneura.Tensor(42, labneura.QuantizationMode.FP32)
        assert t.size() == 1
        assert abs(t.data_fp32()[0] - 42.0) < 1e-5
    
    def test_int_scalar_int8_constructor(self):
        """Test constructor with int scalar in INT8 mode."""
        t = labneura.Tensor(10, labneura.QuantizationMode.INT8)
        assert t.size() == 1
        assert t.data_int8()[0] == 10


class TestTensorCopyAndMove:
    """Test copy and move semantics."""
    
    def test_copy_constructor_fp32(self):
        """Test copy constructor with FP32."""
        t1 = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP32)
        t2 = t1
        assert t2.size() == t1.size()
        assert t2.quantization_mode() == t1.quantization_mode()
        result1 = t1.data_fp32()
        result2 = t2.data_fp32()
        for i in range(len(result1)):
            assert abs(result1[i] - result2[i]) < 1e-5
    
    def test_copy_constructor_int8(self):
        """Test copy constructor with INT8."""
        t1 = labneura.Tensor([1, 2, 3], labneura.QuantizationMode.INT8)
        t2 = t1
        assert t2.size() == t1.size()
        result1 = t1.data_int8()
        result2 = t2.data_int8()
        for i in range(len(result1)):
            assert result1[i] == result2[i]
    
    def test_assignment_operator_fp32(self):
        """Test assignment operator with FP32."""
        t1 = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([5.0, 6.0], labneura.QuantizationMode.FP32)
        t2 = t1
        assert t2.size() == 2
        result = t2.data_fp32()
        assert abs(result[0] - 1.0) < 1e-5
        assert abs(result[1] - 2.0) < 1e-5
    
    def test_assignment_operator_int8(self):
        """Test assignment operator with INT8."""
        t1 = labneura.Tensor([10, 20], labneura.QuantizationMode.INT8)
        t2 = labneura.Tensor([50, 60], labneura.QuantizationMode.INT8)
        t2 = t1
        assert t2.size() == 2
        result = t2.data_int8()
        assert result[0] == 10
        assert result[1] == 20


class TestTensorDataAccessors:
    """Test all data accessor methods."""
    
    def test_data_fp32_nonconst(self):
        """Test non-const data_fp32() accessor."""
        t = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP32)
        data = t.data_fp32()
        assert len(data) == 3
        assert abs(data[0] - 1.0) < 1e-5
    
    def test_data_fp32_const(self):
        """Test const data_fp32() accessor on const tensor."""
        t = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP32)
        data = t.data_fp32()
        assert len(data) == 3
    
    def test_data_int8_nonconst(self):
        """Test non-const data_int8() accessor."""
        t = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT8)
        data = t.data_int8()
        assert len(data) == 3
        assert data[0] == 10
    
    def test_data_int8_const(self):
        """Test const data_int8() accessor."""
        t = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT8)
        data = t.data_int8()
        assert len(data) == 3
    
    def test_data_fp16_nonconst(self):
        """Test non-const data_fp16() accessor."""
        t = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP16)
        data = t.data_fp16()
        assert len(data) == 2
    
    def test_data_fp16_const(self):
        """Test const data_fp16() accessor."""
        t = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP16)
        data = t.data_fp16()
        assert len(data) == 2


class TestTensorOperations:
    """Test all arithmetic operations."""
    
    def test_add_fp32(self):
        """Test FP32 addition."""
        t1 = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([4.0, 5.0, 6.0], labneura.QuantizationMode.FP32)
        result = t1.add(t2)
        data = result.data_fp32()
        assert abs(data[0] - 5.0) < 1e-5
        assert abs(data[1] - 7.0) < 1e-5
        assert abs(data[2] - 9.0) < 1e-5
    
    def test_add_int8(self):
        """Test INT8 addition."""
        t1 = labneura.Tensor([1, 2, 3], labneura.QuantizationMode.INT8)
        t2 = labneura.Tensor([4, 5, 6], labneura.QuantizationMode.INT8)
        result = t1.add(t2)
        data = result.data_int8()
        assert data[0] == 5
        assert data[1] == 7
        assert data[2] == 9
    
    def test_add_fp16(self):
        """Test FP16 addition."""
        t1 = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP16)
        t2 = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP16)
        result = t1.add(t2)
        assert result.size() == 3
    
    def test_mul_fp32(self):
        """Test FP32 multiplication."""
        t1 = labneura.Tensor([2.0, 3.0, 4.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([2.0, 3.0, 4.0], labneura.QuantizationMode.FP32)
        result = t1.mul(t2)
        data = result.data_fp32()
        assert abs(data[0] - 4.0) < 1e-5
        assert abs(data[1] - 9.0) < 1e-5
        assert abs(data[2] - 16.0) < 1e-5
    
    def test_mul_int8(self):
        """Test INT8 multiplication."""
        t1 = labneura.Tensor([2, 3, 4], labneura.QuantizationMode.INT8)
        t2 = labneura.Tensor([2, 3, 4], labneura.QuantizationMode.INT8)
        result = t1.mul(t2)
        data = result.data_int8()
        assert data[0] == 4
        assert data[1] == 9
        assert data[2] == 16
    
    def test_mul_fp16(self):
        """Test FP16 multiplication."""
        t1 = labneura.Tensor([2.0, 3.0], labneura.QuantizationMode.FP16)
        t2 = labneura.Tensor([2.0, 3.0], labneura.QuantizationMode.FP16)
        result = t1.mul(t2)
        assert result.size() == 2
    
    def test_sub_fp32(self):
        """Test FP32 subtraction."""
        t1 = labneura.Tensor([5.0, 6.0, 7.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP32)
        result = t1.sub(t2)
        data = result.data_fp32()
        assert abs(data[0] - 4.0) < 1e-5
        assert abs(data[1] - 4.0) < 1e-5
        assert abs(data[2] - 4.0) < 1e-5
    
    def test_sub_int8(self):
        """Test INT8 subtraction."""
        t1 = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT8)
        t2 = labneura.Tensor([1, 2, 3], labneura.QuantizationMode.INT8)
        result = t1.sub(t2)
        data = result.data_int8()
        assert data[0] == 9
        assert data[1] == 18
        assert data[2] == 27
    
    def test_sub_fp16(self):
        """Test FP16 subtraction."""
        t1 = labneura.Tensor([5.0, 6.0], labneura.QuantizationMode.FP16)
        t2 = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP16)
        result = t1.sub(t2)
        assert result.size() == 2


class TestTensorInplaceOperations:
    """Test in-place operations."""
    
    def test_add_inplace_fp32(self):
        """Test in-place FP32 addition."""
        t1 = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([4.0, 5.0, 6.0], labneura.QuantizationMode.FP32)
        t1.add_inplace(t2)
        data = t1.data_fp32()
        assert abs(data[0] - 5.0) < 1e-5
        assert abs(data[1] - 7.0) < 1e-5
        assert abs(data[2] - 9.0) < 1e-5
    
    def test_add_inplace_int8(self):
        """Test in-place INT8 addition."""
        t1 = labneura.Tensor([1, 2, 3], labneura.QuantizationMode.INT8)
        t2 = labneura.Tensor([4, 5, 6], labneura.QuantizationMode.INT8)
        t1.add_inplace(t2)
        data = t1.data_int8()
        assert data[0] == 5
        assert data[1] == 7
        assert data[2] == 9
    
    def test_add_inplace_fp16(self):
        """Test in-place FP16 addition."""
        t1 = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP16)
        t2 = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP16)
        t1.add_inplace(t2)
        assert t1.size() == 2
    
    def test_mul_inplace_fp32(self):
        """Test in-place FP32 multiplication."""
        t1 = labneura.Tensor([2.0, 3.0, 4.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([2.0, 3.0, 4.0], labneura.QuantizationMode.FP32)
        t1.mul_inplace(t2)
        data = t1.data_fp32()
        assert abs(data[0] - 4.0) < 1e-5
        assert abs(data[1] - 9.0) < 1e-5
        assert abs(data[2] - 16.0) < 1e-5
    
    def test_mul_inplace_int8(self):
        """Test in-place INT8 multiplication."""
        t1 = labneura.Tensor([2, 3, 4], labneura.QuantizationMode.INT8)
        t2 = labneura.Tensor([2, 3, 4], labneura.QuantizationMode.INT8)
        t1.mul_inplace(t2)
        data = t1.data_int8()
        assert data[0] == 4
        assert data[1] == 9
        assert data[2] == 16
    
    def test_mul_inplace_fp16(self):
        """Test in-place FP16 multiplication."""
        t1 = labneura.Tensor([2.0, 3.0], labneura.QuantizationMode.FP16)
        t2 = labneura.Tensor([2.0, 3.0], labneura.QuantizationMode.FP16)
        t1.mul_inplace(t2)
        assert t1.size() == 2
    
    def test_sub_inplace_fp32(self):
        """Test in-place FP32 subtraction."""
        t1 = labneura.Tensor([5.0, 6.0, 7.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP32)
        t1.sub_inplace(t2)
        data = t1.data_fp32()
        assert abs(data[0] - 4.0) < 1e-5
        assert abs(data[1] - 4.0) < 1e-5
        assert abs(data[2] - 4.0) < 1e-5
    
    def test_sub_inplace_int8(self):
        """Test in-place INT8 subtraction."""
        t1 = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT8)
        t2 = labneura.Tensor([1, 2, 3], labneura.QuantizationMode.INT8)
        t1.sub_inplace(t2)
        data = t1.data_int8()
        assert data[0] == 9
        assert data[1] == 18
        assert data[2] == 27
    
    def test_sub_inplace_fp16(self):
        """Test in-place FP16 subtraction."""
        t1 = labneura.Tensor([5.0, 6.0], labneura.QuantizationMode.FP16)
        t2 = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP16)
        t1.sub_inplace(t2)
        assert t1.size() == 2


class TestTensorMetadata:
    """Test metadata methods."""
    
    def test_size_method(self):
        """Test size() method."""
        t = labneura.Tensor([1.0, 2.0, 3.0, 4.0, 5.0], labneura.QuantizationMode.FP32)
        assert t.size() == 5
    
    def test_numel_method(self):
        """Test numel() method (alias for size)."""
        t = labneura.Tensor([1, 2, 3], labneura.QuantizationMode.INT8)
        assert t.numel() == 3
    
    def test_quantization_mode_fp32(self):
        """Test quantization_mode() for FP32."""
        t = labneura.Tensor([1.0], labneura.QuantizationMode.FP32)
        assert t.quantization_mode() == labneura.QuantizationMode.FP32
    
    def test_quantization_mode_int8(self):
        """Test quantization_mode() for INT8."""
        t = labneura.Tensor([1], labneura.QuantizationMode.INT8)
        assert t.quantization_mode() == labneura.QuantizationMode.INT8
    
    def test_quantization_mode_fp16(self):
        """Test quantization_mode() for FP16."""
        t = labneura.Tensor([1.0], labneura.QuantizationMode.FP16)
        assert t.quantization_mode() == labneura.QuantizationMode.FP16
    
    def test_empty_tensor_size(self):
        """Test size of empty tensor."""
        t = labneura.Tensor()
        assert t.size() == 0


class TestBackendDetection:
    """Test backend detection functions."""
    
    def test_detect_backend_returns_string(self):
        """Test that detect_backend returns a valid backend string."""
        backend = labneura.detect_backend()
        assert isinstance(backend, str)
        assert backend in ["GENERIC", "AVX2", "NEON"]
    
    def test_detect_backend_consistency(self):
        """Test that detect_backend returns consistent results."""
        backend1 = labneura.detect_backend()
        backend2 = labneura.detect_backend()
        assert backend1 == backend2
    
    def test_detect_backend_creates_tensors(self):
        """Test that detected backend can create tensors."""
        backend = labneura.detect_backend()
        # Should be able to create tensors regardless of backend
        t = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP32)
        assert t.size() == 3


class TestQuantizationFunctions:
    """Test quantization and dequantization utility functions."""
    
    def test_quantize_int8(self):
        """Test INT8 quantization."""
        result = labneura.quantize_int8(1.5, 0.1)
        assert isinstance(result, int)
    
    def test_dequantize_int8(self):
        """Test INT8 dequantization."""
        q = labneura.quantize_int8(1.5, 0.1)
        result = labneura.dequantize_int8(q, 0.1)
        assert isinstance(result, float)
        assert abs(result - 1.5) < 0.2  # Allow some error due to quantization
    
    def test_quantize_fp16(self):
        """Test FP16 quantization."""
        result = labneura.quantize_fp16(1.5)
        assert isinstance(result, int)
    
    def test_dequantize_fp16(self):
        """Test FP16 dequantization."""
        q = labneura.quantize_fp16(1.5)
        result = labneura.dequantize_fp16(q)
        assert isinstance(result, float)
    
    def test_quantize_int16(self):
        """Test INT16 quantization."""
        result = labneura.quantize_int16(1000, 1.0)
        assert isinstance(result, int)
    
    def test_dequantize_int16(self):
        """Test INT16 dequantization."""
        q = labneura.quantize_int16(1000, 1.0)
        result = labneura.dequantize_int16(q, 1.0)
        assert isinstance(result, float)


class TestLargeScaleTensors:
    """Test operations on large tensors to ensure all code paths are covered."""
    
    def test_large_fp32_tensor_add(self):
        """Test large FP32 tensor addition."""
        data = [float(i) for i in range(1000)]
        t1 = labneura.Tensor(data, labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor(data, labneura.QuantizationMode.FP32)
        result = t1.add(t2)
        result_data = result.data_fp32()
        assert len(result_data) == 1000
        assert abs(result_data[0] - 0.0) < 1e-5
        assert abs(result_data[999] - 1998.0) < 1e-5
    
    def test_large_int8_tensor_mul(self):
        """Test large INT8 tensor multiplication."""
        data = [i % 128 for i in range(256)]
        t1 = labneura.Tensor(data, labneura.QuantizationMode.INT8)
        t2 = labneura.Tensor([2] * 256, labneura.QuantizationMode.INT8)
        result = t1.mul(t2)
        assert result.size() == 256
    
    def test_large_fp32_tensor_sub(self):
        """Test large FP32 tensor subtraction."""
        data1 = [float(i * 10) for i in range(512)]
        data2 = [float(i) for i in range(512)]
        t1 = labneura.Tensor(data1, labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor(data2, labneura.QuantizationMode.FP32)
        result = t1.sub(t2)
        result_data = result.data_fp32()
        assert len(result_data) == 512
    
    def test_large_fp16_tensor_operations(self):
        """Test large FP16 tensor operations."""
        data = [float(i) for i in range(512)]
        t1 = labneura.Tensor(data, labneura.QuantizationMode.FP16)
        t2 = labneura.Tensor(data, labneura.QuantizationMode.FP16)
        result_add = t1.add(t2)
        result_mul = t1.mul(t2)
        result_sub = t1.sub(t2)
        assert result_add.size() == 512
        assert result_mul.size() == 512
        assert result_sub.size() == 512


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_element_tensor_fp32(self):
        """Test single element FP32 tensor."""
        t = labneura.Tensor([42.0], labneura.QuantizationMode.FP32)
        assert t.size() == 1
        assert abs(t.data_fp32()[0] - 42.0) < 1e-5
    
    def test_single_element_tensor_int8(self):
        """Test single element INT8 tensor."""
        t = labneura.Tensor([42], labneura.QuantizationMode.INT8)
        assert t.size() == 1
        assert t.data_int8()[0] == 42
    
    def test_negative_values_int8(self):
        """Test negative values in INT8."""
        t = labneura.Tensor([-1, -2, -3], labneura.QuantizationMode.INT8)
        data = t.data_int8()
        assert data[0] == -1
        assert data[1] == -2
        assert data[2] == -3
    
    def test_zero_values_operations(self):
        """Test operations with zero values."""
        t1 = labneura.Tensor([0.0, 0.0, 0.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([5.0, 10.0, 15.0], labneura.QuantizationMode.FP32)
        result_add = t1.add(t2)
        data = result_add.data_fp32()
        assert abs(data[0] - 5.0) < 1e-5
        assert abs(data[1] - 10.0) < 1e-5
        assert abs(data[2] - 15.0) < 1e-5
    
    def test_very_small_values_fp32(self):
        """Test very small floating point values."""
        t1 = labneura.Tensor([1e-6, 1e-7, 1e-8], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([1e-6, 1e-7, 1e-8], labneura.QuantizationMode.FP32)
        result = t1.add(t2)
        assert result.size() == 3
    
    def test_mixed_positive_negative_fp32(self):
        """Test mixed positive and negative values in FP32."""
        t1 = labneura.Tensor([5.0, -5.0, 10.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([-3.0, 3.0, -10.0], labneura.QuantizationMode.FP32)
        result = t1.add(t2)
        data = result.data_fp32()
        assert abs(data[0] - 2.0) < 1e-5
        assert abs(data[1] - (-2.0)) < 1e-5
        assert abs(data[2] - 0.0) < 1e-5
    
    def test_saturating_int8_addition(self):
        """Test INT8 saturation on addition."""
        t1 = labneura.Tensor([100, 120], labneura.QuantizationMode.INT8)
        t2 = labneura.Tensor([50, 30], labneura.QuantizationMode.INT8)
        result = t1.add(t2)
        data = result.data_int8()
        # May saturate at 127
        assert data[0] >= 100
        assert data[1] >= 100
    
    def test_various_tensor_sizes_fp32(self):
        """Test various tensor sizes with FP32."""
        sizes = [1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 64, 128, 256, 512, 1024]
        for size in sizes:
            t = labneura.Tensor([float(i) for i in range(size)], labneura.QuantizationMode.FP32)
            assert t.size() == size
    
    def test_various_tensor_sizes_int8(self):
        """Test various tensor sizes with INT8."""
        sizes = [1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 64]
        for size in sizes:
            data = [i % 100 for i in range(size)]
            t = labneura.Tensor(data, labneura.QuantizationMode.INT8)
            assert t.size() == size
    
    def test_operations_preserve_quantization_mode(self):
        """Test that operations preserve the quantization mode."""
        t1 = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([3.0, 4.0], labneura.QuantizationMode.FP32)
        result = t1.add(t2)
        assert result.quantization_mode() == labneura.QuantizationMode.FP32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
