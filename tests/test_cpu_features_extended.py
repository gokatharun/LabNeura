"""
Comprehensive tests for CPU feature detection and backend selection.

Tests cpu_features.cpp and backend_factory.cpp behavior.
"""

import pytest
import labneura


class TestBackendDetection:
    """Tests for CPU feature detection and backend selection"""

    def test_backend_detection_consistent(self):
        """Backend detection should be consistent"""
        backend1 = labneura.detect_backend()
        backend2 = labneura.detect_backend()
        assert backend1 == backend2, "Backend detection should be deterministic"

    def test_backend_is_valid(self):
        """Backend should be one of the known types"""
        backend = labneura.detect_backend()
        # Backend function returns an enum, just verify it works
        assert backend is not None

    def test_tensor_respects_backend(self):
        """Tensor operations should work with detected backend"""
        backend = labneura.detect_backend()
        a = labneura.Tensor([1.0, 2.0, 3.0])
        b = labneura.Tensor([1.0, 1.0, 1.0])
        result = a.add(b)
        assert result.size() == 3
        np.testing.assert_allclose(result.data_fp32(), [2.0, 3.0, 4.0], rtol=1e-6)


class TestTensorSizeCoverage:
    """Tests covering different tensor sizes to exercise all loop paths"""

    def test_single_element_fp32(self):
        """Test single element tensor"""
        t = labneura.Tensor([5.0])
        assert t.size() == 1
        assert abs(t.data_fp32()[0] - 5.0) < 1e-6

    def test_single_element_operations(self):
        """Single element add, sub, mul"""
        a = labneura.Tensor([10.0])
        b = labneura.Tensor([3.0])
        a.add_inplace(b)
        assert abs(a.data_fp32()[0] - 13.0) < 1e-6
        a.sub_inplace(b)
        assert abs(a.data_fp32()[0] - 10.0) < 1e-6
        a.mul_inplace(b)
        assert abs(a.data_fp32()[0] - 30.0) < 1e-6

    def test_four_element_tensor(self):
        """Test 4 element tensor (scalar SIMD loop size)"""
        t = labneura.Tensor([1.0, 2.0, 3.0, 4.0])
        assert t.size() == 4
        result = t.data_fp32()
        assert len(result) == 4

    def test_four_element_operations(self):
        """4 element add, sub, mul"""
        a = labneura.Tensor([1.0, 2.0, 3.0, 4.0])
        b = labneura.Tensor([1.0, 1.0, 1.0, 1.0])
        a.add_inplace(b)
        expected = [2.0, 3.0, 4.0, 5.0]
        np.testing.assert_allclose(a.data_fp32(), expected, rtol=1e-6)

    def test_eight_element_tensor(self):
        """Test 8 element tensor (8-element SIMD loop size)"""
        data = list(range(8))
        t = labneura.Tensor(data, labneura.QuantizationMode.FP32)
        assert t.size() == 8

    def test_eight_element_operations(self):
        """8 element operations"""
        a = labneura.Tensor([1.0] * 8)
        b = labneura.Tensor([2.0] * 8)
        a.mul_inplace(b)
        result = a.data_fp32()
        assert all(abs(x - 2.0) < 1e-6 for x in result)

    def test_sixteen_element_tensor(self):
        """Test 16 element tensor (16-element SIMD loop size)"""
        data = [float(i) for i in range(16)]
        t = labneura.Tensor(data)
        assert t.size() == 16

    def test_sixteen_element_operations(self):
        """16 element add operation"""
        a = labneura.Tensor([1.0] * 16)
        b = labneura.Tensor([1.5] * 16)
        c = a.add(b)
        result = c.data_fp32()
        assert all(abs(x - 2.5) < 1e-6 for x in result)

    def test_thirty_two_element_tensor(self):
        """Test 32 element tensor"""
        data = [float(i % 10) for i in range(32)]
        t = labneura.Tensor(data)
        assert t.size() == 32

    def test_sixty_four_element_tensor(self):
        """Test 64 element tensor"""
        data = [float(i) for i in range(64)]
        t = labneura.Tensor(data)
        assert t.size() == 64

    def test_hundred_element_tensor(self):
        """Test 100 element tensor"""
        data = [1.0] * 100
        t = labneura.Tensor(data)
        assert t.size() == 100
        result = t.data_fp32()
        assert len(result) == 100

    def test_thousand_element_tensor(self):
        """Test 1000 element tensor"""
        data = [1.0] * 1000
        t = labneura.Tensor(data)
        assert t.size() == 1000
        result = t.data_fp32()
        assert len(result) == 1000


class TestInt8Operations:
    """Tests for INT8 quantization mode operations"""

    def test_int8_single_element_add(self):
        """Single element INT8 add"""
        a = labneura.Tensor([10], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([5], labneura.QuantizationMode.INT8)
        a.add_inplace(b)
        assert a.data_int8() == [15]

    def test_int8_four_element_add(self):
        """4 element INT8 add"""
        a = labneura.Tensor([10, 20, 30, 40], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([1, 2, 3, 4], labneura.QuantizationMode.INT8)
        a.add_inplace(b)
        assert a.data_int8() == [11, 22, 33, 44]

    def test_int8_four_element_mul(self):
        """4 element INT8 multiplication"""
        a = labneura.Tensor([1, 2, 3, 4], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([2] * 4, labneura.QuantizationMode.INT8)
        a.mul_inplace(b)
        expected = [2, 4, 6, 8]
        assert a.data_int8() == expected

    def test_int8_small_tensor_no_corruption(self):
        """Small INT8 tensor shouldn't have memory corruption"""
        # This was a bug with NEON len4 loops
        a = labneura.Tensor([1, 2, 3, 4, 5], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([-1, -1, -1, -1, -1], labneura.QuantizationMode.INT8)
        a.mul_inplace(b)
        expected = [-1, -2, -3, -4, -5]
        assert a.data_int8() == expected


class TestMixedValues:
    """Tests with various value combinations"""

    def test_mixed_positive_negative_fp32(self):
        """Operations with mixed positive and negative values"""
        a = labneura.Tensor([-1.0, 2.0, -3.0, 4.0])
        b = labneura.Tensor([1.0, -1.0, 1.0, -1.0])
        a.mul_inplace(b)
        expected = [-1.0, -2.0, -3.0, -4.0]
        np.testing.assert_allclose(a.data_fp32(), expected, rtol=1e-6)

    def test_mixed_positive_negative_int8(self):
        """INT8 operations with mixed positive negative"""
        a = labneura.Tensor([-10, 20, -30, 40], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([2, 2, 2, 2], labneura.QuantizationMode.INT8)
        a.mul_inplace(b)
        expected = [-20, 40, -60, 80]
        assert a.data_int8() == expected

    def test_zeros_in_operations(self):
        """Operations with zero values"""
        a = labneura.Tensor([0.0, 1.0, 0.0, 2.0])
        b = labneura.Tensor([100.0, 200.0, 300.0, 400.0])
        a.add_inplace(b)
        expected = [100.0, 201.0, 300.0, 402.0]
        np.testing.assert_allclose(a.data_fp32(), expected, rtol=1e-6)

    def test_very_small_values(self):
        """Operations with very small floating point values"""
        a = labneura.Tensor([1e-6, 1e-7, 1e-8])
        b = labneura.Tensor([1e-6, 1e-7, 1e-8])
        a.add_inplace(b)
        expected = [2e-6, 2e-7, 2e-8]
        np.testing.assert_allclose(a.data_fp32(), expected, rtol=1e-3)


class TestOperationOrder:
    """Tests for operation order independence where applicable"""

    def test_add_commutative(self):
        """Addition should be commutative"""
        a = labneura.Tensor([1.0, 2.0, 3.0])
        b = labneura.Tensor([4.0, 5.0, 6.0])
        
        c = a.add(b)
        d = b.add(a)
        
        np.testing.assert_allclose(c.data_fp32(), d.data_fp32(), rtol=1e-6)

    def test_mul_inplace_commutative(self):
        """Multiplication should be commutative (inplace)"""
        a = labneura.Tensor([2.0, 3.0, 4.0])
        b = labneura.Tensor([5.0, 6.0, 7.0])
        
        # Test that a*b == b*a by doing them in separate tensors
        a1 = labneura.Tensor([2.0, 3.0, 4.0])
        b1 = labneura.Tensor([5.0, 6.0, 7.0])
        a1.mul_inplace(b1)
        
        a2 = labneura.Tensor([5.0, 6.0, 7.0])
        b2 = labneura.Tensor([2.0, 3.0, 4.0])
        a2.mul_inplace(b2)
        
        np.testing.assert_allclose(a1.data_fp32(), a2.data_fp32(), rtol=1e-6)

    def test_sub_not_commutative(self):
        """Subtraction order matters"""
        a = labneura.Tensor([10.0, 20.0, 30.0])
        b = labneura.Tensor([1.0, 2.0, 3.0])
        
        # a - b
        a1 = labneura.Tensor([10.0, 20.0, 30.0])
        b1 = labneura.Tensor([1.0, 2.0, 3.0])
        a1.sub_inplace(b1)
        result_ab = a1.data_fp32()
        
        # b - a (negated)
        a2 = labneura.Tensor([1.0, 2.0, 3.0])
        b2 = labneura.Tensor([10.0, 20.0, 30.0])
        a2.sub_inplace(b2)
        result_ba = a2.data_fp32()
        
        # Results should be negatives of each other
        for ab_val, ba_val in zip(result_ab, result_ba):
            assert abs(ab_val + ba_val) < 1e-5


class TestBackendBehavior:
    """Tests to verify backend-specific behavior through operations"""

    def test_large_result_values(self):
        """Operations resulting in large values"""
        a = labneura.Tensor([1e6] * 8)
        b = labneura.Tensor([2.0] * 8)
        a.mul_inplace(b)
        result = a.data_fp32()
        assert all(abs(x - 2e6) < 1e4 for x in result)

    def test_remainder_handling_non_multiple_of_simd(self):
        """Tensors with sizes not divisible by SIMD width"""
        # 5 elements: some backends process 4+1, others 8+5, etc.
        a = labneura.Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        b = labneura.Tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        a.add_inplace(b)
        expected = [2.0, 3.0, 4.0, 5.0, 6.0]
        np.testing.assert_allclose(a.data_fp32(), expected, rtol=1e-6)

    def test_remainder_handling_size_7(self):
        """7 element tensor remainder handling"""
        a = labneura.Tensor(list(range(1, 8)), labneura.QuantizationMode.FP32)
        b = labneura.Tensor([1.0] * 7)
        a.add_inplace(b)
        expected = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        np.testing.assert_allclose(a.data_fp32(), expected, rtol=1e-6)

    def test_remainder_handling_size_9(self):
        """9 element tensor remainder handling"""
        a = labneura.Tensor(list(range(1, 10)), labneura.QuantizationMode.FP32)
        b = labneura.Tensor([2.0] * 9)
        a.mul_inplace(b)
        expected = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
        np.testing.assert_allclose(a.data_fp32(), expected, rtol=1e-6)


import numpy as np  # Import at end to match test usage
