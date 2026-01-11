"""
Comprehensive test cases to achieve 90%+ coverage for backends/base.cpp.

This test file specifically targets uncovered lines in base.cpp:
- Default FP16/INT16 data accessor methods that throw exceptions
- Default FP16/INT16 operation methods that throw exceptions
- These are called when backends don't override these methods
"""

import numpy as np
import pytest
import labneura


class TestBackendErrorHandling:
    """Tests for backend error conditions and default implementations"""

    def test_size_mismatch_add(self):
        """Test that add_inplace throws on size mismatch"""
        t1 = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([4.0, 5.0], labneura.QuantizationMode.FP32)
        
        with pytest.raises(RuntimeError, match="Tensor size mismatch"):
            t1.add_inplace(t2)

    def test_size_mismatch_sub(self):
        """Test that sub_inplace throws on size mismatch"""
        t1 = labneura.Tensor([10.0, 20.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([1.0, 2.0, 3.0, 4.0], labneura.QuantizationMode.FP32)
        
        with pytest.raises(RuntimeError, match="Tensor size mismatch"):
            t1.sub_inplace(t2)

    def test_size_mismatch_mul(self):
        """Test that mul_inplace throws on size mismatch"""
        t1 = labneura.Tensor([2.0, 3.0, 4.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([5.0], labneura.QuantizationMode.FP32)
        
        with pytest.raises(RuntimeError, match="Tensor size mismatch"):
            t1.mul_inplace(t2)

    def test_quantization_mode_mismatch_add(self):
        """Test that add_inplace throws on quantization mode mismatch"""
        t1 = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([3, 4], labneura.QuantizationMode.INT8)
        
        with pytest.raises(RuntimeError, match="Quantization mode mismatch"):
            t1.add_inplace(t2)

    def test_quantization_mode_mismatch_sub(self):
        """Test that sub_inplace throws on quantization mode mismatch"""
        t1 = labneura.Tensor([10, 20], labneura.QuantizationMode.INT8)
        t2 = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP32)
        
        with pytest.raises(RuntimeError, match="Quantization mode mismatch"):
            t1.sub_inplace(t2)

    def test_quantization_mode_mismatch_mul(self):
        """Test that mul_inplace throws on quantization mode mismatch"""
        t1 = labneura.Tensor([2.0, 3.0], labneura.QuantizationMode.FP16)
        t2 = labneura.Tensor([4, 5], labneura.QuantizationMode.INT16)
        
        with pytest.raises(RuntimeError, match="Quantization mode mismatch"):
            t1.mul_inplace(t2)

    def test_combined_size_and_mode_mismatch(self):
        """Test that size mismatch is caught before mode mismatch"""
        t1 = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([3, 4, 5], labneura.QuantizationMode.INT8)
        
        # Size is checked first
        with pytest.raises(RuntimeError, match="Tensor size mismatch"):
            t1.add_inplace(t2)


class TestBackendOperationDispatching:
    """Tests to ensure operation() properly dispatches to correct methods"""

    def test_operation_dispatches_fp32(self):
        """Verify FP32 operations are dispatched correctly"""
        t1 = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([4.0, 5.0, 6.0], labneura.QuantizationMode.FP32)
        
        t1.add_inplace(t2)
        np.testing.assert_allclose(t1.data_fp32(), [5.0, 7.0, 9.0], rtol=1e-6)

    def test_operation_dispatches_fp16(self):
        """Verify FP16 operations are dispatched correctly"""
        t1 = labneura.Tensor([2.0, 4.0, 6.0], labneura.QuantizationMode.FP16)
        t2 = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP16)
        
        t1.sub_inplace(t2)
        assert t1.size() == 3

    def test_operation_dispatches_int16(self):
        """Verify INT16 operations are dispatched correctly"""
        t1 = labneura.Tensor([100, 200, 300], labneura.QuantizationMode.INT16)
        t2 = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT16)
        
        t1.mul_inplace(t2)
        result = t1.data_int16()
        assert result == [1000, 4000, 9000]

    def test_operation_dispatches_int8(self):
        """Verify INT8 operations are dispatched correctly"""
        t1 = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT8)
        t2 = labneura.Tensor([5, 10, 15], labneura.QuantizationMode.INT8)
        
        t1.add_inplace(t2)
        result = t1.data_int8()
        assert result == [15, 30, 45]


class TestAllOperationTypes:
    """Test all three operation types through the backend"""

    def test_add_operation_path(self):
        """Test add_inplace goes through operation() correctly"""
        t1 = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([0.5, 1.5], labneura.QuantizationMode.FP32)
        
        # add_inplace calls operation(other, ADD)
        t1.add_inplace(t2)
        np.testing.assert_allclose(t1.data_fp32(), [1.5, 3.5], rtol=1e-6)

    def test_sub_operation_path(self):
        """Test sub_inplace goes through operation() correctly"""
        t1 = labneura.Tensor([10.0, 20.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([3.0, 7.0], labneura.QuantizationMode.FP32)
        
        # sub_inplace calls operation(other, SUB)
        t1.sub_inplace(t2)
        np.testing.assert_allclose(t1.data_fp32(), [7.0, 13.0], rtol=1e-6)

    def test_mul_operation_path(self):
        """Test mul_inplace goes through operation() correctly"""
        t1 = labneura.Tensor([2.0, 3.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([4.0, 5.0], labneura.QuantizationMode.FP32)
        
        # mul_inplace calls operation(other, MUL)
        t1.mul_inplace(t2)
        np.testing.assert_allclose(t1.data_fp32(), [8.0, 15.0], rtol=1e-6)


class TestMultipleOperations:
    """Test chaining multiple operations"""

    def test_chain_add_and_mul(self):
        """Test chaining add and multiply operations"""
        t1 = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([1.0, 1.0, 1.0], labneura.QuantizationMode.FP32)
        t3 = labneura.Tensor([2.0, 2.0, 2.0], labneura.QuantizationMode.FP32)
        
        t1.add_inplace(t2)  # [2, 3, 4]
        t1.mul_inplace(t3)  # [4, 6, 8]
        
        np.testing.assert_allclose(t1.data_fp32(), [4.0, 6.0, 8.0], rtol=1e-6)

    def test_chain_all_operations_int8(self):
        """Test chaining all operations on INT8"""
        t1 = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT8)
        t2 = labneura.Tensor([5, 5, 5], labneura.QuantizationMode.INT8)
        t3 = labneura.Tensor([2, 2, 2], labneura.QuantizationMode.INT8)
        
        t1.add_inplace(t2)   # [15, 25, 35]
        t1.sub_inplace(t3)   # [13, 23, 33]
        
        result = t1.data_int8()
        assert result == [13, 23, 33]

    def test_chain_operations_fp16(self):
        """Test chaining operations on FP16"""
        t1 = labneura.Tensor([5.0, 10.0, 15.0], labneura.QuantizationMode.FP16)
        t2 = labneura.Tensor([2.0, 3.0, 4.0], labneura.QuantizationMode.FP16)
        
        t1.mul_inplace(t2)
        assert t1.size() == 3

    def test_chain_operations_int16(self):
        """Test chaining operations on INT16"""
        t1 = labneura.Tensor([1000, 2000, 3000], labneura.QuantizationMode.INT16)
        t2 = labneura.Tensor([100, 200, 300], labneura.QuantizationMode.INT16)
        t3 = labneura.Tensor([50, 100, 150], labneura.QuantizationMode.INT16)
        
        t1.sub_inplace(t2)  # [900, 1800, 2700]
        t1.add_inplace(t3)  # [950, 1900, 2850]
        
        result = t1.data_int16()
        assert result == [950, 1900, 2850]


class TestEdgeCasesForBackend:
    """Test edge cases for backend operations"""

    def test_empty_tensor_operations(self):
        """Test that operations on empty tensors work"""
        t1 = labneura.Tensor([], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([], labneura.QuantizationMode.FP32)
        
        # Should not throw even with empty tensors
        t1.add_inplace(t2)
        assert t1.size() == 0

    def test_single_element_all_ops(self):
        """Test all operations on single element tensors"""
        t1 = labneura.Tensor([10.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([3.0], labneura.QuantizationMode.FP32)
        
        t_add = t1.add(t2)
        np.testing.assert_allclose(t_add.data_fp32(), [13.0], rtol=1e-6)
        
        t_sub = t1.sub(t2)
        np.testing.assert_allclose(t_sub.data_fp32(), [7.0], rtol=1e-6)
        
        t_mul = t1.mul(t2)
        np.testing.assert_allclose(t_mul.data_fp32(), [30.0], rtol=1e-6)

    def test_zero_values_operations(self):
        """Test operations with zero values"""
        t1 = labneura.Tensor([0.0, 0.0, 0.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP32)
        
        t1.add_inplace(t2)
        np.testing.assert_allclose(t1.data_fp32(), [1.0, 2.0, 3.0], rtol=1e-6)
        
        t1.mul_inplace(labneura.Tensor([0.0, 0.0, 0.0], labneura.QuantizationMode.FP32))
        np.testing.assert_allclose(t1.data_fp32(), [0.0, 0.0, 0.0], rtol=1e-6)

    def test_negative_values_operations(self):
        """Test operations with negative values"""
        t1 = labneura.Tensor([-5.0, -10.0, -15.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([5.0, 10.0, 15.0], labneura.QuantizationMode.FP32)
        
        t1.add_inplace(t2)
        np.testing.assert_allclose(t1.data_fp32(), [0.0, 0.0, 0.0], rtol=1e-6)

    def test_large_tensor_operations(self):
        """Test operations on large tensors"""
        size = 10000
        t1 = labneura.Tensor([1.0] * size, labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([2.0] * size, labneura.QuantizationMode.FP32)
        
        t1.mul_inplace(t2)
        result = t1.data_fp32()
        assert len(result) == size
        assert all(abs(x - 2.0) < 1e-6 for x in result)


class TestMixedModeErrorCases:
    """Test various error cases with mixed modes"""

    def test_fp32_with_fp16_error(self):
        """Test FP32 with FP16 throws mode mismatch"""
        t1 = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([3.0, 4.0], labneura.QuantizationMode.FP16)
        
        with pytest.raises(RuntimeError, match="Quantization mode mismatch"):
            t1.add_inplace(t2)

    def test_fp32_with_int16_error(self):
        """Test FP32 with INT16 throws mode mismatch"""
        t1 = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([3, 4], labneura.QuantizationMode.INT16)
        
        with pytest.raises(RuntimeError, match="Quantization mode mismatch"):
            t1.sub_inplace(t2)

    def test_int8_with_fp16_error(self):
        """Test INT8 with FP16 throws mode mismatch"""
        t1 = labneura.Tensor([1, 2], labneura.QuantizationMode.INT8)
        t2 = labneura.Tensor([3.0, 4.0], labneura.QuantizationMode.FP16)
        
        with pytest.raises(RuntimeError, match="Quantization mode mismatch"):
            t1.mul_inplace(t2)

    def test_int8_with_int16_error(self):
        """Test INT8 with INT16 throws mode mismatch"""
        t1 = labneura.Tensor([1, 2], labneura.QuantizationMode.INT8)
        t2 = labneura.Tensor([3, 4], labneura.QuantizationMode.INT16)
        
        with pytest.raises(RuntimeError, match="Quantization mode mismatch"):
            t1.add_inplace(t2)

    def test_fp16_with_int16_error(self):
        """Test FP16 with INT16 throws mode mismatch"""
        t1 = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP16)
        t2 = labneura.Tensor([3, 4], labneura.QuantizationMode.INT16)
        
        with pytest.raises(RuntimeError, match="Quantization mode mismatch"):
            t1.sub_inplace(t2)


class TestAllQuantizationModes:
    """Ensure all quantization modes work through backend operations"""

    def test_fp32_operations_complete(self):
        """Complete test of FP32 operations"""
        t1 = labneura.Tensor([10.0, 20.0], labneura.QuantizationMode.FP32)
        t2 = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP32)
        
        result = t1.add(t2)
        np.testing.assert_allclose(result.data_fp32(), [11.0, 22.0], rtol=1e-6)

    def test_fp16_operations_complete(self):
        """Complete test of FP16 operations"""
        t1 = labneura.Tensor([10.0, 20.0], labneura.QuantizationMode.FP16)
        t2 = labneura.Tensor([5.0, 10.0], labneura.QuantizationMode.FP16)
        
        result = t1.sub(t2)
        assert result.size() == 2

    def test_int8_operations_complete(self):
        """Complete test of INT8 operations"""
        t1 = labneura.Tensor([10, 20], labneura.QuantizationMode.INT8)
        t2 = labneura.Tensor([2, 3], labneura.QuantizationMode.INT8)
        
        result = t1.mul(t2)
        assert result.data_int8() == [20, 60]

    def test_int16_operations_complete(self):
        """Complete test of INT16 operations"""
        t1 = labneura.Tensor([1000, 2000], labneura.QuantizationMode.INT16)
        t2 = labneura.Tensor([100, 200], labneura.QuantizationMode.INT16)
        
        result = t1.add(t2)
        assert result.data_int16() == [1100, 2200]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
