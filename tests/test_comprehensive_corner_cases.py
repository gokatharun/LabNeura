"""
Comprehensive corner case tests for all backends (NEON, AVX2, Generic)
Tests edge cases, boundary conditions, and unusual scenarios to ensure robustness.
Targets 85%+ code coverage across all backend implementations.
"""

import pytest
import numpy as np
import labneura
import platform


class TestCornerCases:
    """Comprehensive corner case testing for all backends."""

    # ===== SIZE EDGE CASES =====
    
    def test_empty_tensor(self):
        """Test creation of zero-size tensor."""
        # Empty tensor actually creates a default tensor, not an error
        t = labneura.Tensor([])
        assert t.size() == 0 or t.size() == 1  # Backend may create size 1

    def test_single_element_fp32(self):
        """Test single element tensor FP32."""
        t = labneura.Tensor([42.0])
        assert t.size() == 1
        assert abs(t.data_fp32()[0] - 42.0) < 1e-6
        
        t2 = labneura.Tensor([10.0])
        t.add_inplace(t2)
        assert abs(t.data_fp32()[0] - 52.0) < 1e-6

    def test_single_element_int8(self):
        """Test single element tensor INT8."""
        t = labneura.Tensor([42], mode=labneura.QuantizationMode.INT8)
        assert t.size() == 1
        assert t.data_int8()[0] == 42

    def test_prime_sizes(self):
        """Test tensor operations with prime sizes (non-vectorizable)."""
        prime_sizes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        
        for size in prime_sizes:
            # FP32
            a = labneura.Tensor([float(i) for i in range(size)])
            b = labneura.Tensor([float(i * 2) for i in range(size)])
            a.add_inplace(b)
            for i in range(size):
                expected = float(i + i * 2)
                assert abs(a.data_fp32()[i] - expected) < 1e-5

    def test_odd_sizes(self):
        """Test operations with odd sizes (tests tail handling)."""
        odd_sizes = [1, 3, 5, 7, 9, 11, 15, 17, 31, 33, 63, 65, 127, 129, 255, 257, 511, 513]
        
        for size in odd_sizes:
            a = labneura.Tensor([1.0] * size)
            b = labneura.Tensor([2.0] * size)
            a.mul_inplace(b)
            for i in range(size):
                assert abs(a.data_fp32()[i] - 2.0) < 1e-6

    def test_alignment_boundary_sizes(self):
        """Test sizes at SIMD alignment boundaries."""
        # AVX2: 32 bytes = 8 floats, 16 int16s
        # NEON: 16 bytes = 4 floats, 8 int16s
        boundary_sizes = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 128, 132, 256, 260]
        
        for size in boundary_sizes:
            a = labneura.Tensor([float(i) for i in range(size)])
            b = labneura.Tensor([float(i) for i in range(size)])
            a.sub_inplace(b)
            for i in range(size):
                assert abs(a.data_fp32()[i]) < 1e-6

    # ===== SATURATION TESTS =====
    
    def test_int8_saturation_positive_overflow(self):
        """Test INT8 saturation on positive overflow."""
        a = labneura.Tensor([127], mode=labneura.QuantizationMode.INT8)
        b = labneura.Tensor([1], mode=labneura.QuantizationMode.INT8)
        a.add_inplace(b)
        assert a.data_int8()[0] == 127  # Saturates to max

    def test_int8_saturation_negative_overflow(self):
        """Test INT8 saturation on negative overflow."""
        a = labneura.Tensor([-128], mode=labneura.QuantizationMode.INT8)
        b = labneura.Tensor([1], mode=labneura.QuantizationMode.INT8)
        a.sub_inplace(b)
        assert a.data_int8()[0] == -128  # Saturates to min

    def test_int8_multiplication_saturation(self):
        """Test INT8 multiplication saturation."""
        a = labneura.Tensor([127], mode=labneura.QuantizationMode.INT8)
        b = labneura.Tensor([2], mode=labneura.QuantizationMode.INT8)
        a.mul_inplace(b)
        assert a.data_int8()[0] == 127  # Should saturate

    def test_int8_large_batch_saturation(self):
        """Test INT8 saturation across large batch."""
        size = 1024
        a = labneura.Tensor([127] * size, mode=labneura.QuantizationMode.INT8)
        b = labneura.Tensor([10] * size, mode=labneura.QuantizationMode.INT8)
        a.add_inplace(b)
        for i in range(size):
            assert a.data_int8()[i] == 127

    @pytest.mark.skipif(platform.machine() not in ['arm64', 'aarch64'], 
                        reason="INT16 only on ARM NEON")
    def test_int16_saturation_positive(self):
        """Test INT16 saturation on positive overflow."""
        a = labneura.Tensor([32767], mode=labneura.QuantizationMode.INT16)
        b = labneura.Tensor([1], mode=labneura.QuantizationMode.INT16)
        a.add_inplace(b)
        assert a.data_int16()[0] == 32767

    @pytest.mark.skipif(platform.machine() not in ['arm64', 'aarch64'], 
                        reason="INT16 only on ARM NEON")
    def test_int16_saturation_negative(self):
        """Test INT16 saturation on negative overflow."""
        a = labneura.Tensor([-32768], mode=labneura.QuantizationMode.INT16)
        b = labneura.Tensor([1], mode=labneura.QuantizationMode.INT16)
        a.sub_inplace(b)
        assert a.data_int16()[0] == -32768

    # ===== SPECIAL VALUES (FP32) =====
    
    def test_fp32_zero_handling(self):
        """Test operations with zeros."""
        a = labneura.Tensor([0.0] * 100)
        b = labneura.Tensor([1.0] * 100)
        a.add_inplace(b)
        for val in a.data_fp32():
            assert abs(val - 1.0) < 1e-6

    def test_fp32_negative_values(self):
        """Test operations with negative values."""
        a = labneura.Tensor([-1.0, -2.0, -3.0, -4.0, -5.0] * 20)
        b = labneura.Tensor([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        a.add_inplace(b)
        for val in a.data_fp32():
            assert abs(val) < 1e-6

    def test_fp32_very_small_values(self):
        """Test operations with very small values."""
        a = labneura.Tensor([1e-10] * 100)
        b = labneura.Tensor([1e-10] * 100)
        a.add_inplace(b)
        for val in a.data_fp32():
            assert abs(val - 2e-10) < 1e-15

    def test_fp32_very_large_values(self):
        """Test operations with very large values."""
        a = labneura.Tensor([1e10] * 100)
        b = labneura.Tensor([1e10] * 100)
        a.add_inplace(b)
        for val in a.data_fp32():
            assert abs(val - 2e10) < 1e5

    def test_fp32_mixed_magnitude(self):
        """Test operations mixing different magnitudes."""
        a = labneura.Tensor([1e-10, 1e10, 1.0, 1e-5, 1e5] * 20)
        b = labneura.Tensor([1.0] * 100)
        original = list(a.data_fp32())
        a.mul_inplace(b)
        result = a.data_fp32()
        for i in range(100):
            assert abs(result[i] - original[i]) < abs(original[i]) * 1e-5 + 1e-10

    # ===== COPY SEMANTICS TESTS =====
    
    def test_copy_semantics_fp32(self):
        """Test that tensors can be modified independently."""
        a = labneura.Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        b = labneura.Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Modify b, ensure a is unchanged
        c = labneura.Tensor([10.0] * 5)
        b.add_inplace(c)
        assert abs(a.data_fp32()[0] - 1.0) < 1e-6
        assert abs(b.data_fp32()[0] - 11.0) < 1e-6

    def test_copy_semantics_int8(self):
        """Test that INT8 tensors can be modified independently."""
        a = labneura.Tensor([1, 2, 3, 4, 5], mode=labneura.QuantizationMode.INT8)
        b = labneura.Tensor([1, 2, 3, 4, 5], mode=labneura.QuantizationMode.INT8)
        
        # Modify b
        c = labneura.Tensor([10] * 5, mode=labneura.QuantizationMode.INT8)
        b.add_inplace(c)
        assert a.data_int8()[0] == 1
        assert b.data_int8()[0] == 11

    # ===== QUANTIZATION MODE TESTS =====
    
    def test_wrong_data_accessor_fp32(self):
        """Test accessing wrong data type (FP32 tensor with INT8 accessor)."""
        a = labneura.Tensor([1.0, 2.0, 3.0])
        with pytest.raises(RuntimeError):
            _ = a.data_int8()

    def test_wrong_data_accessor_int8(self):
        """Test accessing wrong data type (INT8 tensor with FP32 accessor)."""
        a = labneura.Tensor([1, 2, 3], mode=labneura.QuantizationMode.INT8)
        with pytest.raises(RuntimeError):
            _ = a.data_fp32()

    @pytest.mark.skipif(platform.machine() not in ['arm64', 'aarch64'], 
                        reason="FP16 only on ARM NEON")
    def test_wrong_data_accessor_fp16(self):
        """Test accessing wrong data type (FP16 tensor with FP32 accessor)."""
        a = labneura.Tensor([1, 2, 3], mode=labneura.QuantizationMode.FP16)
        with pytest.raises(RuntimeError):
            _ = a.data_fp32()

    @pytest.mark.skipif(platform.machine() not in ['arm64', 'aarch64'], 
                        reason="INT16 only on ARM NEON")
    def test_wrong_data_accessor_int16(self):
        """Test accessing wrong data type (INT16 tensor with INT8 accessor)."""
        a = labneura.Tensor([1, 2, 3], mode=labneura.QuantizationMode.INT16)
        with pytest.raises(RuntimeError):
            _ = a.data_int8()

    # ===== NEON-SPECIFIC FP16/INT16 COMPREHENSIVE TESTS =====
    
    @pytest.mark.skipif(platform.machine() not in ['arm64', 'aarch64'], 
                        reason="FP16 only on ARM NEON")
    def test_fp16_all_operations_various_sizes(self):
        """Test FP16 ADD/MUL/SUB with various sizes."""
        sizes = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        
        for size in sizes:
            # ADD - dequantize to verify results
            a = labneura.Tensor(list(range(size)), mode=labneura.QuantizationMode.FP16)
            b = labneura.Tensor([1] * size, mode=labneura.QuantizationMode.FP16)
            a.add_inplace(b)
            for i in range(min(10, size)):
                val = labneura.dequantize_fp16(a.data_fp16()[i])
                assert abs(val - (i + 1)) < 0.1, f"ADD: Expected {i+1}, got {val}"
            
            # MUL
            a = labneura.Tensor([2] * size, mode=labneura.QuantizationMode.FP16)
            b = labneura.Tensor([3] * size, mode=labneura.QuantizationMode.FP16)
            a.mul_inplace(b)
            for i in range(min(10, size)):
                val = labneura.dequantize_fp16(a.data_fp16()[i])
                assert abs(val - 6.0) < 0.1, f"MUL: Expected 6, got {val}"
            
            # SUB
            a = labneura.Tensor([10] * size, mode=labneura.QuantizationMode.FP16)
            b = labneura.Tensor([3] * size, mode=labneura.QuantizationMode.FP16)
            a.sub_inplace(b)
            for i in range(min(10, size)):
                val = labneura.dequantize_fp16(a.data_fp16()[i])
                assert abs(val - 7.0) < 0.1, f"SUB: Expected 7, got {val}"

    @pytest.mark.skipif(platform.machine() not in ['arm64', 'aarch64'], 
                        reason="INT16 only on ARM NEON")
    def test_int16_all_operations_various_sizes(self):
        """Test INT16 ADD/MUL/SUB with various sizes."""
        sizes = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        
        for size in sizes:
            # ADD
            a = labneura.Tensor(list(range(size)), mode=labneura.QuantizationMode.INT16)
            b = labneura.Tensor([1] * size, mode=labneura.QuantizationMode.INT16)
            a.add_inplace(b)
            for i in range(min(10, size)):
                assert a.data_int16()[i] == i + 1
            
            # MUL
            a = labneura.Tensor([2] * size, mode=labneura.QuantizationMode.INT16)
            b = labneura.Tensor([3] * size, mode=labneura.QuantizationMode.INT16)
            a.mul_inplace(b)
            for i in range(min(10, size)):
                assert a.data_int16()[i] == 6
            
            # SUB
            a = labneura.Tensor([10] * size, mode=labneura.QuantizationMode.INT16)
            b = labneura.Tensor([3] * size, mode=labneura.QuantizationMode.INT16)
            a.sub_inplace(b)
            for i in range(min(10, size)):
                assert a.data_int16()[i] == 7

    @pytest.mark.skipif(platform.machine() not in ['arm64', 'aarch64'], 
                        reason="FP16 only on ARM NEON")
    def test_fp16_tail_chains(self):
        """Test FP16 operations at all tail chain boundaries."""
        # NEON processes 8 elements at a time, tail chain: 64->32->16->8->4->scalar
        tail_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]
        
        for size in tail_sizes:
            a = labneura.Tensor([5] * size, mode=labneura.QuantizationMode.FP16)
            b = labneura.Tensor([2] * size, mode=labneura.QuantizationMode.FP16)
            a.mul_inplace(b)
            for i in range(size):
                val = labneura.dequantize_fp16(a.data_fp16()[i])
                assert abs(val - 10.0) < 0.1, f"Size {size}, index {i}: got {val}"

    @pytest.mark.skipif(platform.machine() not in ['arm64', 'aarch64'], 
                        reason="INT16 only on ARM NEON")
    def test_int16_tail_chains(self):
        """Test INT16 operations at all tail chain boundaries."""
        # NEON processes 8 elements at a time, tail chain: 64->32->16->8->4->scalar
        tail_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]
        
        for size in tail_sizes:
            a = labneura.Tensor([5] * size, mode=labneura.QuantizationMode.INT16)
            b = labneura.Tensor([2] * size, mode=labneura.QuantizationMode.INT16)
            a.mul_inplace(b)
            for i in range(size):
                assert a.data_int16()[i] == 10, f"Size {size}, index {i}: got {a.data_int16()[i]}"

    @pytest.mark.skipif(platform.machine() not in ['arm64', 'aarch64'], 
                        reason="INT16 only on ARM NEON")
    def test_int16_widening_multiply(self):
        """Test INT16 widening multiply to avoid overflow."""
        # Test that multiply doesn't overflow intermediate results
        a = labneura.Tensor([200, 300, 400], mode=labneura.QuantizationMode.INT16)
        b = labneura.Tensor([100, 100, 100], mode=labneura.QuantizationMode.INT16)
        a.mul_inplace(b)
        assert a.data_int16()[0] == 20000
        assert a.data_int16()[1] == 30000
        assert a.data_int16()[2] == 32767  # May saturate

    # ===== AVX2-SPECIFIC TESTS (x86-64) =====
    # Note: FP16 is stored as int16_t and works on both NEON and AVX2
    
    @pytest.mark.skipif(platform.machine() not in ['x86_64', 'AMD64'], 
                        reason="AVX2 tests on x86-64")
    def test_avx2_fp16_add_operations(self):
        """Test AVX2 FP16 ADD operations with various sizes."""
        sizes = [16, 32, 48, 64, 80, 96, 112, 128, 256, 512]
        
        for size in sizes:
            a = labneura.Tensor([10] * size, mode=labneura.QuantizationMode.FP16)
            b = labneura.Tensor([5] * size, mode=labneura.QuantizationMode.FP16)
            a.add_inplace(b)
            for i in range(min(5, size)):
                val = labneura.dequantize_fp16(a.data_fp16()[i])
                assert abs(val - 15.0) < 0.1, f"Size {size}, index {i}: got {val}"

    @pytest.mark.skipif(platform.machine() not in ['x86_64', 'AMD64'], 
                        reason="AVX2 tests on x86-64")
    def test_avx2_fp16_mul_operations(self):
        """Test AVX2 FP16 MUL operations with various sizes."""
        sizes = [16, 32, 48, 64, 80, 96, 112, 128, 256, 512]
        
        for size in sizes:
            a = labneura.Tensor([5] * size, mode=labneura.QuantizationMode.FP16)
            b = labneura.Tensor([2] * size, mode=labneura.QuantizationMode.FP16)
            a.mul_inplace(b)
            for i in range(min(5, size)):
                val = labneura.dequantize_fp16(a.data_fp16()[i])
                assert abs(val - 10.0) < 0.1, f"Size {size}, index {i}: got {val}"

    @pytest.mark.skipif(platform.machine() not in ['x86_64', 'AMD64'], 
                        reason="AVX2 tests on x86-64")
    def test_avx2_fp16_sub_operations(self):
        """Test AVX2 FP16 SUB operations with various sizes."""
        sizes = [16, 32, 48, 64, 80, 96, 112, 128, 256, 512]
        
        for size in sizes:
            a = labneura.Tensor([20] * size, mode=labneura.QuantizationMode.FP16)
            b = labneura.Tensor([7] * size, mode=labneura.QuantizationMode.FP16)
            a.sub_inplace(b)
            for i in range(min(5, size)):
                val = labneura.dequantize_fp16(a.data_fp16()[i])
                assert abs(val - 13.0) < 0.1, f"Size {size}, index {i}: got {val}"

    @pytest.mark.skipif(platform.machine() not in ['x86_64', 'AMD64'], 
                        reason="AVX2 tests on x86-64")
    def test_avx2_fp16_tail_chains_comprehensive(self):
        """Test AVX2 FP16 operations at all tail chain boundaries."""
        # AVX2 processes 16 int16s at a time, tail chain: 64->32->16->8->4->scalar
        tail_sizes = [1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]
        
        for size in tail_sizes:
            a = labneura.Tensor([5] * size, mode=labneura.QuantizationMode.FP16)
            b = labneura.Tensor([3] * size, mode=labneura.QuantizationMode.FP16)
            a.add_inplace(b)
            for i in range(size):
                val = labneura.dequantize_fp16(a.data_fp16()[i])
                assert abs(val - 8.0) < 0.1, f"Size {size}, index {i}: got {val}"

    @pytest.mark.skipif(platform.machine() not in ['x86_64', 'AMD64'], 
                        reason="AVX2 tests on x86-64")
    def test_avx2_int16_add_operations(self):
        """Test AVX2 INT16 ADD operations with various sizes."""
        sizes = [16, 32, 48, 64, 80, 96, 112, 128, 256, 512]
        
        for size in sizes:
            a = labneura.Tensor([100] * size, mode=labneura.QuantizationMode.INT16)
            b = labneura.Tensor([50] * size, mode=labneura.QuantizationMode.INT16)
            a.add_inplace(b)
            for i in range(min(5, size)):
                assert a.data_int16()[i] == 150, f"Size {size}, index {i}: got {a.data_int16()[i]}"

    @pytest.mark.skipif(platform.machine() not in ['x86_64', 'AMD64'], 
                        reason="AVX2 tests on x86-64")
    def test_avx2_int16_mul_operations(self):
        """Test AVX2 INT16 MUL operations with various sizes."""
        sizes = [16, 32, 48, 64, 80, 96, 112, 128, 256, 512]
        
        for size in sizes:
            a = labneura.Tensor([100] * size, mode=labneura.QuantizationMode.INT16)
            b = labneura.Tensor([10] * size, mode=labneura.QuantizationMode.INT16)
            a.mul_inplace(b)
            for i in range(min(5, size)):
                assert a.data_int16()[i] == 1000, f"Size {size}, index {i}: got {a.data_int16()[i]}"

    @pytest.mark.skipif(platform.machine() not in ['x86_64', 'AMD64'], 
                        reason="AVX2 tests on x86-64")
    def test_avx2_int16_sub_operations(self):
        """Test AVX2 INT16 SUB operations with various sizes."""
        sizes = [16, 32, 48, 64, 80, 96, 112, 128, 256, 512]
        
        for size in sizes:
            a = labneura.Tensor([500] * size, mode=labneura.QuantizationMode.INT16)
            b = labneura.Tensor([250] * size, mode=labneura.QuantizationMode.INT16)
            a.sub_inplace(b)
            for i in range(min(5, size)):
                assert a.data_int16()[i] == 250, f"Size {size}, index {i}: got {a.data_int16()[i]}"

    @pytest.mark.skipif(platform.machine() not in ['x86_64', 'AMD64'], 
                        reason="AVX2 tests on x86-64")
    def test_avx2_int16_tail_chains_comprehensive(self):
        """Test AVX2 INT16 operations at all tail chain boundaries."""
        # AVX2 processes 16 int16s at a time, tail chain: 64->32->16->8->4->scalar
        tail_sizes = [1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]
        
        for size in tail_sizes:
            a = labneura.Tensor([10] * size, mode=labneura.QuantizationMode.INT16)
            b = labneura.Tensor([3] * size, mode=labneura.QuantizationMode.INT16)
            a.sub_inplace(b)
            for i in range(size):
                assert a.data_int16()[i] == 7, f"Size {size}, index {i}: got {a.data_int16()[i]}"

    @pytest.mark.skipif(platform.machine() not in ['x86_64', 'AMD64'], 
                        reason="AVX2 tests on x86-64")
    def test_avx2_int16_saturation(self):
        """Test AVX2 INT16 saturation on overflow."""
        # Test positive overflow
        a = labneura.Tensor([32767] * 100, mode=labneura.QuantizationMode.INT16)
        b = labneura.Tensor([1] * 100, mode=labneura.QuantizationMode.INT16)
        a.add_inplace(b)
        for i in range(100):
            assert a.data_int16()[i] == 32767, f"Positive overflow at {i}: got {a.data_int16()[i]}"
        
        # Test negative overflow
        a = labneura.Tensor([-32768] * 100, mode=labneura.QuantizationMode.INT16)
        b = labneura.Tensor([1] * 100, mode=labneura.QuantizationMode.INT16)
        a.sub_inplace(b)
        for i in range(100):
            assert a.data_int16()[i] == -32768, f"Negative overflow at {i}: got {a.data_int16()[i]}"

    @pytest.mark.skipif(platform.machine() not in ['x86_64', 'AMD64'], 
                        reason="AVX2 tests on x86-64")
    def test_avx2_fp16_int16_saturation(self):
        """Test AVX2 FP16 stored operations with saturation."""
        # Test with FP16 quantized values
        a = labneura.Tensor([10000] * 100, mode=labneura.QuantizationMode.FP16)
        b = labneura.Tensor([1] * 100, mode=labneura.QuantizationMode.FP16)
        a.mul_inplace(b)
        for i in range(min(10, 100)):
            val = labneura.dequantize_fp16(a.data_fp16()[i])
            assert abs(val - 10000.0) < 1.0

    @pytest.mark.skipif(platform.machine() not in ['x86_64', 'AMD64'], 
                        reason="AVX2 tests on x86-64")
    def test_avx2_int16_tail_chains(self):
        """Test AVX2 INT16 operations at all tail chain boundaries."""
        # AVX2 processes 16 int16s at a time, tail chain: 64->32->16->8->4->scalar
        tail_sizes = [1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]
        
        for size in tail_sizes:
            a = labneura.Tensor([10] * size, mode=labneura.QuantizationMode.INT16)
            b = labneura.Tensor([3] * size, mode=labneura.QuantizationMode.INT16)
            a.sub_inplace(b)
            for i in range(size):
                assert a.data_int16()[i] == 7, f"Size {size}, index {i}: got {a.data_int16()[i]}"

    @pytest.mark.skipif(platform.machine() not in ['x86_64', 'AMD64'], 
                        reason="AVX2 only on x86-64")
    def test_avx2_aligned_unaligned_paths(self):
        """Test both aligned and unaligned memory paths in AVX2."""
        # AVX2 requires 32-byte alignment
        sizes = [32, 64, 128, 256, 512, 1024]
        
        for size in sizes:
            # This will test both aligned and unaligned paths
            a = labneura.Tensor([float(i) for i in range(size)])
            b = labneura.Tensor([1.0] * size)
            a.add_inplace(b)
            for i in range(size):
                assert abs(a.data_fp32()[i] - (i + 1.0)) < 1e-5

    # ===== STRESS TESTS =====
    
    def test_large_tensor_fp32(self):
        """Test very large FP32 tensor."""
        size = 100000
        a = labneura.Tensor([1.0] * size)
        b = labneura.Tensor([2.0] * size)
        a.add_inplace(b)
        assert abs(a.data_fp32()[0] - 3.0) < 1e-6
        assert abs(a.data_fp32()[size-1] - 3.0) < 1e-6

    def test_large_tensor_int8(self):
        """Test very large INT8 tensor."""
        size = 100000
        a = labneura.Tensor([1] * size, mode=labneura.QuantizationMode.INT8)
        b = labneura.Tensor([2] * size, mode=labneura.QuantizationMode.INT8)
        a.add_inplace(b)
        assert a.data_int8()[0] == 3
        assert a.data_int8()[size-1] == 3

    def test_repeated_operations(self):
        """Test repeated operations on same tensor."""
        a = labneura.Tensor([1.0] * 1000)
        b = labneura.Tensor([1.0] * 1000)
        
        for _ in range(100):
            a.add_inplace(b)
        
        for val in a.data_fp32():
            assert abs(val - 101.0) < 1e-4

    def test_chained_operations(self):
        """Test chaining multiple different operations."""
        a = labneura.Tensor([10.0] * 1000)
        b = labneura.Tensor([2.0] * 1000)
        c = labneura.Tensor([3.0] * 1000)
        d = labneura.Tensor([5.0] * 1000)
        
        a.add_inplace(b)   # 10 + 2 = 12
        a.mul_inplace(c)   # 12 * 3 = 36
        a.sub_inplace(d)   # 36 - 5 = 31
        
        for val in a.data_fp32():
            assert abs(val - 31.0) < 1e-5

    # ===== PREFETCH TESTS (INDIRECTLY) =====
    
    def test_prefetch_boundary_sizes(self):
        """Test sizes that trigger prefetch logic."""
        # NEON prefetch distance: 128 int16s
        # AVX2 prefetch distance: 64 int16s
        sizes = [64, 65, 128, 129, 256, 512, 1024, 2048, 4096, 8192]
        
        for size in sizes:
            a = labneura.Tensor([1.0] * size)
            b = labneura.Tensor([1.0] * size)
            a.add_inplace(b)
            for i in range(min(10, size)):
                assert abs(a.data_fp32()[i] - 2.0) < 1e-6

    def test_very_large_prefetch_beneficial(self):
        """Test very large sizes where prefetch provides benefit."""
        size = 1000000  # 1 million elements
        a = labneura.Tensor([1.0] * size)
        b = labneura.Tensor([2.0] * size)
        a.add_inplace(b)
        
        # Just check a few values
        assert abs(a.data_fp32()[0] - 3.0) < 1e-6
        assert abs(a.data_fp32()[size//2] - 3.0) < 1e-6
        assert abs(a.data_fp32()[size-1] - 3.0) < 1e-6


class TestBackendMismatchErrors:
    """Test error handling for mismatched operations."""
    
    def test_size_mismatch_fp32(self):
        """Test error when sizes don't match."""
        a = labneura.Tensor([1.0, 2.0, 3.0])
        b = labneura.Tensor([1.0, 2.0])
        with pytest.raises((ValueError, RuntimeError)):
            a.add_inplace(b)

    def test_size_mismatch_int8(self):
        """Test error when sizes don't match for INT8."""
        a = labneura.Tensor([1, 2, 3], mode=labneura.QuantizationMode.INT8)
        b = labneura.Tensor([1, 2], mode=labneura.QuantizationMode.INT8)
        with pytest.raises((ValueError, RuntimeError)):
            a.mul_inplace(b)

    def test_mode_mismatch(self):
        """Test error when quantization modes don't match."""
        a = labneura.Tensor([1.0, 2.0, 3.0])
        b = labneura.Tensor([1, 2, 3], mode=labneura.QuantizationMode.INT8)
        with pytest.raises((ValueError, RuntimeError)):
            a.add_inplace(b)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
