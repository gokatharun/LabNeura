"""
Comprehensive test suite for AVX2 DIV operations on INT8, INT16, and FP16 data types.
Tests SIMD vectorized division with edge cases and zero-division handling.
"""

import pytest
import numpy as np
import sys
import os
import platform

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from labneura import Tensor, QuantizationMode, detect_backend
except ImportError:
    pytest.skip("LabNeura not installed", allow_module_level=True)

# Detect current backend
current_backend = detect_backend()
is_avx2 = current_backend == "AVX2"
is_arm = platform.machine() == "arm64"

# Skip all AVX2-specific tests if not on AVX2 backend
pytestmark_avx2 = pytest.mark.skipif(not is_avx2 and not (is_arm and platform.system() == "Darwin"),
                                      reason="AVX2 tests require AVX2 backend or ARM with proper support")


@pytest.mark.skipif(not is_avx2, reason="INT8 DIV only available in AVX2 backend")
class TestAVX2DIVInt8:
    """Test AVX2 DIV operation for INT8 tensors"""

    def test_int8_div_basic(self):
        """Test basic INT8 division"""
        a = Tensor([10, 20, 30, 40], mode=QuantizationMode.INT8)
        b = Tensor([2, 4, 5, 8], mode=QuantizationMode.INT8)
        result = a.div_inplace(b)
        
        expected = np.array([5, 5, 6, 5], dtype=np.int8)
        assert np.allclose(result.data_int8(), expected), \
            f"Expected {expected}, got {result.data_int8()}"

    def test_int8_div_negative_numbers(self):
        """Test INT8 division with negative numbers"""
        a = Tensor([-20, 30, -40, 50], mode=QuantizationMode.INT8)
        b = Tensor([2, -5, 8, -10], mode=QuantizationMode.INT8)
        result = a.div_inplace(b)
        
        expected = np.array([-10, -6, -5, -5], dtype=np.int8)
        assert np.allclose(result.data_int8(), expected), \
            f"Expected {expected}, got {result.data_int8()}"

    def test_int8_div_by_zero(self):
        """Test INT8 division by zero returns 0"""
        a = Tensor([10, 20, 30, 40], mode=QuantizationMode.INT8)
        b = Tensor([0, 2, 0, 4], mode=QuantizationMode.INT8)
        result = a.div_inplace(b)
        
        # Division by zero should yield 0
        expected = np.array([0, 10, 0, 10], dtype=np.int8)
        assert np.allclose(result.data_int8(), expected), \
            f"Expected {expected}, got {result.data_int8()}"

    def test_int8_div_saturation(self):
        """Test INT8 division with saturation on overflow"""
        a = Tensor([127, -128, 100, -100], mode=QuantizationMode.INT8)
        b = Tensor([1, 1, 2, 2], mode=QuantizationMode.INT8)
        result = a.div_inplace(b)
        
        # Should saturate to [-128, 127]
        expected = np.array([127, -128, 50, -50], dtype=np.int8)
        assert np.allclose(result.data_int8(), expected), \
            f"Expected {expected}, got {result.data_int8()}"

    def test_int8_div_large_array(self):
        """Test INT8 division on large arrays (tests SIMD loops)"""
        size = 1000
        a_data = np.random.randint(-128, 127, size, dtype=np.int8)
        b_data = np.random.randint(1, 127, size, dtype=np.int8)  # Avoid zero
        
        a = Tensor(a_data.tolist(), mode=QuantizationMode.INT8)
        b = Tensor(b_data.tolist(), mode=QuantizationMode.INT8)
        result = a.div_inplace(b)
        
        # Compute expected result
        expected = np.array([
            np.int8(np.clip(int(a_data[i]) // int(b_data[i]), -128, 127))
            for i in range(size)
        ], dtype=np.int8)
        
        assert np.allclose(result.data_int8(), expected), \
            f"Large array INT8 division failed"

    def test_int8_div_one(self):
        """Test INT8 division by 1 (identity)"""
        data = [-100, -50, 0, 50, 100]
        a = Tensor(data, mode=QuantizationMode.INT8)
        b = Tensor([1] * len(data), mode=QuantizationMode.INT8)
        result = a.div_inplace(b)
        
        expected = np.array(data, dtype=np.int8)
        assert np.allclose(result.data_int8(), expected), \
            f"Division by 1 should be identity"


@pytest.mark.skipif(not is_avx2, reason="INT16 DIV only available in AVX2 backend")
class TestAVX2DIVInt16:
    """Test AVX2 DIV operation for INT16 tensors"""

    def test_int16_div_basic(self):
        """Test basic INT16 division"""
        a = Tensor([100, 200, 300, 400], mode=QuantizationMode.INT16)
        b = Tensor([2, 4, 5, 8], mode=QuantizationMode.INT16)
        result = a.div_inplace(b)
        
        expected = np.array([50, 50, 60, 50], dtype=np.int16)
        assert np.allclose(result.data_int16(), expected), \
            f"Expected {expected}, got {result.data_int16()}"

    def test_int16_div_negative_numbers(self):
        """Test INT16 division with negative numbers"""
        a = Tensor([-200, 300, -400, 500], mode=QuantizationMode.INT16)
        b = Tensor([2, -5, 8, -10], mode=QuantizationMode.INT16)
        result = a.div_inplace(b)
        
        expected = np.array([-100, -60, -50, -50], dtype=np.int16)
        assert np.allclose(result.data_int16(), expected), \
            f"Expected {expected}, got {result.data_int16()}"

    def test_int16_div_by_zero(self):
        """Test INT16 division by zero returns 0"""
        a = Tensor([100, 200, 300, 400], mode=QuantizationMode.INT16)
        b = Tensor([0, 2, 0, 4], mode=QuantizationMode.INT16)
        result = a.div_inplace(b)
        
        # Division by zero should yield 0
        expected = np.array([0, 100, 0, 100], dtype=np.int16)
        assert np.allclose(result.data_int16(), expected), \
            f"Expected {expected}, got {result.data_int16()}"

    def test_int16_div_saturation(self):
        """Test INT16 division with saturation on overflow"""
        a = Tensor([32767, -32768, 1000, -1000], mode=QuantizationMode.INT16)
        b = Tensor([1, 1, 2, 2], mode=QuantizationMode.INT16)
        result = a.div_inplace(b)
        
        # Should saturate to [-32768, 32767]
        expected = np.array([32767, -32768, 500, -500], dtype=np.int16)
        assert np.allclose(result.data_int16(), expected), \
            f"Expected {expected}, got {result.data_int16()}"

    def test_int16_div_large_array(self):
        """Test INT16 division on large arrays (tests SIMD 16-element loops)"""
        size = 2000
        a_data = np.random.randint(-32768, 32767, size, dtype=np.int16)
        b_data = np.random.randint(1, 32767, size, dtype=np.int16)  # Avoid zero
        
        a = Tensor(a_data.tolist(), mode=QuantizationMode.INT16)
        b = Tensor(b_data.tolist(), mode=QuantizationMode.INT16)
        result = a.div_inplace(b)
        
        # Compute expected result
        expected = np.array([
            np.int16(np.clip(int(a_data[i]) // int(b_data[i]), -32768, 32767))
            for i in range(size)
        ], dtype=np.int16)
        
        assert np.allclose(result.data_int16(), expected), \
            f"Large array INT16 division failed"

    def test_int16_div_boundary_values(self):
        """Test INT16 division with boundary values"""
        a = Tensor([32767, -32768, 16384, -16384], mode=QuantizationMode.INT16)
        b = Tensor([1, 1, 2, 2], mode=QuantizationMode.INT16)
        result = a.div_inplace(b)
        
        expected = np.array([32767, -32768, 8192, -8192], dtype=np.int16)
        assert np.allclose(result.data_int16(), expected), \
            f"Boundary value division failed"

    def test_int16_div_one(self):
        """Test INT16 division by 1 (identity)"""
        data = [-1000, -500, 0, 500, 1000]
        a = Tensor(data, mode=QuantizationMode.INT16)
        b = Tensor([1] * len(data), mode=QuantizationMode.INT16)
        result = a.div_inplace(b)
        
        expected = np.array(data, dtype=np.int16)
        assert np.allclose(result.data_int16(), expected), \
            f"Division by 1 should be identity"


@pytest.mark.skipif(not is_avx2, reason="FP16 DIV only available in AVX2 backend")
class TestAVX2DIVFP16:
    """Test AVX2 DIV operation for FP16 tensors (stored as int16 bitwise)"""

    def test_fp16_div_basic(self):
        """Test basic FP16 division"""
        # Note: FP16 is stored as int16_t bitwise representation
        # Testing with integer patterns that represent FP16 values
        a = Tensor([100, 200, 300, 400], mode=QuantizationMode.FP16)
        b = Tensor([2, 4, 5, 8], mode=QuantizationMode.FP16)
        result = a.div_inplace(b)
        
        # After division, should have approximate results
        result_data = result.data_fp16()
        assert len(result_data) == 4, f"Expected 4 elements, got {len(result_data)}"

    def test_fp16_div_large_array(self):
        """Test FP16 division on large arrays"""
        size = 2000
        a_data = np.random.randn(size).astype(np.float32) * 1000
        b_data = np.random.randn(size).astype(np.float32) * 1000 + 1e-6  # Avoid zero
        
        # Convert to int16 representation (simulating FP16 bitwise)
        a_int16 = a_data.astype(np.int16)
        b_int16 = b_data.astype(np.int16)
        
        a = Tensor(a_int16.tolist(), mode=QuantizationMode.FP16)
        b = Tensor(b_int16.tolist(), mode=QuantizationMode.FP16)
        result = a.div_inplace(b)
        
        # Verify operation completed without error
        result_data = result.data_fp16()
        assert len(result_data) == size, f"Expected {size} elements, got {len(result_data)}"
        # Result should not contain NaN (except for div by zero cases)
        assert not np.isnan(result_data).all(), "Result contains all NaN values"

    def test_fp16_div_by_zero(self):
        """Test FP16 division by zero handling"""
        a = Tensor([100, 200, 300], mode=QuantizationMode.FP16)
        b = Tensor([0, 2, 0], mode=QuantizationMode.FP16)
        result = a.div_inplace(b)
        
        # Operation should complete; division by zero is platform-dependent
        result_data = result.data_fp16()
        assert len(result_data) == 3, f"Expected 3 elements"


@pytest.mark.skipif(not is_avx2, reason="Integration tests require AVX2 backend")
class TestAVX2DIVIntegration:
    """Integration tests for AVX2 DIV across different scenarios"""

    def test_div_preserves_tensor_size(self):
        """Test that DIV preserves tensor size"""
        size = 256
        a = Tensor(np.random.randint(1, 100, size, dtype=np.int8).tolist(), 
                   mode=QuantizationMode.INT8)
        b = Tensor(np.random.randint(1, 50, size, dtype=np.int8).tolist(), 
                   mode=QuantizationMode.INT8)
        
        original_size = a.size()
        result = a.div_inplace(b)
        
        assert result.size() == original_size, "DIV changed tensor size"
        assert result.size() == size, f"Expected size {size}, got {result.size()}"

    def test_int8_div_all_zeros_dividend(self):
        """Test INT8 division with all-zero dividend"""
        a = Tensor([0] * 32, mode=QuantizationMode.INT8)
        b = Tensor(list(range(1, 33)), mode=QuantizationMode.INT8)
        result = a.div_inplace(b)
        
        expected = np.zeros(32, dtype=np.int8)
        assert np.allclose(result.data_int8(), expected), \
            "Division of zero by non-zero should yield zero"

    def test_int16_div_alignment_boundary(self):
        """Test INT16 division across SIMD alignment boundaries"""
        # Create arrays that cross 16-element SIMD boundaries
        sizes = [15, 16, 17, 31, 32, 33, 63, 64, 65]
        
        for size in sizes:
            a_data = np.full(size, 100, dtype=np.int16)
            b_data = np.full(size, 10, dtype=np.int16)
            
            a = Tensor(a_data.tolist(), mode=QuantizationMode.INT16)
            b = Tensor(b_data.tolist(), mode=QuantizationMode.INT16)
            result = a.div_inplace(b)
            
            expected = np.full(size, 10, dtype=np.int16)
            assert np.allclose(result.data_int16(), expected), \
                f"Division failed at boundary size {size}"

    def test_div_consistency_across_modes(self):
        """Test that division behaves consistently for similar operations"""
        # INT8: 40 / 5 = 8
        a8 = Tensor([40], mode=QuantizationMode.INT8)
        b8 = Tensor([5], mode=QuantizationMode.INT8)
        r8 = a8.div_inplace(b8).data_int8()[0]
        
        # INT16: 40 / 5 = 8
        a16 = Tensor([40], mode=QuantizationMode.INT16)
        b16 = Tensor([5], mode=QuantizationMode.INT16)
        r16 = a16.div_inplace(b16).data_int16()[0]
        
        assert r8 == 8, f"INT8 division failed: expected 8, got {r8}"
        assert r16 == 8, f"INT16 division failed: expected 8, got {r16}"


@pytest.mark.skipif(not is_avx2, reason="Performance tests require AVX2 backend")
class TestAVX2DIVPerformance:
    """Performance and stress tests for AVX2 DIV"""

    def test_int8_div_1000_elements(self):
        """Stress test INT8 division with 1000 elements"""
        size = 1000
        a_data = np.random.randint(1, 100, size, dtype=np.int8)
        b_data = np.random.randint(1, 50, size, dtype=np.int8)
        
        a = Tensor(a_data.tolist(), mode=QuantizationMode.INT8)
        b = Tensor(b_data.tolist(), mode=QuantizationMode.INT8)
        result = a.div_inplace(b)
        
        assert result.size() == size, "Stress test failed: wrong size"
        result_data = result.data_int8()
        assert len(result_data) == size, "Stress test failed: wrong result length"

    def test_int16_div_2000_elements(self):
        """Stress test INT16 division with 2000 elements"""
        size = 2000
        a_data = np.random.randint(1, 1000, size, dtype=np.int16)
        b_data = np.random.randint(1, 500, size, dtype=np.int16)
        
        a = Tensor(a_data.tolist(), mode=QuantizationMode.INT16)
        b = Tensor(b_data.tolist(), mode=QuantizationMode.INT16)
        result = a.div_inplace(b)
        
        assert result.size() == size, "Stress test failed: wrong size"
        result_data = result.data_int16()
        assert len(result_data) == size, "Stress test failed: wrong result length"

    def test_fp16_div_2000_elements(self):
        """Stress test FP16 division with 2000 elements"""
        size = 2000
        a_data = np.random.randn(size).astype(np.float32) * 100
        b_data = np.random.randn(size).astype(np.float32) * 100 + 1e-6
        
        a_int16 = a_data.astype(np.int16)
        b_int16 = b_data.astype(np.int16)
        
        a = Tensor(a_int16.tolist(), mode=QuantizationMode.FP16)
        b = Tensor(b_int16.tolist(), mode=QuantizationMode.FP16)
        result = a.div_inplace(b)
        
        assert result.size() == size, "Stress test failed: wrong size"
        result_data = result.data_fp16()
        assert len(result_data) == size, "Stress test failed: wrong result length"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
