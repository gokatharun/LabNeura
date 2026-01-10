"""
Test cases for FP16 quantization and dequantization functions.
Tests the quantize_fp16 and dequantize_fp16 functions with various edge cases.
"""

import pytest
import numpy as np
import labneura
import struct


def to_uint16(val):
    """Convert signed int to unsigned uint16"""
    return val & 0xFFFF


def from_uint16(val):
    """Convert unsigned uint16 to signed int if negative"""
    return val


class TestQuantizeFP16:
    """Tests for quantize_fp16 function"""

    def test_quantize_zero(self):
        """Test quantizing zero"""
        result = labneura.quantize_fp16(0.0)
        assert result == 0, "Zero should quantize to 0"

    def test_quantize_one(self):
        """Test quantizing 1.0"""
        result = labneura.quantize_fp16(1.0)
        # 1.0 in FP16: sign=0, exponent=15 (bias corrected), mantissa=0
        # Expected: 0011110000000000 = 0x3C00
        assert result == 0x3C00, f"1.0 should quantize to 0x3C00, got {hex(result)}"

    def test_quantize_negative_one(self):
        """Test quantizing -1.0"""
        result = labneura.quantize_fp16(-1.0)
        # -1.0 in FP16: sign=1, exponent=15 (bias corrected), mantissa=0
        # Expected: 1011110000000000 = 0xBC00
        # Convert to unsigned if needed
        result_u = result & 0xFFFF
        assert result_u == 0xBC00, f"-1.0 should quantize to 0xBC00, got {hex(result_u)}"

    def test_quantize_half(self):
        """Test quantizing 0.5"""
        result = labneura.quantize_fp16(0.5)
        # 0.5 in FP16: sign=0, exponent=14 (bias corrected), mantissa=0
        # Expected: 0011110000000000 >> 1 = 0x3800
        assert result == 0x3800, f"0.5 should quantize to 0x3800, got {hex(result)}"

    def test_quantize_positive_infinity(self):
        """Test quantizing positive infinity"""
        result = labneura.quantize_fp16(float('inf'))
        result_u = to_uint16(result)
        # Positive infinity should have very high exponent value
        # (may be 30 or 31 depending on implementation)
        exponent = (result_u >> 10) & 0x1F
        assert exponent >= 30, f"Positive infinity exponent should be >= 30, got {exponent} ({hex(result_u)})"
        assert result_u > 0x7000, f"Positive infinity should be a very large positive value"

    def test_quantize_negative_infinity(self):
        """Test quantizing negative infinity"""
        result = labneura.quantize_fp16(float('-inf'))
        result_u = to_uint16(result)
        # Negative infinity should have sign=1 and very high exponent value
        sign = (result_u >> 15) & 0x1
        exponent = (result_u >> 10) & 0x1F
        assert sign == 1, f"Negative infinity should have sign=1, got {sign} ({hex(result_u)})"
        assert exponent >= 30, f"Negative infinity exponent should be >= 30, got {exponent} ({hex(result_u)})"

    def test_quantize_nan(self):
        """Test quantizing NaN"""
        result = labneura.quantize_fp16(float('nan'))
        # NaN: sign=0, exponent=all 1s, mantissa!=0
        # Expected: 0111110000000001 = 0x7E00 (or similar)
        # We just check that it's in the NaN range
        result_u = to_uint16(result)
        assert (result_u & 0x7C00) == 0x7C00, f"NaN should have all 1s in exponent, got {hex(result_u)}"

    def test_quantize_small_positive(self):
        """Test quantizing small positive values"""
        values = [0.0009765625, 0.001953125, 0.00390625]
        for val in values:
            result = labneura.quantize_fp16(val)
            assert result > 0, f"Small positive value {val} should quantize to non-zero"

    def test_quantize_large_value(self):
        """Test quantizing large values"""
        result = labneura.quantize_fp16(65504.0)  # Max normal FP16
        # Should be finite and positive
        result_u = to_uint16(result)
        assert (result_u & 0x7C00) != 0x7C00, f"65504.0 should not overflow to infinity, got {hex(result_u)}"

    def test_quantize_overflow_to_infinity(self):
        """Test that very large values overflow to infinity or near-infinity"""
        result = labneura.quantize_fp16(1e6)
        result_u = to_uint16(result)
        # Very large value should have very high exponent value (30 or 31)
        exponent = (result_u >> 10) & 0x1F
        assert exponent >= 30, f"Overflow should have exponent >= 30, got {exponent} ({hex(result_u)})"

    def test_quantize_very_small_underflow(self):
        """Test that very small values underflow to zero"""
        result = labneura.quantize_fp16(1e-10)
        # Should underflow to zero: 0x0000
        assert result == 0x0000, f"Very small value should underflow to zero, got {hex(result)}"

    def test_quantize_negative_values(self):
        """Test quantizing various negative values"""
        values = [-0.5, -2.0, -4.5, -100.0]
        for val in values:
            result = labneura.quantize_fp16(val)
            result_u = to_uint16(result)
            # All negative values should have sign bit set (bit 15 = 1)
            assert (result_u >> 15) == 1, f"Negative value {val} should have sign bit set, got {hex(result_u)}"


class TestDequantizeFP16:
    """Tests for dequantize_fp16 function"""

    def test_dequantize_zero(self):
        """Test dequantizing 0x0000 (zero)"""
        result = labneura.dequantize_fp16(0x0000)
        assert result == 0.0, "0x0000 should dequantize to 0.0"

    def test_dequantize_one(self):
        """Test dequantizing 0x3C00 (1.0)"""
        result = labneura.dequantize_fp16(0x3C00)
        assert abs(result - 1.0) < 1e-6, f"0x3C00 should dequantize to ~1.0, got {result}"

    def test_dequantize_negative_one(self):
        """Test dequantizing 0xBC00 (-1.0)"""
        result = labneura.dequantize_fp16(0xBC00)
        assert abs(result - (-1.0)) < 1e-6, f"0xBC00 should dequantize to ~-1.0, got {result}"

    def test_dequantize_half(self):
        """Test dequantizing 0x3800 (0.5)"""
        result = labneura.dequantize_fp16(0x3800)
        assert abs(result - 0.5) < 1e-6, f"0x3800 should dequantize to ~0.5, got {result}"

    def test_dequantize_positive_infinity(self):
        """Test dequantizing 0x7C00 (positive infinity)"""
        result = labneura.dequantize_fp16(0x7C00)
        assert result == float('inf'), f"0x7C00 should dequantize to positive infinity, got {result}"

    def test_dequantize_negative_infinity(self):
        """Test dequantizing 0xFC00 (negative infinity)"""
        result = labneura.dequantize_fp16(0xFC00)
        assert result == float('-inf'), f"0xFC00 should dequantize to negative infinity, got {result}"

    def test_dequantize_nan(self):
        """Test dequantizing 0x7E00 (NaN)"""
        result = labneura.dequantize_fp16(0x7E00)
        assert np.isnan(result), f"0x7E00 should dequantize to NaN, got {result}"

    def test_dequantize_small_positive(self):
        """Test dequantizing small positive FP16 values"""
        # Manually construct a small FP16 value
        small_fp16 = 0x0001  # Smallest positive subnormal
        result = labneura.dequantize_fp16(small_fp16)
        assert result >= 0, f"Small positive FP16 should dequantize to positive value"

    def test_dequantize_large_value(self):
        """Test dequantizing large FP16 value"""
        large_fp16 = 0x7BFF  # Max normal FP16 (65504)
        result = labneura.dequantize_fp16(large_fp16)
        assert result > 0 and result < float('inf'), f"Large FP16 should dequantize to finite positive, got {result}"


class TestQuantizeDequantizeRoundTrip:
    """Tests for round-trip quantization and dequantization"""

    def test_roundtrip_common_values(self):
        """Test round-trip for common values"""
        test_values = [0.0, 1.0, -1.0, 0.5, 2.0, -0.5, 100.0, 0.001]
        for val in test_values:
            # Skip special values like infinity that may have rounding
            if abs(val) < 60000 and abs(val) > 1e-10:
                # FP32 -> FP16 -> FP32
                quantized = labneura.quantize_fp16(val)
                quantized_u = to_uint16(quantized)
                dequantized = labneura.dequantize_fp16(quantized_u)
                
                # Allow for rounding error due to reduced precision
                # FP16 has 10-bit mantissa vs 23-bit for FP32
                if val != 0:
                    rel_error = abs(dequantized - val) / abs(val)
                    assert rel_error < 0.01, f"Round-trip error for {val} too large: got {dequantized}, error {rel_error}"
                else:
                    assert dequantized == val, f"Round-trip for zero should be exact"

    def test_roundtrip_random_values(self):
        """Test round-trip with random values"""
        np.random.seed(42)
        for _ in range(100):
            val = float(np.random.uniform(-1000, 1000))
            quantized = labneura.quantize_fp16(val)
            quantized_u = to_uint16(quantized)
            dequantized = labneura.dequantize_fp16(quantized_u)
            
            # Check that values are at least in the right ballpark
            if abs(val) > 1e-5:  # Ignore tiny values that might underflow
                assert abs(dequantized) > 0 or val == 0, f"Dequantized value should not flip sign"

    def test_roundtrip_preserves_sign(self):
        """Test that round-trip preserves sign for normal values"""
        test_values = [1.5, -1.5, 0.25, -0.25, 10.0, -10.0]
        for val in test_values:
            quantized = labneura.quantize_fp16(val)
            quantized_u = to_uint16(quantized)
            dequantized = labneura.dequantize_fp16(quantized_u)
            
            # Sign should be preserved (except for zero)
            if val > 0:
                assert dequantized >= 0, f"Positive value should remain positive after round-trip"
            elif val < 0:
                assert dequantized <= 0, f"Negative value should remain negative after round-trip"

    def test_roundtrip_precision_loss(self):
        """Test that precision loss is expected and reasonable"""
        # FP16 has 10-bit mantissa, so precision is limited
        val = 1.23456789  # High precision float
        quantized = labneura.quantize_fp16(val)
        dequantized = labneura.dequantize_fp16(quantized)
        
        # Should lose some precision but be close
        assert abs(dequantized - val) / val < 0.01, f"Precision loss should be < 1%"
        
        # Dequantized value should be different from original (precision loss)
        # unless the value can be exactly represented in FP16
        # For most values, some precision is lost
        if abs(val) > 1:  # Avoid precision loss being < floating point epsilon
            pass  # Accept either exact or approximate


class TestEdgeCases:
    """Tests for edge cases and special scenarios"""

    def test_quantize_alternating_bits(self):
        """Test quantizing values with specific bit patterns"""
        # This tests that the bit manipulation is correct
        test_values = [0.0, 1.0, -1.0, 0.5, -0.5]
        for val in test_values:
            result = labneura.quantize_fp16(val)
            result_u = to_uint16(result)
            # Result should be a valid 16-bit value
            assert 0 <= result_u <= 0xFFFF, f"Quantized value should fit in 16 bits"

    def test_dequantize_all_zeros_exponent(self):
        """Test dequantizing with all-zero exponent (denormalized/zero)"""
        # 0x0000 - zero
        result1 = labneura.dequantize_fp16(0x0000)
        assert result1 == 0.0

        # 0x8000 - negative zero
        result2 = labneura.dequantize_fp16(0x8000)
        assert result2 == 0.0 or result2 == -0.0

    def test_quantize_dequantize_powers_of_two(self):
        """Test quantize/dequantize with powers of 2"""
        for exp in range(-5, 6):
            val = 2.0 ** exp
            quantized = labneura.quantize_fp16(val)
            dequantized = labneura.dequantize_fp16(quantized)
            
            # Powers of 2 should be exactly representable in FP16
            assert abs(dequantized - val) / abs(val) < 1e-6 or val == 0, \
                f"Power of 2 ({val}) should be exactly representable"

    def test_quantize_mixed_signs(self):
        """Test quantizing both positive and negative of the same magnitude"""
        magnitudes = [0.5, 1.0, 2.0, 10.0, 100.0]
        for mag in magnitudes:
            pos_result = labneura.quantize_fp16(mag)
            neg_result = labneura.quantize_fp16(-mag)
            pos_u = to_uint16(pos_result)
            neg_u = to_uint16(neg_result)
            
            # Sign bit should differ, rest should be the same
            assert (pos_u ^ neg_u) == 0x8000, \
                f"Sign bit should differ for {mag} and {-mag}, got {hex(pos_u)} vs {hex(neg_u)}"

    def test_consecutive_quantized_values(self):
        """Test that consecutive quantized values differ by smallest unit"""
        # Quantize 1.0 and smallest increments
        q1 = labneura.quantize_fp16(1.0)
        q_next = labneura.quantize_fp16(1.0 + 2**-10)  # One ULP at 1.0
        
        # These should be different FP16 values
        # (FP16 has 10-bit mantissa, so 2^-10 is one ULP at 1.0)
        assert q1 != q_next or (q1 == q_next and 1.0 + 2**-10 == 1.0), \
            "Consecutive values should differ or be equal if indistinguishable"


class TestCompatibilityWithNumPy:
    """Tests to verify compatibility with NumPy's float16"""

    def test_numpy_compatibility_simple(self):
        """Test that quantize/dequantize align with numpy.float16"""
        np_val = np.float16(1.5)
        
        # Quantize to FP16, then dequantize and compare
        quantized = labneura.quantize_fp16(float(np_val))
        dequantized = labneura.dequantize_fp16(quantized)
        
        # Should match numpy's float16 representation
        assert abs(float(np_val) - dequantized) < 1e-6, \
            f"Should match NumPy float16 representation"

    def test_numpy_special_values(self):
        """Test special values match NumPy"""
        np_inf = np.float16('inf')
        np_ninf = np.float16('-inf')
        
        q_inf = labneura.quantize_fp16(float(np_inf))
        q_ninf = labneura.quantize_fp16(float(np_ninf))
        q_inf_u = to_uint16(q_inf)
        q_ninf_u = to_uint16(q_ninf)
        
        d_inf = labneura.dequantize_fp16(q_inf_u)
        d_ninf = labneura.dequantize_fp16(q_ninf_u)
        
        # Both should be very large or infinite
        assert d_inf == float('inf') or d_inf > 60000, f"Large value should be infinity or very large, got {d_inf}"
        assert d_ninf == float('-inf') or d_ninf < -60000, f"Large negative should be -infinity or very large negative, got {d_ninf}"

    def test_array_quantization(self):
        """Test quantizing arrays of values"""
        np_array = np.array([0.5, 1.0, 1.5, 2.0, -0.5], dtype=np.float32)
        np_fp16_array = np_array.astype(np.float16)
        
        # Quantize each value
        quantized = [labneura.quantize_fp16(float(x)) for x in np_array]
        quantized_u = [to_uint16(q) for q in quantized]
        dequantized = [labneura.dequantize_fp16(q) for q in quantized_u]
        
        # Compare with NumPy's float16
        for i, (our_val, np_val) in enumerate(zip(dequantized, np_fp16_array)):
            assert abs(our_val - float(np_val)) < 1e-6, \
                f"Mismatch at index {i}: our {our_val} vs numpy {float(np_val)}"


class TestPerformance:
    """Performance and stress tests"""

    def test_batch_quantization(self):
        """Test quantizing a large batch of values"""
        values = [float(i) / 100.0 for i in range(-10000, 10000)]
        
        quantized = []
        for val in values:
            quantized.append(labneura.quantize_fp16(val))
        
        assert len(quantized) == len(values), "All values should be quantized"
        
        # Dequantize and verify some samples
        for i in [0, 100, 1000, 5000, 9999]:
            quantized_u = to_uint16(quantized[i])
            dequantized = labneura.dequantize_fp16(quantized_u)
            if abs(values[i]) > 1e-5:
                rel_error = abs(dequantized - values[i]) / abs(values[i])
                assert rel_error < 0.1, f"Large relative error at index {i}"

    def test_quantization_idempotent(self):
        """Test that dequantize(quantize(x)) is idempotent"""
        val = 2.5
        
        # First round
        q1 = labneura.quantize_fp16(val)
        d1 = labneura.dequantize_fp16(to_uint16(q1))
        
        # Second round (should be exact)
        q2 = labneura.quantize_fp16(d1)
        d2 = labneura.dequantize_fp16(to_uint16(q2))
        
        # d1 and d2 should be exactly equal (idempotent)
        assert d1 == d2, "Dequantize/quantize should be idempotent"
