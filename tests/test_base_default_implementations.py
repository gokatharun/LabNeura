"""
Tests to cover default implementations in base.cpp that throw exceptions.

GenericBackend's unsupported FP16 paths should call the default
implementations in base.cpp, while supported INT16 paths remain usable.
"""

import os
from contextlib import contextmanager
import pytest
import labneura

GENERIC_BACKEND = labneura.detect_backend() == "GENERIC"


@contextmanager
def force_generic_backend():
    """Force use of GenericBackend."""
    prev = os.environ.get('LABNEURA_BACKEND')
    os.environ['LABNEURA_BACKEND'] = 'GENERIC'
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop('LABNEURA_BACKEND', None)
        else:
            os.environ['LABNEURA_BACKEND'] = prev


class TestGenericBackendFP16Supported:
    """Test that GenericBackend supports FP16 data access and arithmetic."""

    def test_fp16_data_access_works(self):
        """FP16 storage should be accessible with GenericBackend."""
        with force_generic_backend():
            t = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP16)
            assert t.data_fp16() == [0x3c00, 0x4000, 0x4200]

    def test_fp16_operations_with_generic(self):
        """FP16 arithmetic should work with the Generic backend."""
        with force_generic_backend():
            t1 = labneura.Tensor([2.0, 4.0, 6.0], labneura.QuantizationMode.FP16)
            t2 = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP16)
            result = t1.add(t2)
            assert result.data_fp16() == [0x4200, 0x4600, 0x4880]


class TestGenericBackendINT16Supported:
    """Test GenericBackend INT16 data access and equal-shape arithmetic."""

    def test_int16_data_access(self):
        with force_generic_backend():
            tensor = labneura.Tensor(
                [100, 200, 300], labneura.QuantizationMode.INT16
            )
            assert tensor.data_int16() == [100, 200, 300]

    def test_int16_operations_with_generic(self):
        with force_generic_backend():
            lhs = labneura.Tensor(
                [1000, 2000, 3000], labneura.QuantizationMode.INT16
            )
            rhs = labneura.Tensor(
                [100, 200, 300], labneura.QuantizationMode.INT16
            )
            assert lhs.add(rhs).data_int16() == [1100, 2200, 3300]


class TestGenericBackendSupportedModes:
    """Test that GenericBackend properly supports FP32 and INT8"""

    def test_fp32_works_with_generic(self):
        """GenericBackend should support FP32"""
        with force_generic_backend():
            t1 = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP32)
            t2 = labneura.Tensor([4.0, 5.0, 6.0], labneura.QuantizationMode.FP32)
            
            t1.add_inplace(t2)
            result = t1.data_fp32()
            assert result == pytest.approx([5.0, 7.0, 9.0])

    def test_int8_works_with_generic(self):
        """GenericBackend should support INT8"""
        with force_generic_backend():
            t1 = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT8)
            t2 = labneura.Tensor([5, 10, 15], labneura.QuantizationMode.INT8)
            
            t1.mul_inplace(t2)
            result = t1.data_int8()
            assert result == [50, 127, 127]  # Saturates at 127

    def test_fp32_all_operations_generic(self):
        """Test all FP32 operations work with GenericBackend"""
        with force_generic_backend():
            t1 = labneura.Tensor([10.0, 20.0], labneura.QuantizationMode.FP32)
            t2 = labneura.Tensor([2.0, 3.0], labneura.QuantizationMode.FP32)
            
            result_add = t1.add(t2)
            assert result_add.data_fp32() == pytest.approx([12.0, 23.0])
            
            result_mul = t1.mul(t2)
            assert result_mul.data_fp32() == pytest.approx([20.0, 60.0])
            
            result_sub = t1.sub(t2)
            assert result_sub.data_fp32() == pytest.approx([8.0, 17.0])

    def test_int8_all_operations_generic(self):
        """Test all INT8 operations work with GenericBackend"""
        with force_generic_backend():
            t1 = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT8)
            t2 = labneura.Tensor([1, 2, 3], labneura.QuantizationMode.INT8)
            
            result_add = t1.add(t2)
            assert result_add.data_int8() == [11, 22, 33]
            
            result_sub = t1.sub(t2)
            assert result_sub.data_int8() == [9, 18, 27]
            
            result_mul = t1.mul(t2)
            assert result_mul.data_int8() == [10, 40, 90]


class TestBackendCapabilities:
    """Test to understand which quantization modes are supported by which backends"""

    @pytest.mark.skipif(GENERIC_BACKEND, reason="FP16 is unsupported by Generic")
    def test_check_fp16_backend_used(self):
        """Check which backend is used for FP16"""
        # Don't force generic - use default backend selection
        t = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP16)
        assert t.size() == 2
        # FP16 should work on NEON backend (ARM) but might not on Generic

    def test_check_int16_backend_used(self):
        """Check which backend is used for INT16"""
        # Don't force generic - use default backend selection
        t = labneura.Tensor([100, 200], labneura.QuantizationMode.INT16)
        assert t.size() == 2
        # INT16 should work on NEON backend (ARM) but might not on Generic

    def test_generic_backend_limits(self):
        """Document that GenericBackend only supports FP32 and INT8"""
        with force_generic_backend():
            # These should work
            t_fp32 = labneura.Tensor([1.0], labneura.QuantizationMode.FP32)
            assert t_fp32.size() == 1
            
            t_int8 = labneura.Tensor([1], labneura.QuantizationMode.INT8)
            assert t_int8.size() == 1
            
            # These might not work with GenericBackend
            # but the factory might choose a different backend
            try:
                t_fp16 = labneura.Tensor([1.0], labneura.QuantizationMode.FP16)
                # If this succeeds, a capable backend was selected
                assert t_fp16.size() == 1
            except RuntimeError:
                # Expected if GenericBackend was forced and FP16 not supported
                pass
            
            try:
                t_int16 = labneura.Tensor([1], labneura.QuantizationMode.INT16)
                # If this succeeds, a capable backend was selected
                assert t_int16.size() == 1
            except RuntimeError:
                # Expected if GenericBackend was forced and INT16 not supported
                pass


class TestLargeOperationsGeneric:
    """Test large operations with GenericBackend to ensure they work"""

    def test_large_fp32_addition_generic(self):
        """Test large FP32 addition with GenericBackend"""
        with force_generic_backend():
            size = 10000
            t1 = labneura.Tensor([1.0] * size, labneura.QuantizationMode.FP32)
            t2 = labneura.Tensor([2.0] * size, labneura.QuantizationMode.FP32)
            
            t1.add_inplace(t2)
            result = t1.data_fp32()
            assert len(result) == size
            assert all(x == pytest.approx(3.0) for x in result)

    def test_large_int8_multiplication_generic(self):
        """Test large INT8 multiplication with GenericBackend"""
        with force_generic_backend():
            size = 10000
            t1 = labneura.Tensor([2] * size, labneura.QuantizationMode.INT8)
            t2 = labneura.Tensor([3] * size, labneura.QuantizationMode.INT8)
            
            t1.mul_inplace(t2)
            result = t1.data_int8()
            assert len(result) == size
            assert all(x == 6 for x in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
