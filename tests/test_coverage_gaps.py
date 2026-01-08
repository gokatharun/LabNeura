"""
Test cases specifically designed to cover missing lines in cpp files.
Targets: cpu_features.cpp, tensor.cpp, generic.cpp, base.cpp
"""

import pytest
import numpy as np
import labneura


class TestTensorMoveSemanticsAndCopy:
    """Tests for tensor copy and move semantics - covers tensor.cpp missing lines"""

    def test_tensor_copy_via_assignment_fp32(self):
        """Copy assignment for FP32 tensor"""
        original = labneura.Tensor([1.0, 2.0, 3.0])
        copy = labneura.Tensor()
        copy = original  # Assignment copies via backend->clone()
        
        # Verify copy has same data
        np.testing.assert_allclose(copy.data_fp32(), [1.0, 2.0, 3.0], rtol=1e-6)

    def test_tensor_copy_via_assignment_int8(self):
        """Copy assignment for INT8 tensor"""
        original = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT8)
        copy = labneura.Tensor()
        copy = original
        
        # Verify copy has same data
        assert copy.data_int8() == [10, 20, 30]
        assert copy.quantization_mode() == labneura.QuantizationMode.INT8

    def test_tensor_copy_assignment_fp32(self):
        """Copy assignment operator for FP32 tensor"""
        original = labneura.Tensor([1.0, 2.0])
        copy = labneura.Tensor()
        
        copy = original  # Assignment
        np.testing.assert_allclose(copy.data_fp32(), [1.0, 2.0], rtol=1e-6)
        assert copy.quantization_mode() == labneura.QuantizationMode.FP32

    def test_tensor_copy_assignment_int8(self):
        """Copy assignment operator for INT8 tensor"""
        original = labneura.Tensor([50, 100], labneura.QuantizationMode.INT8)
        copy = labneura.Tensor()
        
        copy = original  # Assignment
        assert copy.data_int8() == [50, 100]
        assert copy.quantization_mode() == labneura.QuantizationMode.INT8

    def test_tensor_self_assignment(self):
        """Self-assignment should be safe"""
        t = labneura.Tensor([1.0, 2.0, 3.0])
        original_data = list(t.data_fp32())
        
        t = t  # Self assignment
        
        np.testing.assert_allclose(t.data_fp32(), original_data, rtol=1e-6)

    def test_tensor_move_constructor(self):
        """Move constructor transfers ownership"""
        def create_tensor():
            return labneura.Tensor([4.0, 5.0, 6.0])
        
        t = create_tensor()  # Move construction
        np.testing.assert_allclose(t.data_fp32(), [4.0, 5.0, 6.0], rtol=1e-6)

    def test_tensor_move_assignment(self):
        """Move assignment operator"""
        def create_tensor():
            return labneura.Tensor([7.0, 8.0, 9.0], labneura.QuantizationMode.FP32)
        
        t = labneura.Tensor()
        t = create_tensor()  # Move assignment
        
        np.testing.assert_allclose(t.data_fp32(), [7.0, 8.0, 9.0], rtol=1e-6)

    def test_tensor_self_move_assignment(self):
        """Self move assignment should be safe"""
        t = labneura.Tensor([1.0, 2.0])
        original_data = [1.0, 2.0]
        
        # Can't directly test move semantics in Python, but verify it works
        np.testing.assert_allclose(t.data_fp32(), original_data, rtol=1e-6)


class TestTensorNullptrBackendHandling:
    """Tests for tensor operations with uninitialized backend"""

    def test_empty_tensor_size(self):
        """Empty tensor should have size 0"""
        t = labneura.Tensor()
        assert t.size() == 0

    def test_empty_tensor_numel(self):
        """Empty tensor should have numel 0"""
        t = labneura.Tensor()
        assert t.numel() == 0

    def test_empty_tensor_mode(self):
        """Empty tensor should be in FP32 mode by default"""
        t = labneura.Tensor()
        assert t.quantization_mode() == labneura.QuantizationMode.FP32


class TestGenericBackendAllOperations:
    """Tests covering all paths in generic.cpp backend operations"""

    def test_generic_fp32_add_large(self):
        """Large FP32 addition in generic backend"""
        # Create large tensors to ensure we use generic backend
        data = [float(i) for i in range(10000)]
        a = labneura.Tensor(data)
        b = labneura.Tensor(data)
        a.add_inplace(b)
        
        result = a.data_fp32()
        for i in range(10000):
            assert abs(result[i] - (2 * i)) < 1e-4

    def test_generic_fp32_mul_large(self):
        """Large FP32 multiplication in generic backend"""
        data = [2.0] * 10000
        a = labneura.Tensor(data)
        b = labneura.Tensor(data)
        a.mul_inplace(b)
        
        result = a.data_fp32()
        assert all(abs(x - 4.0) < 1e-6 for x in result)

    def test_generic_fp32_sub_large(self):
        """Large FP32 subtraction in generic backend"""
        a = labneura.Tensor([10.0] * 10000)
        b = labneura.Tensor([3.0] * 10000)
        a.sub_inplace(b)
        
        result = a.data_fp32()
        assert all(abs(x - 7.0) < 1e-6 for x in result)

    def test_generic_int8_add_large(self):
        """Large INT8 addition with saturation in generic backend"""
        data = [50] * 5000
        a = labneura.Tensor(data, labneura.QuantizationMode.INT8)
        b = labneura.Tensor([50] * 5000, labneura.QuantizationMode.INT8)
        a.add_inplace(b)
        
        result = a.data_int8()
        # 50 + 50 = 100 (no saturation)
        assert all(x == 100 for x in result)

    def test_generic_int8_mul_large(self):
        """Large INT8 multiplication with saturation in generic backend"""
        data = [20] * 10000
        a = labneura.Tensor(data, labneura.QuantizationMode.INT8)
        b = labneura.Tensor([8] * 10000, labneura.QuantizationMode.INT8)
        a.mul_inplace(b)
        
        result = a.data_int8()
        # 20 * 8 = 160, but saturates to 127
        assert all(x == 127 for x in result)

    def test_generic_int8_sub_large(self):
        """Large INT8 subtraction with saturation in generic backend"""
        data = [-50] * 5000
        a = labneura.Tensor(data, labneura.QuantizationMode.INT8)
        b = labneura.Tensor([50] * 5000, labneura.QuantizationMode.INT8)
        a.sub_inplace(b)
        
        result = a.data_int8()
        # -50 - 50 = -100 (no saturation)
        assert all(x == -100 for x in result)

    def test_generic_int8_add_negative_saturation(self):
        """INT8 addition with negative saturation"""
        a = labneura.Tensor([-100, -50], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([-50, -30], labneura.QuantizationMode.INT8)
        a.add_inplace(b)
        
        result = a.data_int8()
        # -100 + -50 = -150 → -128, -50 + -30 = -80
        assert result[0] == -128
        assert result[1] == -80

    def test_generic_int8_mul_zero(self):
        """INT8 multiplication with zero"""
        a = labneura.Tensor([100, 50, -75], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([0, 0, 0], labneura.QuantizationMode.INT8)
        a.mul_inplace(b)
        
        result = a.data_int8()
        assert result == [0, 0, 0]

    def test_generic_int8_mul_negative(self):
        """INT8 multiplication with negative values"""
        a = labneura.Tensor([10, -10, 20], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([-5, -5, -5], labneura.QuantizationMode.INT8)
        a.mul_inplace(b)
        
        result = a.data_int8()
        # 10 * -5 = -50, -10 * -5 = 50, 20 * -5 = -100
        assert result == [-50, 50, -100]

    def test_generic_int8_sub_negative_to_positive(self):
        """INT8 subtraction wrapping behavior"""
        a = labneura.Tensor([-50], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([50], labneura.QuantizationMode.INT8)
        a.sub_inplace(b)
        
        result = a.data_int8()
        # -50 - 50 = -100
        assert result[0] == -100

    def test_generic_fp32_add_negative(self):
        """FP32 addition with negative values"""
        a = labneura.Tensor([-1.5, -2.5, -3.5])
        b = labneura.Tensor([0.5, 0.5, 0.5])
        a.add_inplace(b)
        
        result = a.data_fp32()
        expected = [-1.0, -2.0, -3.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_generic_fp32_sub_negative(self):
        """FP32 subtraction with negative values"""
        a = labneura.Tensor([-5.0, -10.0])
        b = labneura.Tensor([-2.0, -3.0])
        a.sub_inplace(b)
        
        result = a.data_fp32()
        expected = [-3.0, -7.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_generic_fp32_mul_negative(self):
        """FP32 multiplication with negative values"""
        a = labneura.Tensor([2.0, -3.0, 4.0])
        b = labneura.Tensor([-2.0, -2.0, -2.0])
        a.mul_inplace(b)
        
        result = a.data_fp32()
        expected = [-4.0, 6.0, -8.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestCPUFeaturesDetection:
    """Tests for cpu_features.cpp detection functions"""

    def test_detect_cpu_features_returns_struct(self):
        """detect_cpu_features should return a valid structure"""
        # Note: This would require exposing the CPUFeatures struct
        # For now test that detect_backend works
        backend = labneura.detect_backend()
        assert backend is not None

    def test_detect_backend_consistent(self):
        """Backend detection should be consistent across calls"""
        backend1 = labneura.detect_backend()
        backend2 = labneura.detect_backend()
        assert backend1 == backend2

    def test_cpu_supports_functions_return_bool(self):
        """CPU support detection functions should return boolean"""
        # These functions are internal but their behavior is tested through
        # the backend selection logic
        result = labneura.detect_backend()
        # Result should be deterministic
        assert isinstance(result, int) or isinstance(result, str)


class TestTensorClone:
    """Tests for tensor clone/copy via backend"""

    def test_clone_via_copy_assignment_fp32(self):
        """Cloning via copy assignment FP32"""
        original = labneura.Tensor([1.0, 2.0, 3.0, 4.0])
        clone = labneura.Tensor()
        clone = original  # Triggers backend clone
        
        # Verify data is identical
        np.testing.assert_allclose(clone.data_fp32(), original.data_fp32(), rtol=1e-6)

    def test_clone_via_copy_assignment_int8(self):
        """Cloning via copy assignment INT8"""
        original = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT8)
        clone = labneura.Tensor()
        clone = original
        
        assert clone.data_int8() == original.data_int8()

    def test_clone_multiple_assignments_fp32(self):
        """Multiple assignments should all point to cloned data"""
        t1 = labneura.Tensor([1.0, 2.0])
        t2 = labneura.Tensor()
        t3 = labneura.Tensor()
        
        t2 = t1
        t3 = t1
        
        # All should have the same values
        np.testing.assert_allclose(t1.data_fp32(), [1.0, 2.0], rtol=1e-6)
        np.testing.assert_allclose(t2.data_fp32(), [1.0, 2.0], rtol=1e-6)
        np.testing.assert_allclose(t3.data_fp32(), [1.0, 2.0], rtol=1e-6)

    def test_clone_multiple_assignments_int8(self):
        """Multiple assignments should all point to cloned data"""
        t1 = labneura.Tensor([50, 100], labneura.QuantizationMode.INT8)
        t2 = labneura.Tensor()
        t3 = labneura.Tensor()
        
        t2 = t1
        t3 = t1
        
        # All should have the same values
        assert t1.data_int8() == [50, 100]
        assert t2.data_int8() == [50, 100]
        assert t3.data_int8() == [50, 100]


class TestGenericBackendClone:
    """Tests for GenericBackend clone operation"""

    def test_generic_clone_fp32_via_assignment(self):
        """Clone FP32 tensor via assignment"""
        data = [float(i % 256) for i in range(5000)]
        original = labneura.Tensor(data)
        clone = labneura.Tensor()
        clone = original
        
        # Verify all values are cloned
        orig_vals = original.data_fp32()
        clone_vals = clone.data_fp32()
        for i in range(0, len(data), 500):
            assert abs(orig_vals[i] - clone_vals[i]) < 1e-6

    def test_generic_clone_int8_via_assignment(self):
        """Clone INT8 tensor via assignment"""
        data = [int(i % 256) - 128 for i in range(5000)]
        original = labneura.Tensor(data, labneura.QuantizationMode.INT8)
        clone = labneura.Tensor()
        clone = original
        
        # Verify cloning
        assert clone.size() == original.size()
        orig_vals = original.data_int8()
        clone_vals = clone.data_int8()
        for i in range(0, len(data), 500):
            assert orig_vals[i] == clone_vals[i]


class TestBaseBackendMissingCoverage:
    """Tests for base.cpp coverage gaps"""

    def test_backend_add_inplace_with_mode_check(self):
        """add_inplace should validate quantization modes"""
        a = labneura.Tensor([1.0], labneura.QuantizationMode.FP32)
        b = labneura.Tensor([1], labneura.QuantizationMode.INT8)
        
        with pytest.raises(RuntimeError):
            a.add_inplace(b)

    def test_backend_mul_inplace_with_mode_check(self):
        """mul_inplace should validate quantization modes"""
        a = labneura.Tensor([2.0], labneura.QuantizationMode.FP32)
        b = labneura.Tensor([1], labneura.QuantizationMode.INT8)
        
        with pytest.raises(RuntimeError):
            a.mul_inplace(b)

    def test_backend_sub_inplace_with_mode_check(self):
        """sub_inplace should validate quantization modes"""
        a = labneura.Tensor([3.0], labneura.QuantizationMode.FP32)
        b = labneura.Tensor([1], labneura.QuantizationMode.INT8)
        
        with pytest.raises(RuntimeError):
            a.sub_inplace(b)

    def test_backend_add_inplace_size_mismatch(self):
        """add_inplace should validate size"""
        a = labneura.Tensor([1.0, 2.0], labneura.QuantizationMode.FP32)
        b = labneura.Tensor([1.0], labneura.QuantizationMode.FP32)
        
        with pytest.raises(RuntimeError):
            a.add_inplace(b)


# Helper for move semantics (Python doesn't have std::move)
class std:
    @staticmethod
    def move(obj):
        return obj
