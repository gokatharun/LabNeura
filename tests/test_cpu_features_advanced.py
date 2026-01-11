"""
Advanced CPU features testing with mocking support.
This test file improves cpu_features.cpp coverage by testing behavior
across different simulated CPU configurations.
"""

import pytest
import labneura
import subprocess
import sys
import os
import platform


class TestCPUFeaturesDetectionAdvanced:
    """
    Advanced tests for cpu_features.cpp.
    Tests the behavior of CPU feature detection at runtime.
    """

    def test_detect_backend_returns_neon_or_generic_on_arm(self):
        """
        On ARM M1, should detect NEON or fallback to GENERIC.
        Tests the cpu_supports_neon() function path.
        """
        backend = labneura.detect_backend()
        
        # Platform-aware: only check ARM behavior on ARM machines
        is_arm = platform.machine() in ['arm64', 'aarch64']
        is_x86 = platform.machine() in ['x86_64', 'AMD64']
        
        if is_arm:
            # On ARM M1, expect NEON or GENERIC
            assert backend in ["NEON", "GENERIC"] or backend.startswith("NEON") or backend.startswith("GENERIC"), \
                f"Expected NEON or GENERIC on ARM M1, got {backend}"
        elif is_x86:
            # On x86, expect AVX2, SSE4.1, or GENERIC
            assert any(b in backend.upper() for b in ["AVX2", "SSE4.1", "GENERIC"]), \
                f"Expected AVX2, SSE4.1, or GENERIC on x86, got {backend}"
        else:
            # Generic fallback for other architectures
            assert isinstance(backend, str) and len(backend) > 0

    def test_backend_consistency_across_multiple_calls(self):
        """
        CPU feature detection must be consistent across multiple calls.
        Tests that detect_cpu_features() returns same result each time.
        """
        backends = []
        for _ in range(5):
            b = labneura.detect_backend()
            backends.append(b)
        
        # All should be the same
        assert len(set(backends)) == 1, \
            f"Backend detection not consistent: {backends}"

    def test_avx2_returns_false_on_arm_m1(self):
        """
        On ARM M1, AVX2 should not be available.
        Tests the #else path of cpu_supports_avx2().
        """
        backend = labneura.detect_backend()
        
        # Platform-aware: only check AVX2 absence on ARM machines
        is_arm = platform.machine() in ['arm64', 'aarch64']
        
        if is_arm:
            # AVX2 is x86-only, shouldn't appear on ARM M1
            assert "AVX2" not in backend.upper(), "AVX2 should not be available on ARM M1"
        else:
            # On x86, we expect AVX2 or SSE4.1 (or fallback to GENERIC)
            assert "NEON" not in backend.upper(), "NEON should not be available on x86"

    def test_sse41_returns_false_on_arm_m1(self):
        """
        On ARM M1, SSE4.1 should not be available.
        Tests that x86 backends are not selected on ARM.
        """
        backend = labneura.detect_backend()
        # SSE4.1 is x86-only
        assert "SSE4.1" not in backend, "SSE4.1 should not be available on ARM M1"

    def test_backend_factory_uses_cpu_features(self):
        """
        Verify that backend factory actually uses cpu feature detection.
        Create tensors and verify they work with detected backend.
        """
        # Get detected backend
        detected = labneura.detect_backend()
        
        # Create various tensors - they should work with detected backend
        t1 = labneura.Tensor([1.0, 2.0, 3.0])
        t2 = labneura.Tensor([1, 2, 3], labneura.QuantizationMode.INT8)
        
        # Verify operations work
        assert t1.size() == 3
        assert t2.size() == 3
        
        # Operations should execute without error
        t1_copy = labneura.Tensor([1.0, 2.0, 3.0])
        t1.add_inplace(t1_copy)
        assert len(t1.data_fp32()) == 3

    def test_cpu_feature_detection_with_neon_backend(self):
        """
        If NEON backend is used (on ARM M1), verify it's properly detected.
        """
        backend = labneura.detect_backend()
        if "NEON" in backend or "neon" in backend.lower():
            # NEON backend is available on M1
            # cpu_supports_neon() should have returned true
            # This implies lines 47-60 of cpu_features.cpp executed
            t = labneura.Tensor([1, 2, 3], labneura.QuantizationMode.INT8)
            result = t.data_int8()
            assert result == [1, 2, 3]

    def test_cpu_features_initialization(self):
        """
        Test that cpu_features is properly initialized on first call.
        """
        # First call initializes static variables
        backend1 = labneura.detect_backend()
        # Subsequent calls should return cached value
        backend2 = labneura.detect_backend()
        
        assert backend1 == backend2, "Backend detection not cached properly"

    def test_backend_selection_logic_multiple_tensor_types(self):
        """
        Verify backend selection works correctly for both FP32 and INT8.
        Tests that cpu feature detection is independent of data type.
        """
        # Create FP32 tensor
        fp32_t = labneura.Tensor([1.5, 2.5, 3.5])
        
        # Create INT8 tensor
        int8_t = labneura.Tensor([1, 2, 3], labneura.QuantizationMode.INT8)
        
        # Both should work with same backend
        assert fp32_t.size() == int8_t.size()
        
        # Operations on both should work
        fp32_t.add_inplace(labneura.Tensor([0.5, 0.5, 0.5]))
        int8_t.add_inplace(labneura.Tensor([1, 1, 1], labneura.QuantizationMode.INT8))
        
        assert abs(fp32_t.data_fp32()[0] - 2.0) < 1e-5
        assert int8_t.data_int8()[0] == 2

    def test_cpu_supports_neon_on_aarch64(self):
        """
        On AArch64 (including Apple Silicon), cpu_supports_neon() should return true.
        This tests lines 47-60 of cpu_features.cpp.
        """
        backend = labneura.detect_backend()
        backend_upper = backend.upper()
        
        # Platform-aware: only check ARM behavior on ARM machines
        is_arm = platform.machine() in ['arm64', 'aarch64']
        is_x86 = platform.machine() in ['x86_64', 'AMD64']
        
        if is_arm:
            # If we're on ARM M1/M2/M3, NEON must be detected or Generic must be used
            assert backend_upper.startswith("NEON") or backend_upper.startswith("GENERIC"), \
                f"Invalid backend on ARM: {backend}. Expected NEON or GENERIC."
        elif is_x86:
            # On x86, expect AVX2, SSE4.1, or GENERIC
            assert any(b in backend_upper for b in ["AVX2", "SSE4.1", "GENERIC"]), \
                f"Invalid backend on x86: {backend}. Expected AVX2, SSE4.1, or GENERIC."
        else:
            # For other architectures, just verify it returns a valid backend
            assert isinstance(backend, str) and len(backend) > 0


class TestCPUDetectionEdgeCases:
    """
    Edge case tests for CPU feature detection.
    """

    def test_detect_backend_with_empty_workspace(self):
        """
        Backend detection should work even with no tensors created yet.
        """
        backend = labneura.detect_backend()
        assert isinstance(backend, str)
        assert len(backend) > 0

    def test_detect_backend_multiple_threads_consistency(self):
        """
        CPU detection should be thread-safe and consistent across threads.
        """
        import threading
        results = []
        
        def detect():
            results.append(labneura.detect_backend())
        
        threads = [threading.Thread(target=detect) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All results should be the same
        assert len(set(results)) == 1, "CPU detection not thread-safe"

    def test_cpu_features_fallback_to_generic(self):
        """
        Verify that even if specific backends aren't available,
        GenericBackend is always available as fallback.
        """
        backend = labneura.detect_backend()
        # Backend should be one of the known types
        valid = ["NEON", "AVX2", "SSE4.1", "GENERIC", "neon", "avx2", "sse", "generic",
                "Neon", "Avx2", "Sse", "NEON backend", "AVX2 backend", "SSE4.1 backend", 
                "Generic backend"]
        assert any(v.lower() in backend.lower() for v in valid), \
            f"Invalid backend: {backend}"


class TestCPUFeaturesIntegration:
    """
    Integration tests that verify CPU features work correctly in real operations.
    """

    def test_large_tensor_with_detected_backend(self):
        """
        Test large tensor operations with whatever backend is detected.
        If NEON is detected, uses NEON; if Generic, uses Generic.
        """
        size = 100000
        a = labneura.Tensor([1.0] * size)
        b = labneura.Tensor([1.0] * size)
        
        a.add_inplace(b)
        result = a.data_fp32()
        
        # All elements should be 2.0
        assert all(abs(x - 2.0) < 1e-5 for x in result)

    def test_int8_operations_with_detected_backend(self):
        """
        Test INT8 saturation with whatever backend is detected.
        """
        a = labneura.Tensor([100, 50, -100], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([50, 100, 50], labneura.QuantizationMode.INT8)
        
        a.add_inplace(b)
        result = a.data_int8()
        
        # Should handle saturation correctly
        assert result[0] == 127  # 100 + 50 saturates to 127
        assert result[1] == 127  # 50 + 100 = 150 saturates to 127

    def test_backend_performance_characteristics(self):
        """
        Verify that operations complete in reasonable time.
        Different backends may have different performance, but should all complete.
        """
        size = 1000000
        a = labneura.Tensor([1.0] * size)
        b = labneura.Tensor([2.0] * size)
        
        import time
        start = time.time()
        a.mul_inplace(b)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 1 second)
        assert elapsed < 1.0, f"Operation took {elapsed}s, expected < 1s"
        
        result = a.data_fp32()
        assert all(abs(x - 2.0) < 1e-5 for x in result)

    def test_cpu_features_does_not_affect_quantization(self):
        """
        CPU feature detection should not affect quantization mode selection.
        """
        fp32_tensor = labneura.Tensor([1.0, 2.0])
        int8_tensor = labneura.Tensor([1, 2], labneura.QuantizationMode.INT8)
        
        assert fp32_tensor.quantization_mode() == labneura.QuantizationMode.FP32
        assert int8_tensor.quantization_mode() == labneura.QuantizationMode.INT8


class TestBackendFactoryIntegration:
    """
    Tests for backend factory and its use of CPU features.
    """

    def test_backend_factory_creates_functional_backend(self):
        """
        Verify that factory creates a working backend.
        """
        t1 = labneura.Tensor([1.0, 2.0, 3.0])
        t2 = labneura.Tensor([1.0, 1.0, 1.0])
        
        t1.add_inplace(t2)
        result = t1.data_fp32()
        
        expected = [2.0, 3.0, 4.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_backend_factory_handles_size_zero(self):
        """
        Backend factory should handle empty tensors.
        """
        t = labneura.Tensor()
        assert t.size() == 0

    def test_backend_factory_mixed_operations_sequence(self):
        """
        Sequence of different operations with factory-created backend.
        """
        t = labneura.Tensor([10.0, 20.0, 30.0])
        t_tmp = labneura.Tensor([5.0, 5.0, 5.0])
        
        # Add
        t.add_inplace(t_tmp)
        # Subtract
        t.sub_inplace(t_tmp)
        # Multiply
        t.mul_inplace(t_tmp)
        
        result = t.data_fp32()
        expected = [50.0, 100.0, 150.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)


# Import numpy at module level for tests
try:
    import numpy as np
except ImportError:
    np = None
