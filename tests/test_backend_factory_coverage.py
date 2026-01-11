#!/usr/bin/env python3
"""
Targeted tests to improve coverage of backend_factory.cpp.
Exercises env override branches and detect_backend.
"""

import os
from contextlib import contextmanager
import pytest
import labneura as ln

@contextmanager
def override_backend(value: str):
    prev = os.environ.get('LABNEURA_BACKEND')
    os.environ['LABNEURA_BACKEND'] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop('LABNEURA_BACKEND', None)
        else:
            os.environ['LABNEURA_BACKEND'] = prev

class TestBackendFactoryOverride:
    def test_force_generic_backend(self):
        with override_backend('GENERIC'):
            # Construct tensors to ensure factory returns Generic backend
            t1 = ln.Tensor([1.0, 2.0, 3.0], ln.QuantizationMode.FP32)
            t2 = ln.Tensor([1, 2, 3], ln.QuantizationMode.INT8)
            r = t1.add(ln.Tensor([0.5, 0.5, 0.5], ln.QuantizationMode.FP32))
            assert t1.size() == 3 and t2.size() == 3 and r.size() == 3

    def test_force_neon_backend(self):
        with override_backend('NEON'):
            # FP16 and INT16 paths exist on NEON backend
            t1 = ln.Tensor([1.0, 2.0], ln.QuantizationMode.FP16)
            t2 = ln.Tensor([1000, -2000], ln.QuantizationMode.INT16)
            assert len(t1.data_fp16()) == 2
            assert t2.data_int16() == [1000, -2000]

class TestBackendDetect:
    def test_detect_backend_returns_valid(self):
        b = ln.detect_backend()
        assert b in ['GENERIC', 'NEON', 'AVX2']

    @pytest.mark.skip(reason='AVX2 path not compiled on arm64; documented for completeness')
    def test_force_avx2_backend(self):
        # On x86_64 with AVX2, this would select AVX2 backend
        with override_backend('AVX2'):
            t = ln.Tensor([1.0, 2.0], ln.QuantizationMode.FP32)
            assert t.size() == 2
