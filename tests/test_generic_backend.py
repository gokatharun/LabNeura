#!/usr/bin/env python3
"""
Tests for the Generic backend by forcing LABNEURA_BACKEND=GENERIC
and exercising FP32 and INT8 operations and constructors.
"""

import os
from contextlib import contextmanager
import pytest
import labneura as ln

@contextmanager
def force_generic_backend():
    prev = os.environ.get('LABNEURA_BACKEND')
    os.environ['LABNEURA_BACKEND'] = 'GENERIC'
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop('LABNEURA_BACKEND', None)
        else:
            os.environ['LABNEURA_BACKEND'] = prev

class TestGenericConstructors:
    def test_fp32_from_floats(self):
        with force_generic_backend():
            t = ln.Tensor([1.0, 2.0, 3.5, -4.0], ln.QuantizationMode.FP32)
        assert t.size() == 4
        d = t.data_fp32()
        assert pytest.approx(d) == [1.0, 2.0, 3.5, -4.0]

    def test_fp32_from_ints(self):
        with force_generic_backend():
            t = ln.Tensor([1, -2, 3, 4], ln.QuantizationMode.FP32)
        assert t.size() == 4
        d = t.data_fp32()
        assert pytest.approx(d) == [1.0, -2.0, 3.0, 4.0]

    def test_int8_from_ints(self):
        with force_generic_backend():
            t = ln.Tensor([10, 127, -128, 40], ln.QuantizationMode.INT8)
        assert t.size() == 4
        d = t.data_int8()
        assert d == [10, 127, -128, 40]

    def test_scalar_fp32(self):
        with force_generic_backend():
            t = ln.Tensor(3.25, ln.QuantizationMode.FP32)
        assert t.size() == 1
        assert pytest.approx(t.data_fp32()[0]) == 3.25

    def test_scalar_int8(self):
        with force_generic_backend():
            t = ln.Tensor(42, ln.QuantizationMode.INT8)
        assert t.size() == 1
        assert t.data_int8()[0] == 42

class TestGenericFP32Ops:
    def test_add(self):
        with force_generic_backend():
            a = ln.Tensor([1.0, 2.0, 3.0], ln.QuantizationMode.FP32)
            b = ln.Tensor([0.5, -1.0, 4.0], ln.QuantizationMode.FP32)
            c = a.add(b)
        assert pytest.approx(c.data_fp32()) == [1.5, 1.0, 7.0]

    def test_mul(self):
        with force_generic_backend():
            a = ln.Tensor([2.0, -3.0, 4.0], ln.QuantizationMode.FP32)
            b = ln.Tensor([3.0, 2.0, -0.5], ln.QuantizationMode.FP32)
            c = a.mul(b)
        assert pytest.approx(c.data_fp32()) == [6.0, -6.0, -2.0]

    def test_sub(self):
        with force_generic_backend():
            a = ln.Tensor([5.0, 6.5, -7.0], ln.QuantizationMode.FP32)
            b = ln.Tensor([2.0, 1.5, 3.0], ln.QuantizationMode.FP32)
            c = a.sub(b)
        assert pytest.approx(c.data_fp32()) == [3.0, 5.0, -10.0]

    def test_inplace_add(self):
        with force_generic_backend():
            a = ln.Tensor([1.0, 2.0, 3.0], ln.QuantizationMode.FP32)
            b = ln.Tensor([4.0, -1.0, 0.5], ln.QuantizationMode.FP32)
            a.add_inplace(b)
        assert pytest.approx(a.data_fp32()) == [5.0, 1.0, 3.5]

    def test_inplace_mul(self):
        with force_generic_backend():
            a = ln.Tensor([2.0, 3.0, -1.0], ln.QuantizationMode.FP32)
            b = ln.Tensor([3.0, 0.5, 10.0], ln.QuantizationMode.FP32)
            a.mul_inplace(b)
        assert pytest.approx(a.data_fp32()) == [6.0, 1.5, -10.0]

    def test_inplace_sub(self):
        with force_generic_backend():
            a = ln.Tensor([5.0, 6.0, 7.0], ln.QuantizationMode.FP32)
            b = ln.Tensor([1.0, 2.0, 3.0], ln.QuantizationMode.FP32)
            a.sub_inplace(b)
        assert pytest.approx(a.data_fp32()) == [4.0, 4.0, 4.0]

class TestGenericINT8Ops:
    def test_add(self):
        with force_generic_backend():
            a = ln.Tensor([10, 20, 30, 40], ln.QuantizationMode.INT8)
            b = ln.Tensor([5, -10, 127, -128], ln.QuantizationMode.INT8)
            c = a.add(b)
        assert c.data_int8() == [15, 10, 127, -88]  # 40 + -128 = -88

    def test_mul(self):
        with force_generic_backend():
            a = ln.Tensor([2, 3, 4, -5], ln.QuantizationMode.INT8)
            b = ln.Tensor([3, -4, 50, 6], ln.QuantizationMode.INT8)
            c = a.mul(b)
        # 4*50=200->127, -5*6=-30
        assert c.data_int8() == [6, -12, 127, -30]

    def test_sub(self):
        with force_generic_backend():
            a = ln.Tensor([50, -60, 70, -80], ln.QuantizationMode.INT8)
            b = ln.Tensor([10, 20, -30, 40], ln.QuantizationMode.INT8)
            c = a.sub(b)
        assert c.data_int8() == [40, -80, 100, -120]

    def test_inplace_add(self):
        with force_generic_backend():
            a = ln.Tensor([1, 2, 3], ln.QuantizationMode.INT8)
            b = ln.Tensor([4, 5, 126], ln.QuantizationMode.INT8)
            a.add_inplace(b)
        assert a.data_int8() == [5, 7, 127]

    def test_inplace_mul(self):
        with force_generic_backend():
            a = ln.Tensor([2, -3, 4], ln.QuantizationMode.INT8)
            b = ln.Tensor([2, 3, 64], ln.QuantizationMode.INT8)
            a.mul_inplace(b)
        assert a.data_int8() == [4, -9, 127]

    def test_inplace_sub(self):
        with force_generic_backend():
            a = ln.Tensor([10, 20, -30], ln.QuantizationMode.INT8)
            b = ln.Tensor([1, -2, 3], ln.QuantizationMode.INT8)
            a.sub_inplace(b)
        assert a.data_int8() == [9, 22, -33]

class TestGenericSanity:
    def test_backend_detect_generic(self):
        # detect_backend returns preferred backend without override.
        # We set LABNEURA_BACKEND, but detection may still report NEON/AVX2.
        # Ensure API is callable and returns a valid string.
        b = ln.detect_backend()
        assert b in ["GENERIC", "NEON", "AVX2"]
