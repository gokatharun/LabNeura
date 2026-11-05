#!/usr/bin/env python3
"""
Targeted tests to improve coverage of tensor.cpp.
Covers constructors (FP32/FP16/INT8/INT16), default ctor, data accessors,
and FP32/INT8/FP16/INT16 operations including in-place variants.
"""

import os
from contextlib import contextmanager
import pytest
import labneura as ln

@contextmanager
def force_backend(name: str):
    prev = os.environ.get('LABNEURA_BACKEND')
    os.environ['LABNEURA_BACKEND'] = name
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop('LABNEURA_BACKEND', None)
        else:
            os.environ['LABNEURA_BACKEND'] = prev

class TestTensorConstructorsAndAccessors:
    def test_default_constructor(self):
        t = ln.Tensor()
        assert t.size() == 0
        assert t.numel() == 0
        assert t.quantization_mode() == ln.QuantizationMode.FP32

    def test_fp32_vector_and_access(self):
        t = ln.Tensor([1.0, -2.5, 3.25], ln.QuantizationMode.FP32)
        assert t.size() == 3
        assert pytest.approx(t.data_fp32()) == [1.0, -2.5, 3.25]

    def test_int8_vector_and_access(self):
        t = ln.Tensor([10, 127, -128, 5], ln.QuantizationMode.INT8)
        assert t.size() == 4
        assert t.data_int8() == [10, 127, -128, 5]

    def test_scalar_fp32_and_int8(self):
        tf = ln.Tensor(2.0, ln.QuantizationMode.FP32)
        ti = ln.Tensor(42, ln.QuantizationMode.INT8)
        assert tf.size() == 1 and ti.size() == 1
        assert pytest.approx(tf.data_fp32()[0]) == 2.0
        assert ti.data_int8()[0] == 42

    def test_fp16_vector_and_access_neon(self):
        # FP16 path implemented on NEON backend
        with force_backend('NEON'):
            t = ln.Tensor([1.0, 2.0, 3.0], ln.QuantizationMode.FP16)
            # data_fp16 returns list of uint16 values representing fp16
            d = t.data_fp16()
            assert isinstance(d, list)
            assert len(d) == 3

    def test_int16_vector_and_access_neon(self):
        with force_backend('NEON'):
            t = ln.Tensor([1000, -2000, 3000], ln.QuantizationMode.INT16)
            d = t.data_int16()
            assert d == [1000, -2000, 3000]

class TestTensorOperations:
    def test_fp32_ops(self):
        a = ln.Tensor([1.0, 2.0, -3.0], ln.QuantizationMode.FP32)
        b = ln.Tensor([0.5, -1.0, 4.0], ln.QuantizationMode.FP32)
        c = a.add(b)
        m = a.mul(b)
        s = a.sub(b)
        assert pytest.approx(c.data_fp32()) == [1.5, 1.0, 1.0]
        assert pytest.approx(m.data_fp32()) == [0.5, -2.0, -12.0]
        assert pytest.approx(s.data_fp32()) == [0.5, 3.0, -7.0]

    def test_fp32_inplace_ops(self):
        a = ln.Tensor([1.0, 2.0, 3.0], ln.QuantizationMode.FP32)
        b = ln.Tensor([4.0, -1.0, 0.5], ln.QuantizationMode.FP32)
        a.add_inplace(b)
        a.mul_inplace(ln.Tensor([2.0, 2.0, 2.0], ln.QuantizationMode.FP32))
        a.sub_inplace(ln.Tensor([1.0, 1.0, 1.0], ln.QuantizationMode.FP32))
        assert pytest.approx(a.data_fp32()) == [9.0, 1.0, 6.0]

    def test_int8_ops(self):
        a = ln.Tensor([10, -20, 30], ln.QuantizationMode.INT8)
        b = ln.Tensor([5, 10, -40], ln.QuantizationMode.INT8)
        c = a.add(b)
        m = a.mul(b)
        s = a.sub(b)
        assert c.data_int8() == [15, -10, -10]
        # 30 * -40 = -1200 -> saturates to -128; -20*10 = -200 -> -128
        assert m.data_int8() == [50, -128, -128]
        assert s.data_int8() == [5, -30, 70]

    def test_int8_inplace_ops(self):
        a = ln.Tensor([2, 3, -4], ln.QuantizationMode.INT8)
        b = ln.Tensor([3, -5, 60], ln.QuantizationMode.INT8)
        a.add_inplace(b)
        a.mul_inplace(ln.Tensor([2, 2, 2], ln.QuantizationMode.INT8))
        a.sub_inplace(ln.Tensor([1, 1, 1], ln.QuantizationMode.INT8))
        # 2+3=5; -20+(-5)=-25? (we used -20? here it's 3 + -5 = -2); process and verify final
        assert a.data_int8() == [9, -5, 111]

    def test_fp16_ops_neon(self):
        with force_backend('NEON'):
            a = ln.Tensor([1.0, 2.0, 3.0, 4.0], ln.QuantizationMode.FP16)
            b = ln.Tensor([0.5, 2.0, -1.0, 1.0], ln.QuantizationMode.FP16)
            c = a.add(b)
            m = a.mul(b)
            s = a.sub(b)
            # We can't directly dequantize fp16 via Tensor; check sizes
            assert c.size() == 4 and m.size() == 4 and s.size() == 4

    def test_int16_ops_neon(self):
        with force_backend('NEON'):
            a = ln.Tensor([100, -200, 300, -400], ln.QuantizationMode.INT16)
            b = ln.Tensor([10, 20, -30, 40], ln.QuantizationMode.INT16)
            c = a.add(b)
            m = a.mul(b)
            s = a.sub(b)
            assert c.size() == 4 and m.size() == 4 and s.size() == 4

class TestTensorEdgeCases:
    def test_empty_fp32_operations(self):
        a = ln.Tensor([], ln.QuantizationMode.FP32)
        b = ln.Tensor([], ln.QuantizationMode.FP32)
        c = a.add(b)
        assert c.size() == 0

    def test_int8_saturation_edges(self):
        a = ln.Tensor([127, -128], ln.QuantizationMode.INT8)
        b = ln.Tensor([1, -1], ln.QuantizationMode.INT8)
        c = a.add(b)
        s = a.sub(b)
        assert c.data_int8() == [127, -128]
        assert s.data_int8() == [126, -127]

@pytest.mark.parametrize("size", [1, 8, 31, 32, 33, 63, 64, 65, 96, 97])
def test_avx2_int8_multiplication_lane_order_and_saturation(size):
    if ln.detect_backend() != "AVX2":
        pytest.skip("AVX2 backend is not available on this build and CPU")

    scenarios = [
        ([(i % 32) + 1 for i in range(size)], [2] * size),
        ([(i % 32) - 16 for i in range(size)],
         [[1, 2, -1, -2][i % 4] for i in range(size)]),
        ([([127, -128, 64, -64, 0][i % 5]) for i in range(size)],
         ([2, 2, 2, 2, -128][i % 5] for i in range(size))),
    ]

    for lhs, rhs in scenarios:
        rhs = list(rhs)
        expected = [max(-128, min(127, a * b)) for a, b in zip(lhs, rhs)]

        with force_backend("GENERIC"):
            generic_a = ln.Tensor(lhs, ln.QuantizationMode.INT8)
            generic_b = ln.Tensor(rhs, ln.QuantizationMode.INT8)
            generic_a.mul_inplace(generic_b)
            generic_result = generic_a.data_int8()

        with force_backend("AVX2"):
            avx2_a = ln.Tensor(lhs, ln.QuantizationMode.INT8)
            avx2_b = ln.Tensor(rhs, ln.QuantizationMode.INT8)
            avx2_a.mul_inplace(avx2_b)
            avx2_result = avx2_a.data_int8()

        assert generic_result == expected
        assert avx2_result == expected
