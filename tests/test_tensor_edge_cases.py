import numpy as np
import pytest
import labneura


def make_fp32(values):
    return labneura.Tensor(list(values), labneura.QuantizationMode.FP32)


def make_int8(values):
    return labneura.Tensor(list(values), labneura.QuantizationMode.INT8)


def test_detect_backend_label():
    backend = labneura.detect_backend()
    assert backend in {"AVX2", "NEON", "GENERIC"}


def test_size_mismatch_raises():
    a = make_fp32([1.0, 2.0])
    b = make_fp32([1.0])
    with pytest.raises(RuntimeError, match="size mismatch"):
        a.add_inplace(b)


def test_quantization_mode_mismatch_raises():
    a = make_fp32([1.0])
    b = make_int8([1])
    with pytest.raises(RuntimeError, match="Quantization mode mismatch"):
        a.add_inplace(b)


def test_fp32_sub_and_mul_cover_remainders():
    a = make_fp32([5.0, 7.0, -1.0, 3.5, 0.5])
    b = make_fp32([1.0, 2.0, 3.0, -1.5, 0.25])
    a.sub_inplace(b)
    np.testing.assert_allclose(a.data_fp32(), [4.0, 5.0, -4.0, 5.0, 0.25], rtol=1e-6)
    c = make_fp32([2.0, -1.0, 0.5, 4.0, -2.0])
    c.mul_inplace(make_fp32([3.0, -2.0, 2.0, 0.5, -3.0]))
    np.testing.assert_allclose(c.data_fp32(), [6.0, 2.0, 1.0, 2.0, 6.0], rtol=1e-6)


def test_int8_mul_and_sub_saturation():
    a = make_int8([10, 50, -120, -100])
    b = make_int8([12, 3, -2, 50])
    a.mul_inplace(b)
    assert a.data_int8() == [120, 127, 127, -128]
    c = make_int8([100, -100])
    d = make_int8([-50, 50])
    c.sub_inplace(d)
    assert c.data_int8() == [127, -128]


def test_zero_length_tensor_noop():
    a = labneura.Tensor()
    b = labneura.Tensor()
    assert a.size() == 0 and b.size() == 0
    a.add_inplace(b)
    assert a.data_fp32() == []


def test_quantize_dequantize_with_zero_point_and_zero_scale():
    assert labneura.quantize_int8(1.0, 0.5, zero_point=3) == 5
    assert labneura.dequantize_int8(5, 0.5, zero_point=3) == pytest.approx(1.0, rel=1e-6)
    assert labneura.quantize_int8(10.0, 0.0, zero_point=7) == 7
    assert labneura.dequantize_int8(7, 0.0, zero_point=7) == 0.0


def test_int_scalar_clamping():
    pos = labneura.Tensor(200, labneura.QuantizationMode.INT8)
    neg = labneura.Tensor(-200, labneura.QuantizationMode.INT8)
    assert pos.data_int8() == [127]
    assert neg.data_int8() == [-128]


def test_add_returns_new_tensor_independent():
    a = make_fp32([1.0, 2.0])
    b = make_fp32([3.0, 4.0])
    c = a.add(b)
    np.testing.assert_allclose(c.data_fp32(), [4.0, 6.0], rtol=1e-6)
    a.sub_inplace(b)
    np.testing.assert_allclose(a.data_fp32(), [-2.0, -2.0], rtol=1e-6)
    np.testing.assert_allclose(c.data_fp32(), [4.0, 6.0], rtol=1e-6)
