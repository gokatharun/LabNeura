import pytest
import numpy as np
import labneura


# Helpers

def fp32(vals):
    return labneura.Tensor(list(vals), labneura.QuantizationMode.FP32)


def int8(vals):
    return labneura.Tensor(list(vals), labneura.QuantizationMode.INT8)


def test_default_tensor_empty():
    t = labneura.Tensor()
    assert t.size() == 0
    assert t.data_fp32() == []


def test_constructor_int_vector_fp32_and_int8_clamp():
    t_fp32 = labneura.Tensor([1, -2, 3], labneura.QuantizationMode.FP32)
    np.testing.assert_allclose(t_fp32.data_fp32(), [1.0, -2.0, 3.0])

    t_int8 = labneura.Tensor([200, -200, 0], labneura.QuantizationMode.INT8)
    assert t_int8.data_int8() == [127, -128, 0]


def test_scalar_constructors_int_and_float_modes():
    t_float = labneura.Tensor(3.5)
    np.testing.assert_allclose(t_float.data_fp32(), [3.5])

    t_int_fp32 = labneura.Tensor(5, labneura.QuantizationMode.FP32)
    np.testing.assert_allclose(t_int_fp32.data_fp32(), [5.0])

    t_int_int8 = labneura.Tensor(5, labneura.QuantizationMode.INT8)
    assert t_int_int8.data_int8() == [5]


def test_copy_semantics_clone_backend():
    a = fp32([1.0, 2.0, 3.0])
    # Python bindings don't expose a Tensor-from-Tensor ctor; simulate copy via data
    b = labneura.Tensor(a.data_fp32(), labneura.QuantizationMode.FP32)
    a.add_inplace(fp32([1.0, 1.0, 1.0]))
    # b should remain unchanged after a mutates
    np.testing.assert_allclose(b.data_fp32(), [1.0, 2.0, 3.0])


def test_data_access_mode_errors():
    a = fp32([1.0])
    with pytest.raises(RuntimeError, match="not in INT8"):
        a.data_int8()

    b = int8([1])
    with pytest.raises(RuntimeError, match="not in FP32"):
        b.data_fp32()


def test_fp32_sub_mul_remainders_and_independent_add():
    a = fp32([5.0, 7.0, -1.0, 3.5, 0.5])
    b = fp32([1.0, 2.0, 3.0, -1.5, 0.25])
    a.sub_inplace(b)
    np.testing.assert_allclose(a.data_fp32(), [4.0, 5.0, -4.0, 5.0, 0.25], rtol=1e-6)

    c = fp32([2.0, -1.0, 0.5, 4.0, -2.0])
    c.mul_inplace(fp32([3.0, -2.0, 2.0, 0.5, -3.0]))
    np.testing.assert_allclose(c.data_fp32(), [6.0, 2.0, 1.0, 2.0, 6.0], rtol=1e-6)

    # add returns a new tensor and should not mutate operands
    x = fp32([1.0, 2.0])
    y = fp32([3.0, 4.0])
    z = x.add(y)
    np.testing.assert_allclose(z.data_fp32(), [4.0, 6.0], rtol=1e-6)
    np.testing.assert_allclose(x.data_fp32(), [1.0, 2.0], rtol=1e-6)
    np.testing.assert_allclose(y.data_fp32(), [3.0, 4.0], rtol=1e-6)


def test_quantize_dequantize_edges():
    assert labneura.quantize_int8(1.0, 0.5, zero_point=3) == 5
    assert labneura.dequantize_int8(5, 0.5, zero_point=3) == pytest.approx(1.0, rel=1e-6)
    assert labneura.quantize_int8(10.0, 0.0, zero_point=7) == 7
    assert labneura.dequantize_int8(7, 0.0, zero_point=7) == 0.0


def test_detect_backend_invocation():
    backend = labneura.detect_backend()
    assert backend in {"AVX2", "NEON", "GENERIC"}
