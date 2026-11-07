import numpy as np
import pytest

import labneura


FP32 = labneura.QuantizationMode.FP32


def test_add_into_writes_reusable_output_without_modifying_inputs():
    a_values = np.array([1.0, -2.0, 3.5, 4.0, -5.25], dtype=np.float32)
    b_values = np.array([2.0, 3.0, -1.5, 0.25, 5.0], dtype=np.float32)
    a = labneura.Tensor(a_values.tolist(), FP32)
    b = labneura.Tensor(b_values.tolist(), FP32)
    out = labneura.Tensor([-99.0] * len(a_values), FP32)

    labneura.add_into(a, b, out)
    np.testing.assert_array_equal(out.data_fp32(), a_values + b_values)
    np.testing.assert_array_equal(a.data_fp32(), a_values)
    np.testing.assert_array_equal(b.data_fp32(), b_values)

    labneura.add_into(a, b, out)
    np.testing.assert_array_equal(out.data_fp32(), a_values + b_values)


def test_add_into_preserves_tensor_add_allocating_behavior():
    a = labneura.Tensor([1.0, 2.0, 3.0], FP32)
    b = labneura.Tensor([4.0, 5.0, 6.0], FP32)

    result = a.add(b)

    assert result.shape == a.shape
    assert result.data_fp32() == [5.0, 7.0, 9.0]
    assert a.data_fp32() == [1.0, 2.0, 3.0]
    assert b.data_fp32() == [4.0, 5.0, 6.0]


@pytest.mark.parametrize("output_name", ["a", "b"])
def test_add_into_supports_exact_input_alias(output_name):
    a = labneura.Tensor([1.0, -2.0, 3.0], FP32)
    b = labneura.Tensor([4.0, 5.0, -6.0], FP32)
    out = a if output_name == "a" else b

    labneura.add_into(a, b, out)

    assert out.data_fp32() == [5.0, 3.0, -3.0]


def test_add_into_validates_shape_and_dtype():
    a = labneura.Tensor([1.0, 2.0, 3.0, 4.0], FP32)
    b = labneura.Tensor([5.0, 6.0, 7.0, 8.0], FP32)
    output = labneura.Tensor([0.0] * 4, FP32)

    with pytest.raises(ValueError, match="identical tensor shapes"):
        labneura.add_into(a, b, labneura.Tensor([[0.0, 0.0], [0.0, 0.0]], FP32))
    with pytest.raises(ValueError, match="identical tensor shapes"):
        labneura.add_into(a, labneura.Tensor([1.0, 2.0, 3.0], FP32), output)
    with pytest.raises(ValueError, match="FP32"):
        labneura.add_into(
            a, labneura.Tensor([1, 2, 3, 4], labneura.QuantizationMode.INT8),
            output,
        )
    with pytest.raises(ValueError, match="FP32"):
        labneura.add_into(
            a, b,
            labneura.Tensor([0, 0, 0, 0], labneura.QuantizationMode.INT8),
        )


def test_add_into_empty_tensor():
    empty_a = labneura.Tensor([], FP32)
    empty_b = labneura.Tensor([], FP32)
    empty_out = labneura.Tensor([], FP32)

    labneura.add_into(empty_a, empty_b, empty_out)
