import sys

import pytest

import labneura


FP32 = labneura.QuantizationMode.FP32


@pytest.mark.parametrize(
    ("data", "shape"),
    [
        ([1.0, 2.0, 3.0], [3]),
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]),
        (list(map(float, range(1, 25))), [2, 3, 4]),
        (list(map(float, range(1, 49))), [2, 2, 3, 4]),
        ([7.0, 8.0], [1, 2, 1]),
    ],
)
def test_from_flat_preserves_arbitrary_rank_shape_and_flat_order(data, shape):
    tensor = labneura.Tensor.from_flat(data=data, shape=shape)

    assert tensor.shape == tuple(shape)
    assert tensor.ndim() == len(shape)
    assert tensor.numel() == len(data)
    assert tensor.data_fp32() == data


def test_from_flat_empty_shape_is_a_scalar():
    scalar = labneura.Tensor.from_flat([42.0], [])

    assert scalar.shape == ()
    assert scalar.ndim() == 0
    assert scalar.numel() == 1
    assert scalar.data_fp32() == [42.0]
    with pytest.raises(ValueError, match="element count"):
        labneura.Tensor.from_flat([], [])


def test_from_flat_zero_dimension_produces_empty_tensor():
    empty = labneura.Tensor.from_flat([], [2, 0, 7])

    assert empty.shape == (2, 0, 7)
    assert empty.numel() == 0
    assert empty.data_fp32() == []


def test_from_flat_supports_integer_data_and_explicit_dtype():
    tensor = labneura.Tensor.from_flat(
        [7, -3, 9, 1], [2, 2], labneura.QuantizationMode.INT8
    )

    assert tensor.shape == (2, 2)
    assert tensor.quantization_mode() == labneura.QuantizationMode.INT8
    assert tensor.data_int8() == [7, -3, 9, 1]


def test_from_flat_auto_mode_is_fp32_and_enum_modes_are_accepted():
    automatic = labneura.Tensor.from_flat([1, 2], [2], mode="auto")
    explicit = labneura.Tensor.from_flat([1.0, 2.0], [2], mode=FP32)

    assert automatic.quantization_mode() == FP32
    assert automatic.data_fp32() == [1.0, 2.0]
    assert explicit.quantization_mode() == FP32


@pytest.mark.parametrize(
    ("data", "shape"),
    [
        ([1.0, 2.0], [3]),
        ([], [1]),
        ([1.0], [0]),
    ],
)
def test_from_flat_rejects_shape_data_count_mismatch(data, shape):
    with pytest.raises(ValueError, match="element count"):
        labneura.Tensor.from_flat(data, shape)


@pytest.mark.parametrize(
    ("shape", "exception"),
    [
        (None, TypeError),
        (3, TypeError),
        ("12", TypeError),
        ([1.5], TypeError),
        ([True], TypeError),
        ([-1], ValueError),
    ],
)
def test_from_flat_rejects_invalid_shape_types(shape, exception):
    with pytest.raises(exception):
        labneura.Tensor.from_flat([], shape)


def test_from_flat_rejects_element_and_byte_count_overflow():
    with pytest.raises(OverflowError, match="element count"):
        labneura.Tensor.from_flat([], [sys.maxsize, 3])

    with pytest.raises(OverflowError, match="byte size"):
        labneura.Tensor.from_flat([], [sys.maxsize + 1])

    with pytest.raises(OverflowError, match="byte size"):
        labneura.Tensor.from_flat(
            [], [sys.maxsize + 1], mode=labneura.QuantizationMode.FP16
        )

    empty = labneura.Tensor.from_flat([], [sys.maxsize, 0, sys.maxsize])
    assert empty.shape == (sys.maxsize, 0, sys.maxsize)
    assert empty.numel() == 0


def test_from_flat_rejects_unsupported_mode():
    with pytest.raises(ValueError, match="mode"):
        labneura.Tensor.from_flat([1.0], [], mode="unknown")


def test_nd_arithmetic_works_for_matching_shapes_without_broadcasting():
    values = list(map(float, range(1, 25)))
    a = labneura.Tensor.from_flat(values, [2, 3, 4])
    b = labneura.Tensor.from_flat([2.0] * 24, [2, 3, 4])

    assert (a + b).shape == (2, 3, 4)
    assert (a + b).data_fp32() == [value + 2.0 for value in values]
    assert (a - b).data_fp32() == [value - 2.0 for value in values]
    assert (a * b).data_fp32() == [value * 2.0 for value in values]
    assert (a / b).data_fp32() == [value / 2.0 for value in values]

    in_place = labneura.Tensor.from_flat(values, [2, 3, 4])
    in_place.add_inplace(b)
    assert in_place.data_fp32() == [value + 2.0 for value in values]
    in_place = labneura.Tensor.from_flat(values, [2, 3, 4])
    in_place.sub_inplace(b)
    assert in_place.data_fp32() == [value - 2.0 for value in values]
    in_place = labneura.Tensor.from_flat(values, [2, 3, 4])
    in_place.mul_inplace(b)
    assert in_place.data_fp32() == [value * 2.0 for value in values]
    in_place = labneura.Tensor.from_flat(values, [2, 3, 4])
    in_place.div_inplace(b)
    assert in_place.data_fp32() == [value / 2.0 for value in values]

    output = labneura.Tensor.from_flat([0.0] * 24, [2, 3, 4])
    labneura.add_into(a, b, output)
    assert output.shape == (2, 3, 4)
    assert output.data_fp32() == [value + 2.0 for value in values]


def test_nd_arithmetic_rejects_different_shapes_even_when_numel_matches():
    a = labneura.Tensor.from_flat([1.0] * 24, [2, 3, 4])
    b = labneura.Tensor.from_flat([2.0] * 24, [4, 6])

    with pytest.raises(RuntimeError, match="shape mismatch"):
        a.add(b)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        a.add_inplace(b)


def test_four_dimensional_arithmetic_and_vector_tail():
    values = list(map(float, range(1, 25)))
    four_dimensional = labneura.Tensor.from_flat(values, [2, 2, 3, 2])
    twos = labneura.Tensor.from_flat([2.0] * 24, [2, 2, 3, 2])
    result = four_dimensional.add(twos)
    assert result.shape == (2, 2, 3, 2)
    assert result.data_fp32() == [value + 2.0 for value in values]

    tail_values = list(map(float, range(1, 30)))
    tail = labneura.Tensor.from_flat(tail_values, [1, 29])
    ones = labneura.Tensor.from_flat([1.0] * 29, [1, 29])
    assert tail.add(ones).data_fp32() == [value + 1.0 for value in tail_values]


def test_existing_one_dimensional_and_nested_two_dimensional_constructors_remain():
    vector = labneura.Tensor([1.0, 2.0, 3.0])
    matrix = labneura.Tensor([[1.0, 2.0], [3.0, 4.0]])

    assert vector.shape == (3,)
    assert matrix.shape == (2, 2)
    assert matrix.data_fp32() == [1.0, 2.0, 3.0, 4.0]
