import gc
import sys

import pytest

import labneura


FP32 = labneura.QuantizationMode.FP32


def _row_major_strides(shape):
    strides = [0] * len(shape)
    stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = stride
        stride *= shape[axis]
    return tuple(strides)


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
    assert tensor.strides == _row_major_strides(shape)
    assert tensor.storage_offset == 0
    assert tensor.is_contiguous()
    assert tensor.ndim() == len(shape)
    assert tensor.numel() == len(data)
    assert tensor.data_fp32() == data


def test_from_flat_empty_shape_is_a_scalar():
    scalar = labneura.Tensor.from_flat([42.0], [])

    assert scalar.shape == ()
    assert scalar.strides == ()
    assert scalar.storage_offset == 0
    assert scalar.is_contiguous()
    assert scalar.ndim() == 0
    assert scalar.numel() == 1
    assert scalar.data_fp32() == [42.0]
    with pytest.raises(ValueError, match="element count"):
        labneura.Tensor.from_flat([], [])


def test_from_flat_zero_dimension_produces_empty_tensor():
    empty = labneura.Tensor.from_flat([], [2, 0, 7])

    assert empty.shape == (2, 0, 7)
    assert empty.strides == (0, 7, 1)
    assert empty.storage_offset == 0
    assert empty.is_contiguous()
    assert empty.numel() == 0
    assert empty.data_fp32() == []


def test_from_flat_supports_integer_data_and_explicit_dtype():
    tensor = labneura.Tensor.from_flat(
        [7, -3, 9, 1], [2, 2], labneura.QuantizationMode.INT8
    )

    assert tensor.shape == (2, 2)
    assert tensor.strides == (2, 1)
    assert tensor.storage_offset == 0
    assert tensor.is_contiguous()
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
    assert empty.strides == (0, sys.maxsize, 1)
    assert empty.is_contiguous()
    with pytest.raises(OverflowError, match="stride"):
        labneura.Tensor.from_flat([], [0, sys.maxsize, sys.maxsize])


def test_existing_constructors_and_shape_transforms_have_contiguous_layout():
    tensors = [
        labneura.Tensor(),
        labneura.Tensor([1.0, 2.0]),
        labneura.Tensor([1, 2, 3]),
        labneura.Tensor([[1.0, 2.0], [3.0, 4.0]]),
        labneura.Tensor(1.0),
        labneura.Tensor(1),
        labneura.Tensor.from_flat([1.0] * 24, [2, 3, 4]),
    ]
    for tensor in tensors:
        assert tensor.storage_offset == 0
        assert tensor.strides == _row_major_strides(tensor.shape)
        assert tensor.is_contiguous()

    source = labneura.Tensor.from_flat(list(map(float, range(6))), [2, 3])
    transformed = [
        (source.reshape([3, 2]), (3, 2)),
        (source.flatten(), (6,)),
        (source.add(source), (2, 3)),
    ]
    for tensor, shape in transformed:
        assert tensor.shape == shape
        assert tensor.strides == _row_major_strides(shape)
        assert tensor.storage_offset == 0
        assert tensor.is_contiguous()


def test_contiguous_strides_cover_zero_and_singleton_dimensions():
    singleton = labneura.Tensor.from_flat([7.0, 8.0], [1, 2, 1])
    empty = labneura.Tensor.from_flat([], [2, 0, 3])

    assert singleton.strides == (2, 1, 1)
    assert singleton.is_contiguous()
    assert empty.strides == (0, 3, 1)
    assert empty.numel() == 0
    assert empty.is_contiguous()


def test_from_flat_rejects_unsupported_mode():
    with pytest.raises(ValueError, match="mode"):
        labneura.Tensor.from_flat([1.0], [], mode="unknown")


def test_reshape_preserves_order_and_supports_rank_changes():
    vector = labneura.Tensor.from_flat(list(range(6)), [6])
    matrix = vector.reshape([2, 3])
    volume = matrix.reshape([1, 2, 3])
    flattened = volume.flatten()

    assert matrix.shape == (2, 3)
    assert volume.shape == (1, 2, 3)
    assert flattened.shape == (6,)
    assert matrix.data_fp32() == list(map(float, range(6)))
    assert volume.data_fp32() == list(map(float, range(6)))
    assert flattened.data_fp32() == list(map(float, range(6)))


def test_reshape_same_shape_and_flatten_return_independent_storage():
    source = labneura.Tensor.from_flat(list(map(float, range(6))), [2, 3])
    same_shape = source.reshape([2, 3])
    flattened = source.flatten()
    delta = labneura.Tensor.from_flat([10.0] * 6, [2, 3])
    flattened_delta = delta.flatten()

    same_shape.add_inplace(delta)
    flattened.add_inplace(flattened_delta)

    assert source.data_fp32() == list(map(float, range(6)))
    assert same_shape.data_fp32() == [value + 10.0 for value in range(6)]
    assert flattened.shape == (6,)
    assert flattened.data_fp32() == [value + 10.0 for value in range(6)]


def test_reshape_view_shares_storage_in_both_directions_and_keeps_metadata():
    values = list(map(float, range(1, 25)))
    source = labneura.Tensor.from_flat(values, [2, 3, 4])
    first_view = source.reshape_view([4, 6])
    second_view = source.reshape_view([2, 2, 6])

    assert first_view.shape == (4, 6)
    assert first_view.strides == (6, 1)
    assert first_view.storage_offset == source.storage_offset == 0
    assert first_view.is_contiguous()
    assert first_view.numel() == source.numel() == 24
    assert first_view.quantization_mode() == source.quantization_mode()
    assert first_view.data_fp32() == values
    assert second_view.data_fp32() == values

    delta = labneura.Tensor.from_flat([10.0] * 24, [4, 6])
    first_view.add_inplace(delta)
    changed = [value + 10.0 for value in values]
    assert source.data_fp32() == changed
    assert second_view.data_fp32() == changed

    source.sub_inplace(labneura.Tensor.from_flat([10.0] * 24, [2, 3, 4]))
    assert first_view.data_fp32() == values
    assert second_view.data_fp32() == values


def test_operations_through_view_share_only_inplace_results():
    source = labneura.Tensor.from_flat(list(map(float, range(1, 14))), [13])
    view = source.reshape_view([1, 13])
    rhs = labneura.Tensor.from_flat([2.0] * 13, [1, 13])

    copied_values = view.data_fp32()
    added = view.add(rhs)
    multiplied = view.mul(rhs)
    expected_add = [value + 2.0 for value in range(1, 14)]
    expected_mul = [value * 2.0 for value in range(1, 14)]

    assert added.data_fp32() == expected_add
    assert multiplied.data_fp32() == expected_mul
    assert source.data_fp32() == list(map(float, range(1, 14)))
    added_values = added.data_fp32()
    added_values[0] = -100.0
    assert source.data_fp32()[0] == 1.0
    assert added.data_fp32()[0] == 3.0

    view.add_inplace(rhs)
    assert source.data_fp32() == expected_add
    assert view.data_fp32() == expected_add
    assert copied_values == list(map(float, range(1, 14)))


def test_reshape_view_survives_source_destruction_and_copy_is_independent():
    def make_view():
        owner = labneura.Tensor.from_flat(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]
        )
        return owner.reshape_view([3, 2])

    view = make_view()
    gc.collect()
    assert view.shape == (3, 2)
    assert view.data_fp32() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    independent = view.reshape([2, 3])
    independent.add_inplace(
        labneura.Tensor.from_flat([100.0] * 6, [2, 3])
    )
    assert view.data_fp32() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert independent.data_fp32() == [101.0, 102.0, 103.0, 104.0, 105.0, 106.0]


def test_reshape_view_preserves_scalar_empty_and_integer_dtype_semantics():
    scalar = labneura.Tensor.from_flat([42.0], [])
    scalar_view = scalar.reshape_view([1])
    assert scalar_view.shape == (1,)
    assert scalar_view.strides == (1,)
    assert scalar_view.storage_offset == 0
    assert scalar_view.data_fp32() == [42.0]

    empty = labneura.Tensor.from_flat([], [2, 0, 3])
    empty_view = empty.reshape_view([0, 6])
    assert empty_view.shape == (0, 6)
    assert empty_view.strides == (6, 1)
    assert empty_view.storage_offset == 0
    assert empty_view.is_contiguous()
    assert empty_view.numel() == 0
    assert empty_view.data_fp32() == []

    integer = labneura.Tensor.from_flat(
        [7, -3, 9, 1], [2, 2], labneura.QuantizationMode.INT8
    )
    integer_view = integer.reshape_view([4])
    assert integer_view.quantization_mode() == labneura.QuantizationMode.INT8
    assert integer_view.data_int8() == [7, -3, 9, 1]
    integer_view.add_inplace(
        labneura.Tensor.from_flat(
            [1, 2, 3, 4], [4], labneura.QuantizationMode.INT8
        )
    )
    assert integer.data_int8() == [8, -1, 12, 5]


def test_reshape_view_rejects_mismatched_and_invalid_shapes():
    source = labneura.Tensor.from_flat([1.0, 2.0, 3.0, 4.0], [2, 2])
    with pytest.raises(ValueError, match="same number"):
        source.reshape_view([3])
    with pytest.raises(OverflowError, match="element count"):
        source.reshape_view([sys.maxsize, 3])
    with pytest.raises(TypeError):
        source.reshape_view(None)
    with pytest.raises(ValueError):
        source.reshape_view([-1])


def test_reshape_scalar_and_zero_sized_tensors():
    scalar = labneura.Tensor.from_flat([42.0], [])
    reshaped_scalar = scalar.reshape([])
    one_element_scalar = scalar.reshape([1])
    flattened_scalar = scalar.flatten()

    assert reshaped_scalar.shape == ()
    assert reshaped_scalar.strides == ()
    assert reshaped_scalar.storage_offset == 0
    assert reshaped_scalar.is_contiguous()
    assert reshaped_scalar.data_fp32() == [42.0]
    assert one_element_scalar.shape == (1,)
    assert one_element_scalar.strides == (1,)
    assert one_element_scalar.storage_offset == 0
    assert one_element_scalar.data_fp32() == [42.0]
    assert flattened_scalar.shape == (1,)
    assert flattened_scalar.strides == (1,)
    assert flattened_scalar.storage_offset == 0
    assert flattened_scalar.data_fp32() == [42.0]

    empty = labneura.Tensor.from_flat([], [2, 0, 3])
    reshaped_empty = empty.reshape([0, 6])
    flattened_empty = empty.flatten()

    assert reshaped_empty.shape == (0, 6)
    assert reshaped_empty.strides == (6, 1)
    assert reshaped_empty.storage_offset == 0
    assert reshaped_empty.is_contiguous()
    assert reshaped_empty.numel() == 0
    assert flattened_empty.shape == (0,)
    assert flattened_empty.strides == (1,)
    assert flattened_empty.storage_offset == 0
    assert flattened_empty.is_contiguous()
    assert flattened_empty.data_fp32() == []


def test_reshape_preserves_dtype():
    int8 = labneura.Tensor.from_flat(
        [7, -3, 9, 1], [2, 2], labneura.QuantizationMode.INT8
    )
    reshaped_int8 = int8.reshape([4])
    assert reshaped_int8.quantization_mode() == labneura.QuantizationMode.INT8
    assert reshaped_int8.data_int8() == [7, -3, 9, 1]

    if labneura.detect_backend() == "GENERIC":
        pytest.skip("FP16 and INT16 storage are not supported by Generic")

    half = labneura.Tensor.from_flat(
        [1.0, -2.0, 3.5, 4.0], [2, 2], labneura.QuantizationMode.FP16
    )
    reshaped_half = half.reshape([4])
    assert reshaped_half.quantization_mode() == labneura.QuantizationMode.FP16
    assert reshaped_half.data_fp16() == half.data_fp16()

    int16 = labneura.Tensor.from_flat(
        [-10, 20, -30, 40], [2, 2], labneura.QuantizationMode.INT16
    )
    reshaped_int16 = int16.reshape([4])
    assert reshaped_int16.quantization_mode() == labneura.QuantizationMode.INT16


@pytest.mark.parametrize(
    "shape",
    [
        [3],
        [2, 2, 2],
        [sys.maxsize, 3],
    ],
)
def test_reshape_rejects_mismatched_or_overflowing_element_counts(shape):
    source = labneura.Tensor.from_flat([1.0, 2.0, 3.0, 4.0], [2, 2])
    expected = OverflowError if shape == [sys.maxsize, 3] else ValueError
    with pytest.raises(expected):
        source.reshape(shape)


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
def test_reshape_rejects_invalid_shape_dimensions(shape, exception):
    source = labneura.Tensor([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(exception):
        source.reshape(shape)


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
