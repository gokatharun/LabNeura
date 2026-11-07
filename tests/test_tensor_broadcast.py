import numpy as np
import pytest

import labneura


GENERIC_BACKEND = labneura.detect_backend() == "GENERIC"
FP32 = labneura.QuantizationMode.FP32

OPERATIONS = {
    "add": (np.add, "add_inplace"),
    "sub": (np.subtract, "sub_inplace"),
    "mul": (np.multiply, "mul_inplace"),
    "div": (np.divide, "div_inplace"),
}

BROADCAST_CASES = [
    ([], [3.0], [], [2.0]),
    ([3], [1.0, -2.0, 3.0], [], [2.0]),
    ([2, 3], [1, 2, 3, 4, 5, 6], [3], [10, 20, 30]),
    ([2, 3], [1, 2, 3, 4, 5, 6], [2, 3], [6, 5, 4, 3, 2, 1]),
    ([2, 1, 3], list(range(1, 7)), [1, 4, 1], [10, 20, 30, 40]),
    ([2, 3, 4], list(range(1, 25)), [4], [2, 3, 4, 5]),
    ([0, 3], [], [1, 3], [1, 2, 3]),
]


def _tensor(values, shape):
    return labneura.Tensor.from_flat(values, shape, FP32)


def _numpy(values, shape):
    return np.asarray(values, dtype=np.float32).reshape(shape)


def _int8_tensor(values, shape):
    return labneura.Tensor.from_flat(
        values, shape, labneura.QuantizationMode.INT8
    )


def _int8_numpy(values, shape):
    return np.asarray(values, dtype=np.int8).reshape(shape)


def _int16_tensor(values, shape):
    return labneura.Tensor.from_flat(
        values, shape, labneura.QuantizationMode.INT16
    )


def _int16_numpy(values, shape):
    return np.asarray(values, dtype=np.int16).reshape(shape)


def _int8_reference(op_name, lhs, rhs):
    lhs, rhs = np.broadcast_arrays(
        np.asarray(lhs, dtype=np.int16), np.asarray(rhs, dtype=np.int16)
    )
    if op_name == "add":
        result = lhs + rhs
    elif op_name == "sub":
        result = lhs - rhs
    elif op_name == "mul":
        result = lhs * rhs
    else:
        result = np.zeros(lhs.shape, dtype=np.int16)
        nonzero = rhs != 0
        result[nonzero] = np.trunc(
            lhs[nonzero].astype(np.float64) / rhs[nonzero].astype(np.float64)
        ).astype(np.int16)
    return np.clip(result, -128, 127).astype(np.int8)


def _int16_reference(op_name, lhs, rhs):
    lhs, rhs = np.broadcast_arrays(
        np.asarray(lhs, dtype=np.int64), np.asarray(rhs, dtype=np.int64)
    )
    if op_name == "add":
        result = lhs + rhs
    elif op_name == "sub":
        result = lhs - rhs
    elif op_name == "mul":
        result = lhs * rhs
    else:
        result = np.zeros(lhs.shape, dtype=np.int64)
        nonzero = rhs != 0
        result[nonzero] = np.trunc(
            lhs[nonzero].astype(np.float64) / rhs[nonzero].astype(np.float64)
        ).astype(np.int64)
    return np.clip(result, -32768, 32767).astype(np.int16)


@pytest.mark.skipif(not GENERIC_BACKEND, reason="M3.2.2 broadcasting is Generic-only")
@pytest.mark.parametrize("op_name", sorted(OPERATIONS))
@pytest.mark.parametrize(
    "lhs_shape,lhs_values,rhs_shape,rhs_values", BROADCAST_CASES
)
def test_allocating_fp32_broadcast_matches_numpy(
    op_name, lhs_shape, lhs_values, rhs_shape, rhs_values
):
    ufunc, _ = OPERATIONS[op_name]
    lhs_before = list(lhs_values)
    rhs_before = list(rhs_values)
    lhs = _tensor(lhs_values, lhs_shape)
    rhs = _tensor(rhs_values, rhs_shape)

    with np.errstate(all="ignore"):
        expected = ufunc(_numpy(lhs_values, lhs_shape), _numpy(rhs_values, rhs_shape))
    result = getattr(lhs, op_name)(rhs)

    assert result.shape == expected.shape
    assert result.quantization_mode() == FP32
    np.testing.assert_allclose(
        np.asarray(result.data_fp32(), dtype=np.float32).reshape(result.shape),
        expected,
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert lhs.data_fp32() == lhs_before
    assert rhs.data_fp32() == rhs_before


@pytest.mark.skipif(not GENERIC_BACKEND, reason="M3.2.2 broadcasting is Generic-only")
@pytest.mark.parametrize("op_name", sorted(OPERATIONS))
def test_inplace_fp32_rhs_broadcast_mutates_destination(op_name):
    ufunc, inplace_name = OPERATIONS[op_name]
    source_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    rhs_values = [2.0, 3.0, 4.0]
    destination = _tensor(source_values, [2, 3])
    rhs = _tensor(rhs_values, [3])

    with np.errstate(all="ignore"):
        expected = ufunc(
            _numpy(source_values, [2, 3]),
            _numpy(rhs_values, [3]),
        )
    getattr(destination, inplace_name)(rhs)

    assert destination.shape == (2, 3)
    np.testing.assert_allclose(
        np.asarray(destination.data_fp32(), dtype=np.float32).reshape(2, 3),
        expected,
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert rhs.data_fp32() == rhs_values


@pytest.mark.skipif(not GENERIC_BACKEND, reason="M3.2.2 broadcasting is Generic-only")
@pytest.mark.parametrize("op_name", sorted(OPERATIONS))
def test_allocating_fp32_broadcast_reads_nonzero_offset_view(op_name):
    ufunc, _ = OPERATIONS[op_name]
    original = list(map(float, range(1, 13)))
    source = _tensor(original, [4, 3])
    view = source.slice_first_axis(1, 3)
    rhs = _tensor([2.0, 3.0, 4.0], [3])
    assert view.storage_offset == 3

    with np.errstate(all="ignore"):
        expected = ufunc(
            _numpy(original, [4, 3])[1:3],
            np.asarray([2.0, 3.0, 4.0], dtype=np.float32),
        )
    result = getattr(view, op_name)(rhs)

    assert result.shape == (2, 3)
    assert result.storage_offset == 0
    np.testing.assert_array_equal(
        np.asarray(result.data_fp32(), dtype=np.float32).reshape(2, 3),
        expected,
    )
    assert source.data_fp32() == original


@pytest.mark.skipif(not GENERIC_BACKEND, reason="M3.2.2 broadcasting is Generic-only")
@pytest.mark.parametrize("op_name", sorted(OPERATIONS))
def test_inplace_fp32_broadcast_through_nonzero_offset_view(op_name):
    _, inplace_name = OPERATIONS[op_name]
    source = _tensor(list(map(float, range(1, 13))), [4, 3])
    rhs = _tensor([2.0, 3.0, 4.0], [3])
    view = source.slice_first_axis(1, 3)
    assert view.storage_offset == 3

    getattr(view, inplace_name)(rhs)
    expected_source = np.arange(1, 13, dtype=np.float32).reshape(4, 3)
    with np.errstate(all="ignore"):
        expected_source[1:3] = OPERATIONS[op_name][0](
            expected_source[1:3], np.asarray([2.0, 3.0, 4.0], dtype=np.float32)
        )
    np.testing.assert_allclose(
        np.asarray(source.data_fp32(), dtype=np.float32).reshape(4, 3),
        expected_source,
        rtol=0,
        atol=0,
        equal_nan=True,
    )


@pytest.mark.skipif(not GENERIC_BACKEND, reason="M3.2.2 broadcasting is Generic-only")
@pytest.mark.parametrize("op_name", sorted(OPERATIONS))
def test_broadcast_rejects_incompatible_shapes_and_destination_expansion(op_name):
    with pytest.raises(RuntimeError, match="shape mismatch"):
        getattr(_tensor([1.0] * 6, [2, 3]), op_name)(
            _tensor([1.0] * 12, [4, 3])
        )

    scalar = _tensor([2.0], [])
    vector = _tensor([1.0, 2.0], [2])
    _, inplace_name = OPERATIONS[op_name]
    with pytest.raises(RuntimeError, match="cannot expand"):
        getattr(scalar, inplace_name)(vector)


@pytest.mark.skipif(not GENERIC_BACKEND, reason="M3.2.2 broadcasting is Generic-only")
def test_broadcast_rejects_overlapping_scalar_alias():
    source = _tensor([1.0, 2.0, 3.0], [3])
    first_element = source.select_first_axis(0)
    with pytest.raises(ValueError, match="partially overlapping"):
        source.add_inplace(first_element)


@pytest.mark.skipif(not GENERIC_BACKEND, reason="M3.2.2 broadcasting is Generic-only")
def test_fp32_division_by_zero_nan_infinity_and_signed_zero():
    lhs_values = [0.0, -0.0, 1.0, -1.0, np.inf, np.nan]
    lhs = _tensor(lhs_values, [2, 3])
    zero = _tensor([0.0], [])
    with np.errstate(all="ignore"):
        expected = np.divide(_numpy(lhs_values, [2, 3]), np.float32(0.0))
    result = np.asarray(lhs.div(zero).data_fp32(), dtype=np.float32)
    expected = expected.reshape(-1)

    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(result), np.isinf(expected))
    np.testing.assert_array_equal(np.signbit(result[np.isinf(result)]),
                                  np.signbit(expected[np.isinf(expected)]))


@pytest.mark.skipif(not GENERIC_BACKEND, reason="M3.2.2 broadcasting is Generic-only")
@pytest.mark.parametrize("op_name", sorted(OPERATIONS))
def test_fp32_special_and_extreme_values_match_numpy(op_name):
    ufunc, _ = OPERATIONS[op_name]
    lhs_values = [
        0.0,
        -0.0,
        np.inf,
        -np.inf,
        np.nan,
        np.finfo(np.float32).tiny,
        np.finfo(np.float32).max,
    ]
    rhs_values = [0.0, -0.0, 2.0, np.inf, 1.0, 1.0e-10, 2.0]
    lhs = _tensor(lhs_values, [1, 7])
    rhs = _tensor(rhs_values, [7])
    with np.errstate(all="ignore"):
        expected = ufunc(
            _numpy(lhs_values, [1, 7]),
            _numpy(rhs_values, [7]),
        ).reshape(-1)
    actual = np.asarray(getattr(lhs, op_name)(rhs).data_fp32(), dtype=np.float32)

    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
    np.testing.assert_array_equal(
        np.signbit(actual[np.isinf(actual)]),
        np.signbit(expected[np.isinf(expected)]),
    )
    finite_nonzero = np.isfinite(expected) & (expected != 0.0)
    np.testing.assert_array_equal(actual[finite_nonzero], expected[finite_nonzero])
    zero = expected == 0.0
    np.testing.assert_array_equal(np.signbit(actual[zero]), np.signbit(expected[zero]))


@pytest.mark.skipif(not GENERIC_BACKEND, reason="M3.2.2 broadcasting is Generic-only")
def test_broadcast_result_owns_independent_storage():
    lhs = _tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    rhs = _tensor([10.0, 20.0, 30.0], [3])
    result = lhs.add(rhs)
    result.add_inplace(_tensor([1.0], []))

    assert lhs.data_fp32() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert rhs.data_fp32() == [10.0, 20.0, 30.0]


INT8_BROADCAST_CASES = [
    ([], [100], [], [50]),
    ([6], [-128, -100, -5, 5, 100, 127], [], [2]),
    ([2, 3], [100, -100, 1, -1, 127, -128], [3], [50, -50, 0]),
    ([2, 3], [-128, -100, -1, 0, 1, 100], [2, 3],
     [-1, 2, 0, 127, -128, 1]),
    ([2, 1, 3], [-128, -100, -1, 0, 1, 100], [1, 4, 1],
     [-1, 0, 2, 127]),
    ([0, 3], [], [1, 3], [1, 2, 3]),
]


@pytest.mark.skipif(not GENERIC_BACKEND, reason="INT8 broadcasting is Generic-only")
@pytest.mark.parametrize("op_name", sorted(OPERATIONS))
@pytest.mark.parametrize(
    "lhs_shape,lhs_values,rhs_shape,rhs_values", INT8_BROADCAST_CASES
)
def test_allocating_int8_broadcast_matches_saturating_reference(
    op_name, lhs_shape, lhs_values, rhs_shape, rhs_values
):
    lhs = _int8_tensor(lhs_values, lhs_shape)
    rhs = _int8_tensor(rhs_values, rhs_shape)
    lhs_before = list(lhs.data_int8())
    rhs_before = list(rhs.data_int8())
    expected = _int8_reference(
        op_name, _int8_numpy(lhs_values, lhs_shape),
        _int8_numpy(rhs_values, rhs_shape)
    )

    result = getattr(lhs, op_name)(rhs)

    assert result.shape == expected.shape
    assert result.quantization_mode() == labneura.QuantizationMode.INT8
    np.testing.assert_array_equal(
        np.asarray(result.data_int8(), dtype=np.int8).reshape(result.shape),
        expected,
    )
    assert result.is_contiguous()
    assert lhs.data_int8() == lhs_before
    assert rhs.data_int8() == rhs_before


@pytest.mark.skipif(not GENERIC_BACKEND, reason="INT8 broadcasting is Generic-only")
@pytest.mark.parametrize("op_name", sorted(OPERATIONS))
def test_inplace_int8_broadcast_mutates_destination_and_offset_view(op_name):
    _, inplace_name = OPERATIONS[op_name]
    original = list(range(1, 13))
    source = _int8_tensor(original, [4, 3])
    view = source.slice_first_axis(1, 3)
    rhs_values = [2, 3, 4]
    rhs = _int8_tensor(rhs_values, [3])
    assert view.storage_offset == 3

    expected_view = _int8_reference(
        op_name,
        _int8_numpy(original, [4, 3])[1:3],
        _int8_numpy(rhs_values, [3]),
    )
    expected_source = _int8_numpy(original, [4, 3])
    expected_source[1:3] = expected_view
    getattr(view, inplace_name)(rhs)

    np.testing.assert_array_equal(
        np.asarray(source.data_int8(), dtype=np.int8).reshape(4, 3),
        expected_source,
    )
    assert rhs.data_int8() == rhs_values

    scalar_destination = _int8_tensor([-128, -100, -1, 0, 1, 100], [2, 3])
    scalar_rhs = _int8_tensor([2], [])
    expected_scalar = _int8_reference(
        op_name,
        _int8_numpy([-128, -100, -1, 0, 1, 100], [2, 3]),
        _int8_numpy([2], []),
    )
    getattr(scalar_destination, inplace_name)(scalar_rhs)
    np.testing.assert_array_equal(
        scalar_destination.data_int8(), expected_scalar.reshape(-1).tolist()
    )


@pytest.mark.skipif(not GENERIC_BACKEND, reason="INT8 broadcasting is Generic-only")
@pytest.mark.parametrize("op_name", sorted(OPERATIONS))
def test_int8_broadcast_errors_aliasing_and_independent_storage(op_name):
    with pytest.raises(RuntimeError, match="shape mismatch"):
        getattr(_int8_tensor([1] * 6, [2, 3]), op_name)(
            _int8_tensor([1] * 12, [4, 3])
        )

    scalar = _int8_tensor([1], [])
    with pytest.raises(RuntimeError, match="cannot expand"):
        getattr(scalar, OPERATIONS[op_name][1])(
            _int8_tensor([1, 2], [2])
        )

    source = _int8_tensor([1, 2, 3], [3])
    with pytest.raises(ValueError, match="partially overlapping"):
        source.add_inplace(source.select_first_axis(0))

    lhs = _int8_tensor([100, -100, 1], [3])
    rhs = _int8_tensor([50, -50, 2], [3])
    result = lhs.add(rhs)
    result.add_inplace(_int8_tensor([1], []))
    assert result.data_int8() == [127, -127, 4]
    assert lhs.data_int8() == [100, -100, 1]
    assert rhs.data_int8() == [50, -50, 2]

    exact_alias = _int8_tensor([100, -100, 2], [3])
    exact_alias.add_inplace(exact_alias)
    assert exact_alias.data_int8() == [127, -128, 4]

    empty = _int8_tensor([], [0, 3])
    empty.add_inplace(_int8_tensor([1, 2, 3], [1, 3]))
    assert empty.shape == (0, 3)
    assert empty.data_int8() == []


@pytest.mark.skipif(not GENERIC_BACKEND, reason="INT8 broadcasting is Generic-only")
def test_int8_division_zero_truncation_and_minimum_overflow():
    lhs = _int8_tensor([-128, 127, -7, 7, 5], [5])
    rhs = _int8_tensor([-1, 0, 3, -3, 2], [5])
    assert lhs.div(rhs).data_int8() == [127, 0, -2, -2, 2]


@pytest.mark.skipif(not GENERIC_BACKEND, reason="INT8 broadcasting is Generic-only")
@pytest.mark.parametrize("op_name", sorted(OPERATIONS))
def test_int8_operation_saturation_boundaries(op_name):
    lhs_values = [127, -128, 100, -100, -128, 127]
    rhs_values = [1, -1, 50, -50, -1, 0]
    expected = _int8_reference(
        op_name, _int8_numpy(lhs_values, [2, 3]),
        _int8_numpy(rhs_values, [2, 3])
    )
    result = getattr(
        _int8_tensor(lhs_values, [2, 3]), op_name
    )(_int8_tensor(rhs_values, [2, 3]))
    np.testing.assert_array_equal(result.data_int8(), expected.reshape(-1).tolist())


INT16_BROADCAST_CASES = [
    ([], [32767], [], [1]),
    ([4], [-32768, -100, 100, 32767], [], [3]),
    ([2, 3], [32760, -32760, -7, 7, 32767, -32768], [3],
     [10, -10, 3]),
    ([2, 3], [32760, -32760, -7, 7, 32767, -32768], [1, 3],
     [1, 2, 3]),
    ([2, 3], [32760, -32760, -7, 7, 32767, -32768], [2, 1],
     [10, -10]),
    ([2, 1, 3], [-32768, -30000, -7, 0, 7, 30000], [1, 4, 1],
     [-1, 0, 2, 32767]),
    ([0, 3], [], [1, 3], [1, 2, 3]),
]


@pytest.mark.skipif(not GENERIC_BACKEND, reason="INT16 broadcasting is Generic-only")
@pytest.mark.parametrize("op_name", sorted(OPERATIONS))
@pytest.mark.parametrize(
    "lhs_shape,lhs_values,rhs_shape,rhs_values", INT16_BROADCAST_CASES
)
def test_allocating_int16_broadcast_matches_widened_reference(
    op_name, lhs_shape, lhs_values, rhs_shape, rhs_values
):
    lhs = _int16_tensor(lhs_values, lhs_shape)
    rhs = _int16_tensor(rhs_values, rhs_shape)
    lhs_before = list(lhs.data_int16())
    rhs_before = list(rhs.data_int16())
    expected = _int16_reference(
        op_name, _int16_numpy(lhs_values, lhs_shape),
        _int16_numpy(rhs_values, rhs_shape)
    )

    result = getattr(lhs, op_name)(rhs)

    assert result.shape == expected.shape
    assert result.quantization_mode() == labneura.QuantizationMode.INT16
    assert result.is_contiguous()
    assert result.storage_offset == 0
    np.testing.assert_array_equal(
        np.asarray(result.data_int16(), dtype=np.int16).reshape(result.shape),
        expected,
    )
    assert lhs.data_int16() == lhs_before
    assert rhs.data_int16() == rhs_before


@pytest.mark.skipif(not GENERIC_BACKEND, reason="INT16 broadcasting is Generic-only")
@pytest.mark.parametrize("op_name", sorted(OPERATIONS))
def test_inplace_int16_broadcast_scalar_and_offset_view(op_name):
    _, inplace_name = OPERATIONS[op_name]
    original = list(range(1, 13))
    source = _int16_tensor(original, [4, 3])
    view = source.slice_first_axis(1, 3)
    rhs_values = [32767, 3, -32768]
    rhs = _int16_tensor(rhs_values, [3])

    expected_source = _int16_numpy(original, [4, 3])
    expected_source[1:3] = _int16_reference(
        op_name, expected_source[1:3], _int16_numpy(rhs_values, [3])
    )
    getattr(view, inplace_name)(rhs)
    np.testing.assert_array_equal(
        np.asarray(source.data_int16(), dtype=np.int16).reshape(4, 3),
        expected_source,
    )
    assert rhs.data_int16() == rhs_values

    destination_values = [-32768, -30000, -7, 0, 7, 30000]
    destination = _int16_tensor(destination_values, [2, 3])
    scalar = _int16_tensor([3], [])
    expected = _int16_reference(
        op_name, _int16_numpy(destination_values, [2, 3]),
        _int16_numpy([3], [])
    )
    getattr(destination, inplace_name)(scalar)
    np.testing.assert_array_equal(
        destination.data_int16(), expected.reshape(-1).tolist()
    )


@pytest.mark.skipif(not GENERIC_BACKEND, reason="INT16 broadcasting is Generic-only")
@pytest.mark.parametrize("op_name", sorted(OPERATIONS))
def test_int16_broadcast_shape_errors_alias_and_equal_shape(op_name):
    with pytest.raises(RuntimeError, match="shape mismatch"):
        getattr(_int16_tensor([1] * 6, [2, 3]), op_name)(
            _int16_tensor([1] * 12, [4, 3])
        )

    scalar = _int16_tensor([1], [])
    with pytest.raises(RuntimeError, match="cannot expand"):
        getattr(scalar, OPERATIONS[op_name][1])(
            _int16_tensor([1, 2], [2])
        )

    source = _int16_tensor([1, 2, 3], [3])
    with pytest.raises(ValueError, match="partially overlapping"):
        source.add_inplace(source.select_first_axis(0))

    lhs_values = [32767, -32768, -7, 7, 30000, -30000]
    rhs_values = [1, -1, 3, 0, -3, 2]
    expected = _int16_reference(
        op_name, _int16_numpy(lhs_values, [2, 3]),
        _int16_numpy(rhs_values, [2, 3])
    )
    result = getattr(_int16_tensor(lhs_values, [2, 3]), op_name)(
        _int16_tensor(rhs_values, [2, 3])
    )
    np.testing.assert_array_equal(
        result.data_int16(), expected.reshape(-1).tolist()
    )

    empty = _int16_tensor([], [0, 3])
    empty.add_inplace(_int16_tensor([1, 2, 3], [1, 3]))
    assert empty.shape == (0, 3)
    assert empty.data_int16() == []


@pytest.mark.skipif(not GENERIC_BACKEND, reason="INT16 broadcasting is Generic-only")
def test_int16_division_truncation_zero_and_minimum_overflow():
    lhs = _int16_tensor([-32768, 32767, -7, 7, 32767], [5])
    rhs = _int16_tensor([-1, 0, 3, -3, 2], [5])
    assert lhs.div(rhs).data_int16() == [32767, 0, -2, -2, 16383]


def test_non_generic_backend_does_not_silently_run_generic_broadcast():
    if GENERIC_BACKEND:
        pytest.skip("This test checks non-Generic backend behavior")
    lhs = _tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    rhs = _tensor([10.0, 20.0, 30.0], [3])
    with pytest.raises(RuntimeError, match="Tensor size mismatch"):
        lhs.add(rhs)


def test_non_generic_backend_does_not_silently_run_generic_int8_broadcast():
    if GENERIC_BACKEND:
        pytest.skip("This test checks non-Generic backend behavior")
    lhs = _int8_tensor([1, 2, 3, 4, 5, 6], [2, 3])
    rhs = _int8_tensor([10, 20, 30], [3])
    with pytest.raises(RuntimeError, match="Tensor size mismatch"):
        lhs.add(rhs)


def test_non_generic_backend_does_not_silently_run_generic_int16_broadcast():
    if GENERIC_BACKEND:
        pytest.skip("This test checks non-Generic backend behavior")
    lhs = _int16_tensor([1, 2, 3, 4, 5, 6], [2, 3])
    rhs = _int16_tensor([10, 20, 30], [3])
    with pytest.raises(RuntimeError, match="Tensor size mismatch"):
        lhs.add(rhs)
