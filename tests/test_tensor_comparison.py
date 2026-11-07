import numpy as np
import pytest

import labneura


GENERIC_DTYPES = [
    labneura.QuantizationMode.FP32,
    labneura.QuantizationMode.FP16,
    labneura.QuantizationMode.INT16,
    labneura.QuantizationMode.INT8,
]

COMPARISONS = {
    "eq": np.equal,
    "ne": np.not_equal,
    "lt": np.less,
    "le": np.less_equal,
    "gt": np.greater,
    "ge": np.greater_equal,
}

NUMPY_DTYPES = {
    labneura.QuantizationMode.FP32: np.float32,
    labneura.QuantizationMode.FP16: np.float16,
    labneura.QuantizationMode.INT16: np.int16,
    labneura.QuantizationMode.INT8: np.int8,
}


def _force_generic(monkeypatch):
    monkeypatch.setenv("LABNEURA_BACKEND", "GENERIC")
    assert labneura.detect_backend() == "GENERIC"


def _force_neon(monkeypatch):
    monkeypatch.setenv("LABNEURA_BACKEND", "NEON")
    try:
        selected = labneura.detect_backend()
    except (RuntimeError, ValueError) as error:
        pytest.skip(f"NEON backend is unavailable: {error}")
    if selected != "NEON":
        pytest.skip("NEON backend is unavailable on this host")


def _tensor(values, shape, dtype):
    return labneura.Tensor.from_flat(values, shape, dtype)


def _assert_bool_result(result, expected, shape):
    assert result.quantization_mode() == labneura.QuantizationMode.BOOL
    assert result.shape == shape
    assert result.numel() == np.asarray(expected).size
    assert result.is_contiguous()
    assert result.storage_offset == 0
    actual = result.data_bool()
    assert actual == np.asarray(expected, dtype=np.bool_).reshape(-1).tolist()
    assert all(type(value) is bool for value in actual)


def _data(tensor, dtype):
    if dtype == labneura.QuantizationMode.FP32:
        return tensor.data_fp32()
    if dtype == labneura.QuantizationMode.FP16:
        return tensor.data_fp16()
    if dtype == labneura.QuantizationMode.INT16:
        return tensor.data_int16()
    return tensor.data_int8()


@pytest.mark.parametrize("operation,numpy_operation", COMPARISONS.items())
@pytest.mark.parametrize("dtype", GENERIC_DTYPES)
def test_generic_comparisons_all_dtypes_shapes_and_broadcast(
    monkeypatch, operation, numpy_operation, dtype
):
    _force_generic(monkeypatch)
    numpy_dtype = NUMPY_DTYPES[dtype]
    lhs_values = [-3, -1, 0, 1, 3]
    rhs_values = [-1, -1, 0, -2, 4]
    lhs = _tensor(lhs_values, [5], dtype)
    rhs = _tensor(rhs_values, [5], dtype)
    lhs_before = list(_data(lhs, dtype))
    rhs_before = list(_data(rhs, dtype))

    expected = numpy_operation(
        np.asarray(lhs_values, dtype=numpy_dtype),
        np.asarray(rhs_values, dtype=numpy_dtype),
    )
    result = getattr(lhs, operation)(rhs)
    _assert_bool_result(result, expected, (5,))
    assert _data(lhs, dtype) == lhs_before
    assert _data(rhs, dtype) == rhs_before

    matrix_values = [1, 4, 7, 2, 5, 8]
    row_values = [2, 5, 9]
    matrix = _tensor(matrix_values, [2, 3], dtype)
    row = _tensor(row_values, [3], dtype)
    broadcast_expected = numpy_operation(
        np.asarray(matrix_values, dtype=numpy_dtype).reshape(2, 3),
        np.asarray(row_values, dtype=numpy_dtype),
    )
    broadcast = getattr(matrix, operation)(row)
    _assert_bool_result(broadcast, broadcast_expected, (2, 3))

    scalar = _tensor([2], [], dtype)
    scalar_result = getattr(matrix, operation)(scalar)
    scalar_expected = numpy_operation(
        np.asarray(matrix_values, dtype=numpy_dtype).reshape(2, 3),
        np.asarray(2, dtype=numpy_dtype),
    )
    _assert_bool_result(scalar_result, scalar_expected, (2, 3))


@pytest.mark.parametrize("operation,numpy_operation", COMPARISONS.items())
def test_generic_comparison_high_rank_broadcast(
    monkeypatch, operation, numpy_operation
):
    _force_generic(monkeypatch)
    lhs_values = [1, 4, 7, 2]
    rhs_values = [0, 4, 8, 2, 5, 6, 9, 3]
    lhs = _tensor(lhs_values, [2, 1, 2, 1], labneura.QuantizationMode.INT16)
    rhs = _tensor(rhs_values, [1, 2, 1, 4], labneura.QuantizationMode.INT16)
    expected = numpy_operation(
        np.asarray(lhs_values, dtype=np.int16).reshape(2, 1, 2, 1),
        np.asarray(rhs_values, dtype=np.int16).reshape(1, 2, 1, 4),
    )

    result = getattr(lhs, operation)(rhs)
    _assert_bool_result(result, expected, (2, 2, 2, 4))


@pytest.mark.parametrize("operation,numpy_operation", COMPARISONS.items())
def test_generic_comparison_three_dimensional_broadcast(
    monkeypatch, operation, numpy_operation
):
    _force_generic(monkeypatch)
    lhs_values = [1, 3, 5, 7, 9, 11]
    rhs_values = [0, 4, 8, 2]
    lhs = _tensor(lhs_values, [2, 1, 3], labneura.QuantizationMode.INT8)
    rhs = _tensor(rhs_values, [1, 4, 1], labneura.QuantizationMode.INT8)
    expected = numpy_operation(
        np.asarray(lhs_values, dtype=np.int8).reshape(2, 1, 3),
        np.asarray(rhs_values, dtype=np.int8).reshape(1, 4, 1),
    )

    result = getattr(lhs, operation)(rhs)
    _assert_bool_result(result, expected, (2, 4, 3))


@pytest.mark.parametrize("operation,numpy_operation", COMPARISONS.items())
@pytest.mark.parametrize(
    "dtype,numpy_dtype",
    [
        (labneura.QuantizationMode.FP32, np.float32),
        (labneura.QuantizationMode.FP16, np.float16),
    ],
)
def test_generic_float_comparison_special_values(
    monkeypatch, operation, numpy_operation, dtype, numpy_dtype
):
    _force_generic(monkeypatch)
    lhs_values = [np.nan, -np.inf, np.inf, -0.0, 0.0]
    rhs_values = [np.nan, -np.inf, np.inf, 0.0, -0.0]
    lhs = _tensor(lhs_values, [5], dtype)
    rhs = _tensor(rhs_values, [5], dtype)
    expected = numpy_operation(
        np.asarray(lhs_values, dtype=numpy_dtype),
        np.asarray(rhs_values, dtype=numpy_dtype),
    )

    result = getattr(lhs, operation)(rhs)
    _assert_bool_result(result, expected, (5,))


@pytest.mark.parametrize("operation,numpy_operation", COMPARISONS.items())
def test_generic_comparison_zero_sized_and_offset_view(
    monkeypatch, operation, numpy_operation
):
    _force_generic(monkeypatch)
    dtype = labneura.QuantizationMode.FP32

    empty = _tensor([], [0, 3], dtype)
    row = _tensor([1, 2, 3], [1, 3], dtype)
    empty_result = getattr(empty, operation)(row)
    _assert_bool_result(
        empty_result,
        np.empty((0, 3), dtype=np.bool_),
        (0, 3),
    )

    values = list(range(12))
    source = _tensor(values, [4, 3], dtype)
    view = source.slice_first_axis(1, 3)
    rhs = _tensor([4, 5, 6], [3], dtype)
    assert view.storage_offset == 3
    before = list(source.data_fp32())
    expected = numpy_operation(
        np.asarray(values, dtype=np.float32).reshape(4, 3)[1:3],
        np.asarray([4, 5, 6], dtype=np.float32),
    )

    result = getattr(view, operation)(rhs)
    _assert_bool_result(result, expected, (2, 3))
    assert source.data_fp32() == before


def test_generic_comparison_errors_and_bool_input_policy(monkeypatch):
    _force_generic(monkeypatch)
    fp32 = labneura.QuantizationMode.FP32
    int8 = labneura.QuantizationMode.INT8
    lhs = _tensor([1, 2, 3, 4], [2, 2], fp32)

    with pytest.raises(RuntimeError, match="Tensor shape mismatch"):
        lhs.eq(_tensor([1, 2, 3, 4, 5, 6], [2, 3], fp32))
    with pytest.raises(RuntimeError, match="Quantization mode mismatch"):
        lhs.eq(_tensor([1, 2, 3, 4], [2, 2], int8))

    bool_dtype = labneura.QuantizationMode.BOOL
    bool_lhs = labneura.Tensor.from_flat([False, True], [2], bool_dtype)
    bool_rhs = labneura.Tensor.from_flat([True, True], [2], bool_dtype)
    with pytest.raises(RuntimeError, match="not supported for BOOL"):
        bool_lhs.eq(bool_rhs)


def test_comparison_api_does_not_depend_on_python_operators(monkeypatch):
    _force_generic(monkeypatch)
    lhs = _tensor([1, 2], [2], labneura.QuantizationMode.INT8)
    result = lhs.lt(_tensor([2, 1], [2], labneura.QuantizationMode.INT8))
    _assert_bool_result(result, [True, False], (2,))


@pytest.mark.parametrize("operation,numpy_operation", COMPARISONS.items())
@pytest.mark.parametrize("dtype", GENERIC_DTYPES)
def test_neon_comparison_parity_all_dtypes_and_broadcasts(
    monkeypatch, operation, numpy_operation, dtype
):
    numpy_dtype = NUMPY_DTYPES[dtype]
    if dtype == labneura.QuantizationMode.INT8:
        lhs_values = [-128, -127, -100, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                      126, 127, -128]
        rhs_values = [127, -128, -99, -2, 0, 1, 0, 2, 4, 3, 5, 7, 6, 9,
                      127, 126, 127]
    elif dtype == labneura.QuantizationMode.INT16:
        lhs_values = [-32768, -32767, -100, -2, -1, 0, 1, 2, 3, 4, 5, 6,
                      7, 8, 32766, 32767, -32768]
        rhs_values = [32767, -32768, -99, -2, 0, 1, 0, 2, 4, 3, 5, 7, 6,
                      9, 32767, 32766, 32767]
    else:
        lhs_values = [-8.0, -4.5, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0,
                      3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        rhs_values = [-7.0, -4.5, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 2.0,
                      4.0, 4.0, 6.0, 5.0, 8.0, 7.0, 10.0, 10.0]

    lhs_reference = np.asarray(lhs_values, dtype=numpy_dtype)
    rhs_reference = np.asarray(rhs_values, dtype=numpy_dtype)
    expected = numpy_operation(lhs_reference, rhs_reference)

    _force_generic(monkeypatch)
    generic_lhs = _tensor(lhs_values, [17], dtype)
    generic_rhs = _tensor(rhs_values, [17], dtype)
    generic_result = getattr(generic_lhs, operation)(generic_rhs)
    _assert_bool_result(generic_result, expected, (17,))
    generic_values = generic_result.data_bool()

    _force_neon(monkeypatch)
    neon_lhs = _tensor(lhs_values, [17], dtype)
    neon_rhs = _tensor(rhs_values, [17], dtype)
    lhs_before = list(_data(neon_lhs, dtype))
    rhs_before = list(_data(neon_rhs, dtype))
    neon_result = getattr(neon_lhs, operation)(neon_rhs)
    _assert_bool_result(neon_result, expected, (17,))
    assert neon_result.data_bool() == generic_values
    assert _data(neon_lhs, dtype) == lhs_before
    assert _data(neon_rhs, dtype) == rhs_before
    assert labneura.detect_backend() == "NEON"

    for size in (3, 15, 16, 17, 31, 32, 33):
        boundary_lhs = np.resize(lhs_reference, size)
        boundary_rhs = np.resize(rhs_reference, size)
        boundary_expected = numpy_operation(boundary_lhs, boundary_rhs)

        _force_generic(monkeypatch)
        generic_boundary = getattr(
            _tensor(boundary_lhs.tolist(), [size], dtype), operation
        )(_tensor(boundary_rhs.tolist(), [size], dtype))
        _assert_bool_result(generic_boundary, boundary_expected, (size,))

        _force_neon(monkeypatch)
        neon_boundary = getattr(
            _tensor(boundary_lhs.tolist(), [size], dtype), operation
        )(_tensor(boundary_rhs.tolist(), [size], dtype))
        _assert_bool_result(neon_boundary, boundary_expected, (size,))
        assert neon_boundary.data_bool() == generic_boundary.data_bool()

    broadcast_lhs_values = lhs_values[:6]
    broadcast_rhs_values = rhs_values[:4]
    broadcast_expected = numpy_operation(
        np.asarray(broadcast_lhs_values, dtype=numpy_dtype).reshape(2, 1, 3),
        np.asarray(broadcast_rhs_values, dtype=numpy_dtype).reshape(1, 4, 1),
    )
    _force_generic(monkeypatch)
    generic_broadcast = getattr(
        _tensor(broadcast_lhs_values, [2, 1, 3], dtype), operation
    )(_tensor(broadcast_rhs_values, [1, 4, 1], dtype))
    _assert_bool_result(generic_broadcast, broadcast_expected, (2, 4, 3))

    _force_neon(monkeypatch)
    neon_broadcast = getattr(
        _tensor(broadcast_lhs_values, [2, 1, 3], dtype), operation
    )(_tensor(broadcast_rhs_values, [1, 4, 1], dtype))
    _assert_bool_result(neon_broadcast, broadcast_expected, (2, 4, 3))
    assert neon_broadcast.data_bool() == generic_broadcast.data_bool()

    scalar_expected = numpy_operation(
        np.asarray(lhs_values, dtype=numpy_dtype),
        np.asarray(2, dtype=numpy_dtype),
    )
    _force_generic(monkeypatch)
    generic_scalar = getattr(_tensor(lhs_values, [17], dtype), operation)(
        _tensor([2], [], dtype)
    )
    _assert_bool_result(generic_scalar, scalar_expected, (17,))
    _force_neon(monkeypatch)
    neon_scalar = getattr(_tensor(lhs_values, [17], dtype), operation)(
        _tensor([2], [], dtype)
    )
    _assert_bool_result(neon_scalar, scalar_expected, (17,))
    assert neon_scalar.data_bool() == generic_scalar.data_bool()


@pytest.mark.parametrize("operation,numpy_operation", COMPARISONS.items())
@pytest.mark.parametrize(
    "dtype,numpy_dtype",
    [
        (labneura.QuantizationMode.FP32, np.float32),
        (labneura.QuantizationMode.FP16, np.float16),
    ],
)
def test_neon_float_comparison_special_values(
    monkeypatch, operation, numpy_operation, dtype, numpy_dtype
):
    lhs_values = [np.nan, -np.inf, np.inf, -0.0, 0.0]
    rhs_values = [np.nan, -np.inf, np.inf, 0.0, -0.0]
    expected = numpy_operation(
        np.asarray(lhs_values, dtype=numpy_dtype),
        np.asarray(rhs_values, dtype=numpy_dtype),
    )

    _force_generic(monkeypatch)
    generic = getattr(_tensor(lhs_values, [5], dtype), operation)(
        _tensor(rhs_values, [5], dtype)
    )
    _assert_bool_result(generic, expected, (5,))

    _force_neon(monkeypatch)
    neon = getattr(_tensor(lhs_values, [5], dtype), operation)(
        _tensor(rhs_values, [5], dtype)
    )
    _assert_bool_result(neon, expected, (5,))
    assert neon.data_bool() == generic.data_bool()


@pytest.mark.parametrize("operation,numpy_operation", COMPARISONS.items())
@pytest.mark.parametrize("dtype", GENERIC_DTYPES)
def test_neon_zero_sized_and_offset_view_broadcasts(
    monkeypatch, operation, numpy_operation, dtype
):
    numpy_dtype = NUMPY_DTYPES[dtype]
    values = np.arange(12, dtype=numpy_dtype)
    row_values = np.asarray([4, 5, 6], dtype=numpy_dtype)
    expected = numpy_operation(values.reshape(4, 3)[1:3], row_values)

    _force_neon(monkeypatch)
    empty = _tensor([], [0, 3], dtype)
    row = _tensor([1, 2, 3], [1, 3], dtype)
    empty_result = getattr(empty, operation)(row)
    _assert_bool_result(
        empty_result, np.empty((0, 3), dtype=np.bool_), (0, 3)
    )

    source = _tensor(values.tolist(), [4, 3], dtype)
    view = source.slice_first_axis(1, 3)
    rhs = _tensor(row_values.tolist(), [3], dtype)
    assert view.storage_offset == 3
    source_before = list(_data(source, dtype))
    result = getattr(view, operation)(rhs)
    _assert_bool_result(result, expected, (2, 3))
    assert _data(source, dtype) == source_before
    assert labneura.detect_backend() == "NEON"
