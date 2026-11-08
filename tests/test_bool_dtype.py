import copy

import pytest

import labneura


BOOL = labneura.QuantizationMode.BOOL


def _force_generic(monkeypatch):
    monkeypatch.setenv("LABNEURA_BACKEND", "GENERIC")
    assert labneura.detect_backend() == "GENERIC"


def test_bool_sequence_construction_and_metadata(monkeypatch):
    _force_generic(monkeypatch)
    tensor = labneura.Tensor([True, False, True], dtype=BOOL)

    assert tensor.shape == (3,)
    assert tensor.numel() == 3
    assert tensor.size() == 3
    assert tensor.quantization_mode() == BOOL
    values = tensor.data_bool()
    assert values == [True, False, True]
    assert all(type(value) is bool for value in values)


def test_bool_empty_and_zero_sized_shapes(monkeypatch):
    _force_generic(monkeypatch)
    empty = labneura.Tensor([], dtype=BOOL)
    empty_nd = labneura.Tensor.from_flat([], [0, 3], BOOL)

    assert empty.shape == (0,)
    assert empty.numel() == 0
    assert empty.quantization_mode() == BOOL
    assert empty.data_bool() == []
    assert empty_nd.shape == (0, 3)
    assert empty_nd.numel() == 0
    assert empty_nd.quantization_mode() == BOOL
    assert empty_nd.data_bool() == []


def test_bool_from_flat_and_copy_preserve_dtype_and_data(monkeypatch):
    _force_generic(monkeypatch)
    original = labneura.Tensor.from_flat(
        [True, False, True, False], [2, 2], BOOL
    )
    duplicate = copy.copy(original)

    assert duplicate.shape == original.shape
    assert duplicate.numel() == original.numel()
    assert duplicate.quantization_mode() == BOOL
    assert duplicate.data_bool() == original.data_bool()


def test_bool_constructor_normalizes_nonzero_values(monkeypatch):
    _force_generic(monkeypatch)
    tensor = labneura.Tensor([0, 1, -3, 0.0, 0.5], dtype=BOOL)
    from_flat = labneura.Tensor.from_flat([0, 2, -1], [3], BOOL)

    assert tensor.data_bool() == [False, True, True, False, True]
    assert from_flat.data_bool() == [False, True, True]


def test_bool_constructor_rejects_non_numeric_values(monkeypatch):
    _force_generic(monkeypatch)
    with pytest.raises(TypeError, match="bool or numeric values"):
        labneura.Tensor(["false"], dtype=BOOL)

    with pytest.raises(TypeError, match="bool or numeric values"):
        labneura.Tensor.from_flat([None], [1], BOOL)


def test_bool_constructor_requires_bool_dtype_keyword(monkeypatch):
    _force_generic(monkeypatch)
    with pytest.raises(TypeError, match="requires BOOL dtype"):
        labneura.Tensor([True, False], dtype=labneura.QuantizationMode.INT8)


def test_bool_backend_selection_is_explicit_and_preserved(monkeypatch):
    _force_generic(monkeypatch)
    tensor = labneura.Tensor([True, False], dtype=BOOL)
    duplicate = copy.copy(tensor)

    assert labneura.detect_backend() == "GENERIC"
    assert tensor.quantization_mode() == BOOL
    assert duplicate.quantization_mode() == BOOL
    assert duplicate.data_bool() == [True, False]


def test_bool_arithmetic_is_not_implicitly_enabled(monkeypatch):
    _force_generic(monkeypatch)
    tensor = labneura.Tensor([True, False], dtype=BOOL)
    with pytest.raises(RuntimeError, match="not supported for BOOL"):
        tensor.add(tensor)


def test_generic_logical_not_values_scalars_and_empty(monkeypatch):
    _force_generic(monkeypatch)
    mixed = labneura.Tensor.from_flat(
        [True, False, True, False], [2, 2], BOOL
    )
    before = mixed.data_bool()
    result = mixed.logical_not()

    assert result is not mixed
    assert result.quantization_mode() == BOOL
    assert result.shape == (2, 2)
    assert result.is_contiguous()
    assert result.data_bool() == [False, True, False, True]
    assert mixed.data_bool() == before

    assert labneura.Tensor([False, False], dtype=BOOL).logical_not().data_bool() == [
        True,
        True,
    ]
    assert labneura.Tensor([True, True], dtype=BOOL).logical_not().data_bool() == [
        False,
        False,
    ]
    scalar_false = labneura.Tensor.from_flat([False], [], BOOL).logical_not()
    scalar_true = labneura.Tensor.from_flat([True], [], BOOL).logical_not()
    assert scalar_false.shape == ()
    assert scalar_false.data_bool() == [True]
    assert scalar_true.shape == ()
    assert scalar_true.data_bool() == [False]

    empty = labneura.Tensor.from_flat([], [0, 3], BOOL)
    empty_result = empty.logical_not()
    assert empty_result.shape == (0, 3)
    assert empty_result.numel() == 0
    assert empty_result.data_bool() == []


def test_generic_logical_not_offset_view_repeated_and_input_immutable(
    monkeypatch,
):
    _force_generic(monkeypatch)
    source = labneura.Tensor.from_flat(
        [False, True, True, False, False, False, True, True],
        [4, 2],
        BOOL,
    )
    view = source.slice_first_axis(1, 3)
    assert view.storage_offset == 2
    source_before = source.data_bool()

    first = view.logical_not()
    second = view.logical_not()
    assert first.shape == (2, 2)
    assert first.is_contiguous()
    assert first.data_bool() == [False, True, True, True]
    assert second.data_bool() == first.data_bool()
    assert second is not first
    assert source.data_bool() == source_before
    assert labneura.detect_backend() == "GENERIC"


def test_logical_not_rejects_non_bool_input(monkeypatch):
    _force_generic(monkeypatch)
    numeric = labneura.Tensor([0.0, 1.0])
    with pytest.raises(RuntimeError, match="requires a BOOL input tensor"):
        numeric.logical_not()


def test_avx2_logical_not_values_vector_tails_views_and_storage(monkeypatch):
    monkeypatch.setenv("LABNEURA_BACKEND", "AVX2")
    try:
        selected = labneura.detect_backend()
    except (RuntimeError, ValueError) as error:
        pytest.skip(f"AVX2 backend unavailable: {error}")
    if selected != "AVX2":
        pytest.skip("AVX2 backend unavailable on this host")

    for size in range(67):
        values = [index % 2 == 0 for index in range(size)]
        tensor = labneura.Tensor(values, dtype=BOOL)
        before = tensor.data_bool()
        result = tensor.logical_not()

        assert labneura.detect_backend() == "AVX2"
        assert result.quantization_mode() == BOOL
        assert result.shape == (size,)
        assert result.is_contiguous()
        assert result.data_bool() == [not value for value in values]
        assert all(type(value) is bool for value in result.data_bool())
        assert tensor.data_bool() == before

    assert labneura.Tensor([False] * 64, dtype=BOOL).logical_not().data_bool() == [
        True
    ] * 64
    assert labneura.Tensor([True] * 64, dtype=BOOL).logical_not().data_bool() == [
        False
    ] * 64

    for value in (False, True):
        scalar = labneura.Tensor.from_flat([value], [], BOOL)
        result = scalar.logical_not()
        assert result.shape == ()
        assert result.data_bool() == [not value]

    empty = labneura.Tensor.from_flat([], [0, 3], BOOL)
    empty_result = empty.logical_not()
    assert empty_result.shape == (0, 3)
    assert empty_result.numel() == 0
    assert empty_result.data_bool() == []

    source_values = [index % 2 == 0 for index in range(66)]
    source = labneura.Tensor.from_flat(source_values, [2, 33], BOOL)
    view = source.slice_first_axis(1, 2)
    source_before = source.data_bool()
    first = view.logical_not()
    second = view.logical_not()
    assert view.storage_offset == 33
    assert first.shape == (1, 33)
    assert first.data_bool() == [not value for value in source_values[33:]]
    assert second.data_bool() == first.data_bool()
    assert second is not first
    assert source.data_bool() == source_before


def test_neon_logical_not_values_vector_tails_views_and_storage(monkeypatch):
    monkeypatch.setenv("LABNEURA_BACKEND", "NEON")
    try:
        selected = labneura.detect_backend()
    except (RuntimeError, ValueError) as error:
        pytest.skip(f"NEON backend unavailable: {error}")
    if selected != "NEON":
        pytest.skip("NEON backend unavailable on this host")

    for size in range(34):
        values = [index % 2 == 0 for index in range(size)]
        tensor = labneura.Tensor(values, dtype=BOOL)
        before = tensor.data_bool()
        result = tensor.logical_not()

        assert labneura.detect_backend() == "NEON"
        assert result.quantization_mode() == BOOL
        assert result.shape == (size,)
        assert result.is_contiguous()
        assert result.data_bool() == [not value for value in values]
        assert all(type(value) is bool for value in result.data_bool())
        assert tensor.data_bool() == before

    assert labneura.Tensor([False] * 32, dtype=BOOL).logical_not().data_bool() == [
        True
    ] * 32
    assert labneura.Tensor([True] * 32, dtype=BOOL).logical_not().data_bool() == [
        False
    ] * 32

    for value in (False, True):
        scalar = labneura.Tensor.from_flat([value], [], BOOL)
        result = scalar.logical_not()
        assert result.shape == ()
        assert result.data_bool() == [not value]

    empty = labneura.Tensor.from_flat([], [0, 3], BOOL)
    empty_result = empty.logical_not()
    assert empty_result.shape == (0, 3)
    assert empty_result.numel() == 0
    assert empty_result.data_bool() == []

    source_values = [index % 2 == 0 for index in range(34)]
    source = labneura.Tensor.from_flat(source_values, [2, 17], BOOL)
    view = source.slice_first_axis(1, 2)
    source_before = source.data_bool()
    first = view.logical_not()
    second = view.logical_not()
    assert view.storage_offset == 17
    assert first.shape == (1, 17)
    assert first.data_bool() == [not value for value in source_values[17:]]
    assert second.data_bool() == first.data_bool()
    assert second is not first
    assert source.data_bool() == source_before
