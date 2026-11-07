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
