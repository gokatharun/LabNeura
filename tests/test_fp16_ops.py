import pytest
import labneura
import numpy as np

backend = labneura.detect_backend()

@pytest.mark.skipif(backend != "NEON", reason="FP16 ops currently implemented for NEON backend")
def test_fp16_add_small():
    a = labneura.Tensor([1.0, 2.0, 3.0, 4.0], mode=labneura.QuantizationMode.FP16)
    b = labneura.Tensor([4.0, 5.0, 6.0, 7.0], mode=labneura.QuantizationMode.FP16)
    c = a.add(b)
    # Dequantize and check
    out = [labneura.dequantize_fp16(x) for x in c.data_fp16()]
    assert np.allclose(out, [5.0, 7.0, 9.0, 11.0], rtol=1e-3, atol=1e-3)

@pytest.mark.skipif(backend != "NEON", reason="FP16 ops currently implemented for NEON backend")
def test_fp16_mul_various_sizes():
    for size in [1, 3, 8, 16, 31, 64, 65]:
        a = labneura.Tensor([2.0] * size, mode=labneura.QuantizationMode.FP16)
        b = labneura.Tensor([3.0] * size, mode=labneura.QuantizationMode.FP16)
        a.mul_inplace(b)
        out = [labneura.dequantize_fp16(x) for x in a.data_fp16()]
        assert np.allclose(out, [6.0] * size, rtol=1e-3, atol=1e-3)

@pytest.mark.skipif(backend != "NEON", reason="FP16 ops currently implemented for NEON backend")
def test_fp16_sub_tails_and_alignment():
    # Choose sizes to exercise tail paths: 4, 8, 16, 32, 64 boundaries
    for size in [4, 7, 8, 15, 16, 23, 32, 63, 64, 79]:
        a = labneura.Tensor([5.0] * size, mode=labneura.QuantizationMode.FP16)
        b = labneura.Tensor([1.5] * size, mode=labneura.QuantizationMode.FP16)
        a.sub_inplace(b)
        out = [labneura.dequantize_fp16(x) for x in a.data_fp16()]
        expected = [3.5] * size
        assert np.allclose(out, expected, rtol=1e-3, atol=1e-3)

@pytest.mark.skipif(backend != "NEON", reason="FP16 ops currently implemented for NEON backend")
def test_fp16_mixed_values():
    size = 32
    a = labneura.Tensor([1.0, -1.0] * (size // 2), mode=labneura.QuantizationMode.FP16)
    b = labneura.Tensor([-2.0, 2.0] * (size // 2), mode=labneura.QuantizationMode.FP16)
    c = a.add(b)
    out = [labneura.dequantize_fp16(x) for x in c.data_fp16()]
    expected = [-1.0, 1.0] * (size // 2)
    assert np.allclose(out, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(backend != "NEON", reason="FP16 ops currently implemented for NEON backend")
def test_fp16_tiny_values_stability():
    # Tiny values exercise subnormal handling and tail paths
    tiny = 1e-4
    size = 17  # hits multiple tails (8, 4, scalar)
    a = labneura.Tensor([tiny] * size, mode=labneura.QuantizationMode.FP16)
    b = labneura.Tensor([tiny] * size, mode=labneura.QuantizationMode.FP16)
    a.add_inplace(b)
    out = [labneura.dequantize_fp16(x) for x in a.data_fp16()]
    assert np.allclose(out, [2 * tiny] * size, rtol=1e-2, atol=1e-5)


@pytest.mark.skipif(backend != "NEON", reason="FP16 ops currently implemented for NEON backend")
def test_fp16_near_max_without_overflow():
    # Values remain within FP16 finite range (~65504)
    val = 30000.0
    size = 8
    a = labneura.Tensor([val] * size, mode=labneura.QuantizationMode.FP16)
    b = labneura.Tensor([val] * size, mode=labneura.QuantizationMode.FP16)
    c = a.add(b)
    out = [labneura.dequantize_fp16(x) for x in c.data_fp16()]
    assert np.allclose(out, [val * 2] * size, rtol=1e-2, atol=1e-2)


SIMD_BACKEND = backend in ("NEON", "AVX2")

_FP16_SPECIALS = np.array(
    [0.0, -0.0, np.inf, -np.inf, np.nan,
     2.0 ** -24, -(2.0 ** -24), 3 * 2.0 ** -24, 1023 * 2.0 ** -24,
     2.0 ** -14, -(2.0 ** -14), 65504.0, -65504.0,
     1.0, -1.0, 2.0, 0.5, 3.0, 1.0 + 2.0 ** -10, 2.0 ** -11, 8.0, 16.0],
    dtype=np.float16,
)


def _fp16_tensor(values):
    return labneura.Tensor(values.astype(np.float32).tolist(), mode=labneura.QuantizationMode.FP16)


def _bits(tensor):
    return np.asarray(tensor.data_fp16()).astype(np.uint16)


def _assert_bits_match(actual_bits, expected, context):
    expected_bits = expected.view(np.uint16)
    nan_mask = np.isnan(expected)
    actual_nan = np.isnan(actual_bits.view(np.float16))
    assert np.array_equal(nan_mask, actual_nan), f"{context}: NaN lanes differ"
    assert np.array_equal(actual_bits[~nan_mask], expected_bits[~nan_mask]), (
        f"{context}: bit mismatch at {np.flatnonzero(actual_bits != expected_bits)[:8]}"
    )


_OPS = {
    "add": (np.add, "add_inplace"),
    "sub": (np.subtract, "sub_inplace"),
    "mul": (np.multiply, "mul_inplace"),
    "div": (np.divide, "div_inplace"),
}


@pytest.mark.skipif(not SIMD_BACKEND, reason="FP16 kernels require the NEON or AVX2 backend")
@pytest.mark.parametrize("size", [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65])
@pytest.mark.parametrize("op_name", sorted(_OPS))
def test_fp16_ieee_special_values_match_numpy(size, op_name):
    """Every special/special pair, rotated per size, must match IEEE binary16 (NumPy)."""
    ufunc, inplace_name = _OPS[op_name]
    pairs = np.array([(a, b) for a in _FP16_SPECIALS for b in _FP16_SPECIALS], dtype=np.float16)
    offset = (size * 37) % len(pairs)
    idx = (np.arange(size) + offset) % len(pairs)
    a, b = pairs[idx, 0], pairs[idx, 1]
    with np.errstate(all="ignore"):
        expected = ufunc(a, b)

    lhs, rhs = _fp16_tensor(a), _fp16_tensor(b)
    _assert_bits_match(_bits(getattr(lhs, op_name)(rhs)), expected, f"{op_name} allocating")
    getattr(lhs, inplace_name)(rhs)
    _assert_bits_match(_bits(lhs), expected, f"{op_name} in-place")


@pytest.mark.skipif(not SIMD_BACKEND, reason="FP16 kernels require the NEON or AVX2 backend")
@pytest.mark.parametrize("size", [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65])
@pytest.mark.parametrize("op_name", sorted(_OPS))
def test_fp16_distinct_lane_values_match_numpy(size, op_name):
    """Distinct finite values per lane expose any lane reordering."""
    ufunc, _ = _OPS[op_name]
    i = np.arange(size, dtype=np.float64)
    a = ((i * 3 + 1) * 0.5 * np.where(i % 5 == 4, -1, 1)).astype(np.float16)
    b = (((i * 7) % 13 + 1) * 0.25 * np.where(i % 3 == 2, -1, 1)).astype(np.float16)
    expected = ufunc(a, b)
    _assert_bits_match(_bits(getattr(_fp16_tensor(a), op_name)(_fp16_tensor(b))), expected, op_name)


@pytest.mark.skipif(not SIMD_BACKEND, reason="FP16 kernels require the NEON or AVX2 backend")
def test_fp16_division_by_zero_is_ieee():
    a = _fp16_tensor(np.array([1.0, -1.0, 0.0, np.inf], dtype=np.float16))
    b = _fp16_tensor(np.array([0.0, 0.0, 0.0, np.inf], dtype=np.float16))
    bits = _bits(a.div(b))
    assert bits[0] == 0x7C00 and bits[1] == 0xFC00
    assert np.isnan(bits[2:].view(np.float16)).all()
