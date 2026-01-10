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
