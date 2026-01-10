import labneura
import numpy as np


def test_int16_zero_scale_guard():
    q = labneura.quantize_int16(5.0, 0.0)
    assert q == 0
    dq = labneura.dequantize_int16(q, 0.0)
    assert dq == 0.0


def test_int16_roundtrip_basic():
    scale = 0.5
    values = [-10.0, -1.0, 0.0, 1.0, 10.0]
    for v in values:
        q = labneura.quantize_int16(v, scale)
        dq = labneura.dequantize_int16(q, scale)
        assert np.isclose(dq, v, atol=scale), f"roundtrip failed for {v}: {dq}"


def test_int16_saturation_limits():
    scale = 1.0
    q_max = labneura.quantize_int16(40000.0, scale)
    q_min = labneura.quantize_int16(-40000.0, scale)
    assert q_max == 32767
    assert q_min == -32768


def test_int16_zero_point_offset():
    scale = 2.0
    zp = 100
    q = labneura.quantize_int16(10.0, scale, zero_point=zp)
    dq = labneura.dequantize_int16(q, scale, zero_point=zp)
    assert np.isclose(dq, 10.0, atol=scale)


def test_int16_random_roundtrip():
    np.random.seed(0)
    scale = 0.25
    zp = -50
    for _ in range(50):
        v = float(np.random.uniform(-1000, 1000))
        q = labneura.quantize_int16(v, scale, zero_point=zp)
        dq = labneura.dequantize_int16(q, scale, zero_point=zp)
        # Allow one scale unit of error
        assert np.isclose(dq, v, atol=scale), f"v={v}, q={q}, dq={dq}"
