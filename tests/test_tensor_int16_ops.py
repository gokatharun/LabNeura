import pytest
import labneura


def make_int16(values):
    return labneura.Tensor(values, labneura.QuantizationMode.INT16)


def test_int16_add_basic():
    a = make_int16([1000, -2000, 150])
    b = make_int16([500, 1000, -50])
    a.add_inplace(b)
    assert a.data_int16() == [1500, -1000, 100]


def test_int16_add_saturation_positive():
    # Push into positive saturation: 32000 + 5000 = 37000 -> 32767
    a = make_int16([32000, 30000])
    b = make_int16([5000, 2000])
    a.add_inplace(b)
    assert a.data_int16() == [32767, 32000]


def test_int16_add_saturation_negative():
    # Push into negative saturation: -32000 + (-5000) = -37000 -> -32768
    a = make_int16([-32000, -30000])
    b = make_int16([-5000, -1000])
    a.add_inplace(b)
    assert a.data_int16() == [-32768, -31000]


def test_int16_sub_basic():
    a = make_int16([3000, -1000, 500])
    b = make_int16([1000, -500, 200])
    a.sub_inplace(b)
    assert a.data_int16() == [2000, -500, 300]


def test_int16_sub_saturation_positive():
    # 32000 - (-5000) = 37000 -> 32767
    a = make_int16([32000, 30000])
    b = make_int16([-5000, -1000])
    a.sub_inplace(b)
    assert a.data_int16() == [32767, 31000]


def test_int16_sub_saturation_negative():
    # -32000 - 5000 = -37000 -> -32768
    a = make_int16([-32000, -30000])
    b = make_int16([5000, 1000])
    a.sub_inplace(b)
    assert a.data_int16() == [-32768, -31000]


def test_int16_mul_basic():
    a = make_int16([100, -50, 200])
    b = make_int16([3, -2, 4])
    a.mul_inplace(b)
    assert a.data_int16() == [300, 100, 800]


def test_int16_mul_saturation_positive():
    # 200 * 200 = 40000 -> 32767
    a = make_int16([200, 300, 400])
    b = make_int16([200, 150, 100])
    a.mul_inplace(b)
    assert a.data_int16() == [32767, 32767, 32767]


def test_int16_mul_saturation_negative():
    # -400 * 100 = -40000 -> -32768
    a = make_int16([-400, 300, -500])
    b = make_int16([100, -200, 100])
    a.mul_inplace(b)
    assert a.data_int16() == [-32768, -32768, -32768]


def test_int16_hierarchical_tail_len70():
    # Test 64 + 4 + 2 scalar tail = 70 elements
    values_a = list(range(70))
    values_b = [1] * 70
    a = make_int16(values_a)
    b = make_int16(values_b)
    a.add_inplace(b)
    expected = [i + 1 for i in range(70)]
    assert a.data_int16() == expected


def test_int16_hierarchical_tail_len35():
    # Test 32 + 3 scalar = 35 elements
    values_a = [100] * 35
    values_b = [10] * 35
    a = make_int16(values_a)
    b = make_int16(values_b)
    a.mul_inplace(b)
    expected = [1000] * 35
    assert a.data_int16() == expected


def test_int16_hierarchical_tail_len18():
    # Test 16 + 2 scalar = 18 elements
    values_a = [500] * 18
    values_b = [250] * 18
    a = make_int16(values_a)
    b = make_int16(values_b)
    a.sub_inplace(b)
    expected = [250] * 18
    assert a.data_int16() == expected


def test_int16_hierarchical_tail_len9():
    # Test 8 + 1 scalar = 9 elements
    values_a = [-1000] * 9
    values_b = [500] * 9
    a = make_int16(values_a)
    b = make_int16(values_b)
    a.add_inplace(b)
    expected = [-500] * 9
    assert a.data_int16() == expected


def test_int16_hierarchical_tail_len5():
    # Test 4 + 1 scalar = 5 elements
    values_a = [2000, 3000, 4000, 5000, 6000]
    values_b = [1000, 1000, 1000, 1000, 1000]
    a = make_int16(values_a)
    b = make_int16(values_b)
    a.sub_inplace(b)
    assert a.data_int16() == [1000, 2000, 3000, 4000, 5000]


def test_int16_mode_mismatch_raises():
    a = make_int16([1, 2, 3])
    b = labneura.Tensor([1, 2, 3], labneura.QuantizationMode.INT8)
    with pytest.raises(RuntimeError):
        a.add_inplace(b)


def test_int16_edge_case_single_element():
    a = make_int16([100])
    b = make_int16([50])
    a.add_inplace(b)
    assert a.data_int16() == [150]


def test_int16_edge_case_zero_operands():
    a = make_int16([100, -200, 300])
    b = make_int16([0, 0, 0])
    a.add_inplace(b)
    assert a.data_int16() == [100, -200, 300]
    a.mul_inplace(b)
    assert a.data_int16() == [0, 0, 0]


def test_int16_large_arrays():
    # Test 1000 elements (15*64 + 40 tail)
    size = 1000
    a = make_int16([10] * size)
    b = make_int16([5] * size)
    a.add_inplace(b)
    assert a.data_int16() == [15] * size

