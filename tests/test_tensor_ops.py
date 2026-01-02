import labneura
import numpy as np

def test_add_fp32():
    a = labneura.Tensor([1.0, 2.0, 3.0])
    b = labneura.Tensor([4.0, 5.0, 6.0])
    c = a.add(b)
    assert np.allclose(c.data_fp32(), [5.0, 7.0, 9.0])

def test_mul_fp32():
    a = labneura.Tensor([1.0, 2.0, 3.0])
    b = labneura.Tensor([2.0, 3.0, 4.0])
    a.mul_inplace(b)
    assert np.allclose(a.data_fp32(), [2.0, 6.0, 12.0])