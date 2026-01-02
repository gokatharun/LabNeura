import labneura
import numpy as np

def test_tensor_creation_fp32():
    t = labneura.Tensor([1.0, 2.0, 3.0])
    assert t.size() == 3

def test_tensor_creation_int8():
    t = labneura.Tensor([1, 2, 3], mode=labneura.QuantizationMode.INT8)
    assert t.size() == 3