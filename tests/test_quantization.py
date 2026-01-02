import labneura
import numpy as np

def test_int8_saturation():
    a = labneura.Tensor([120, 120], mode=labneura.QuantizationMode.INT8)
    b = labneura.Tensor([20, 20], mode=labneura.QuantizationMode.INT8)
    c = a.add(b)
    assert c.data_int8() == [127, 127]