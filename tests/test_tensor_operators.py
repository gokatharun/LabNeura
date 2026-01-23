"""
Test cases for Tensor operator overloading.
Tests +, -, *, /, +=, -=, *=, /= operators
"""

import pytest
import labneura


class TestBinaryOperators:
    """Test binary arithmetic operators (+, -, *, /)"""

    def test_add_operator(self):
        """Test + operator for tensor addition"""
        a = labneura.Tensor([1.0, 2.0, 3.0])
        b = labneura.Tensor([4.0, 5.0, 6.0])
        
        c = a + b
        result = c.data_fp32()
        
        assert len(result) == 3
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(7.0)
        assert result[2] == pytest.approx(9.0)

    def test_sub_operator(self):
        """Test - operator for tensor subtraction"""
        a = labneura.Tensor([10.0, 20.0, 30.0])
        b = labneura.Tensor([1.0, 2.0, 3.0])
        
        c = a - b
        result = c.data_fp32()
        
        assert len(result) == 3
        assert result[0] == pytest.approx(9.0)
        assert result[1] == pytest.approx(18.0)
        assert result[2] == pytest.approx(27.0)

    def test_mul_operator(self):
        """Test * operator for tensor multiplication"""
        a = labneura.Tensor([2.0, 3.0, 4.0])
        b = labneura.Tensor([5.0, 6.0, 7.0])
        
        c = a * b
        result = c.data_fp32()
        
        assert len(result) == 3
        assert result[0] == pytest.approx(10.0)
        assert result[1] == pytest.approx(18.0)
        assert result[2] == pytest.approx(28.0)

    def test_truediv_operator(self):
        """Test / operator for tensor division"""
        a = labneura.Tensor([10.0, 20.0, 30.0])
        b = labneura.Tensor([2.0, 5.0, 10.0])
        
        c = a / b
        result = c.data_fp32()
        
        assert len(result) == 3
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(4.0)
        assert result[2] == pytest.approx(3.0)

    def test_add_operator_single_element(self):
        """Test + operator with single element tensors"""
        a = labneura.Tensor(5.0)
        b = labneura.Tensor(3.0)
        
        c = a + b
        result = c.data_fp32()
        
        assert len(result) == 1
        assert result[0] == pytest.approx(8.0)

    def test_sub_operator_single_element(self):
        """Test - operator with single element tensors"""
        a = labneura.Tensor(10.0)
        b = labneura.Tensor(3.0)
        
        c = a - b
        result = c.data_fp32()
        
        assert len(result) == 1
        assert result[0] == pytest.approx(7.0)

    def test_mul_operator_single_element(self):
        """Test * operator with single element tensors"""
        a = labneura.Tensor(6.0)
        b = labneura.Tensor(7.0)
        
        c = a * b
        result = c.data_fp32()
        
        assert len(result) == 1
        assert result[0] == pytest.approx(42.0)

    def test_truediv_operator_single_element(self):
        """Test / operator with single element tensors"""
        a = labneura.Tensor(20.0)
        b = labneura.Tensor(4.0)
        
        c = a / b
        result = c.data_fp32()
        
        assert len(result) == 1
        assert result[0] == pytest.approx(5.0)

    def test_add_with_negative_values(self):
        """Test + operator with negative values"""
        a = labneura.Tensor([-1.0, -2.0, -3.0])
        b = labneura.Tensor([1.0, 2.0, 3.0])
        
        c = a + b
        result = c.data_fp32()
        
        assert len(result) == 3
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.0)

    def test_mul_with_negative_values(self):
        """Test * operator with negative values"""
        a = labneura.Tensor([-2.0, -3.0, -4.0])
        b = labneura.Tensor([2.0, 3.0, 4.0])
        
        c = a * b
        result = c.data_fp32()
        
        assert len(result) == 3
        assert result[0] == pytest.approx(-4.0)
        assert result[1] == pytest.approx(-9.0)
        assert result[2] == pytest.approx(-16.0)


class TestInPlaceOperators:
    """Test in-place arithmetic operators (+=, -=, *=, /=)"""

    def test_iadd_operator(self):
        """Test += operator for in-place addition"""
        a = labneura.Tensor([1.0, 2.0, 3.0])
        b = labneura.Tensor([4.0, 5.0, 6.0])
        
        a += b
        result = a.data_fp32()
        
        assert len(result) == 3
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(7.0)
        assert result[2] == pytest.approx(9.0)

    def test_isub_operator(self):
        """Test -= operator for in-place subtraction"""
        a = labneura.Tensor([10.0, 20.0, 30.0])
        b = labneura.Tensor([1.0, 2.0, 3.0])
        
        a -= b
        result = a.data_fp32()
        
        assert len(result) == 3
        assert result[0] == pytest.approx(9.0)
        assert result[1] == pytest.approx(18.0)
        assert result[2] == pytest.approx(27.0)

    def test_imul_operator(self):
        """Test *= operator for in-place multiplication"""
        a = labneura.Tensor([2.0, 3.0, 4.0])
        b = labneura.Tensor([5.0, 6.0, 7.0])
        
        a *= b
        result = a.data_fp32()
        
        assert len(result) == 3
        assert result[0] == pytest.approx(10.0)
        assert result[1] == pytest.approx(18.0)
        assert result[2] == pytest.approx(28.0)

    def test_itruediv_operator(self):
        """Test /= operator for in-place division"""
        a = labneura.Tensor([10.0, 20.0, 30.0])
        b = labneura.Tensor([2.0, 5.0, 10.0])
        
        a /= b
        result = a.data_fp32()
        
        assert len(result) == 3
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(4.0)
        assert result[2] == pytest.approx(3.0)

    def test_iadd_operator_single_element(self):
        """Test += operator with single element tensors"""
        a = labneura.Tensor(5.0)
        b = labneura.Tensor(3.0)
        
        a += b
        result = a.data_fp32()
        
        assert len(result) == 1
        assert result[0] == pytest.approx(8.0)

    def test_isub_operator_single_element(self):
        """Test -= operator with single element tensors"""
        a = labneura.Tensor(10.0)
        b = labneura.Tensor(3.0)
        
        a -= b
        result = a.data_fp32()
        
        assert len(result) == 1
        assert result[0] == pytest.approx(7.0)

    def test_imul_operator_single_element(self):
        """Test *= operator with single element tensors"""
        a = labneura.Tensor(6.0)
        b = labneura.Tensor(7.0)
        
        a *= b
        result = a.data_fp32()
        
        assert len(result) == 1
        assert result[0] == pytest.approx(42.0)

    def test_itruediv_operator_single_element(self):
        """Test /= operator with single element tensors"""
        a = labneura.Tensor(20.0)
        b = labneura.Tensor(4.0)
        
        a /= b
        result = a.data_fp32()
        
        assert len(result) == 1
        assert result[0] == pytest.approx(5.0)

    def test_iadd_with_negative_values(self):
        """Test += operator with negative values"""
        a = labneura.Tensor([-1.0, -2.0, -3.0])
        b = labneura.Tensor([1.0, 2.0, 3.0])
        
        a += b
        result = a.data_fp32()
        
        assert len(result) == 3
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.0)

    def test_imul_with_negative_values(self):
        """Test *= operator with negative values"""
        a = labneura.Tensor([-2.0, -3.0, -4.0])
        b = labneura.Tensor([2.0, 3.0, 4.0])
        
        a *= b
        result = a.data_fp32()
        
        assert len(result) == 3
        assert result[0] == pytest.approx(-4.0)
        assert result[1] == pytest.approx(-9.0)
        assert result[2] == pytest.approx(-16.0)

    def test_multiple_inplace_operations(self):
        """Test chaining multiple in-place operations"""
        a = labneura.Tensor([10.0, 20.0, 30.0])
        
        # Start with [10, 20, 30]
        a += labneura.Tensor([5.0, 5.0, 5.0])  # [15, 25, 35]
        result = a.data_fp32()
        assert result[0] == pytest.approx(15.0)
        
        a -= labneura.Tensor([5.0, 5.0, 5.0])  # [10, 20, 30]
        result = a.data_fp32()
        assert result[0] == pytest.approx(10.0)
        
        a *= labneura.Tensor([2.0, 2.0, 2.0])  # [20, 40, 60]
        result = a.data_fp32()
        assert result[0] == pytest.approx(20.0)
        
        a /= labneura.Tensor([2.0, 2.0, 2.0])  # [10, 20, 30]
        result = a.data_fp32()
        assert result[0] == pytest.approx(10.0)


class TestOperatorConsistency:
    """Test that operators are consistent with method calls"""

    def test_add_operator_vs_method(self):
        """Test that + operator gives same result as add() method"""
        a = labneura.Tensor([1.0, 2.0, 3.0])
        b = labneura.Tensor([4.0, 5.0, 6.0])
        
        result_op = (a + b).data_fp32()
        result_method = a.add(b).data_fp32()
        
        for i in range(len(result_op)):
            assert result_op[i] == pytest.approx(result_method[i])

    def test_sub_operator_vs_method(self):
        """Test that - operator gives same result as sub() method"""
        a = labneura.Tensor([10.0, 20.0, 30.0])
        b = labneura.Tensor([1.0, 2.0, 3.0])
        
        result_op = (a - b).data_fp32()
        result_method = a.sub(b).data_fp32()
        
        for i in range(len(result_op)):
            assert result_op[i] == pytest.approx(result_method[i])

    def test_mul_operator_vs_method(self):
        """Test that * operator gives same result as mul() method"""
        a = labneura.Tensor([2.0, 3.0, 4.0])
        b = labneura.Tensor([5.0, 6.0, 7.0])
        
        result_op = (a * b).data_fp32()
        result_method = a.mul(b).data_fp32()
        
        for i in range(len(result_op)):
            assert result_op[i] == pytest.approx(result_method[i])

    def test_truediv_operator_vs_method(self):
        """Test that / operator gives same result as div() method"""
        a = labneura.Tensor([10.0, 20.0, 30.0])
        b = labneura.Tensor([2.0, 5.0, 10.0])
        
        result_op = (a / b).data_fp32()
        result_method = a.div(b).data_fp32()
        
        for i in range(len(result_op)):
            assert result_op[i] == pytest.approx(result_method[i])

    def test_iadd_operator_vs_method(self):
        """Test that += operator gives same result as add_inplace() method"""
        a = labneura.Tensor([1.0, 2.0, 3.0])
        b = labneura.Tensor([4.0, 5.0, 6.0])
        
        a_copy = labneura.Tensor([1.0, 2.0, 3.0])
        b_copy = labneura.Tensor([4.0, 5.0, 6.0])
        
        a += b
        result_op = a.data_fp32()
        
        a_copy.add_inplace(b_copy)
        result_method = a_copy.data_fp32()
        
        for i in range(len(result_op)):
            assert result_op[i] == pytest.approx(result_method[i])

    def test_isub_operator_vs_method(self):
        """Test that -= operator gives same result as sub_inplace() method"""
        a = labneura.Tensor([10.0, 20.0, 30.0])
        b = labneura.Tensor([1.0, 2.0, 3.0])
        
        a_copy = labneura.Tensor([10.0, 20.0, 30.0])
        b_copy = labneura.Tensor([1.0, 2.0, 3.0])
        
        a -= b
        result_op = a.data_fp32()
        
        a_copy.sub_inplace(b_copy)
        result_method = a_copy.data_fp32()
        
        for i in range(len(result_op)):
            assert result_op[i] == pytest.approx(result_method[i])

    def test_imul_operator_vs_method(self):
        """Test that *= operator gives same result as mul_inplace() method"""
        a = labneura.Tensor([2.0, 3.0, 4.0])
        b = labneura.Tensor([5.0, 6.0, 7.0])
        
        a_copy = labneura.Tensor([2.0, 3.0, 4.0])
        b_copy = labneura.Tensor([5.0, 6.0, 7.0])
        
        a *= b
        result_op = a.data_fp32()
        
        a_copy.mul_inplace(b_copy)
        result_method = a_copy.data_fp32()
        
        for i in range(len(result_op)):
            assert result_op[i] == pytest.approx(result_method[i])

    def test_itruediv_operator_vs_method(self):
        """Test that /= operator gives same result as div_inplace() method"""
        a = labneura.Tensor([10.0, 20.0, 30.0])
        b = labneura.Tensor([2.0, 5.0, 10.0])
        
        a_copy = labneura.Tensor([10.0, 20.0, 30.0])
        b_copy = labneura.Tensor([2.0, 5.0, 10.0])
        
        a /= b
        result_op = a.data_fp32()
        
        a_copy.div_inplace(b_copy)
        result_method = a_copy.data_fp32()
        
        for i in range(len(result_op)):
            assert result_op[i] == pytest.approx(result_method[i])


class TestOperatorsWithFP16:
    """Test operators with FP16 quantization mode"""

    def test_add_fp16(self):
        """Test + operator with FP16 tensors"""
        a = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP16)
        b = labneura.Tensor([4.0, 5.0, 6.0], labneura.QuantizationMode.FP16)
        
        c = a + b
        assert c.quantization_mode() == labneura.QuantizationMode.FP16
        result = c.data_fp16()
        
        # FP16 has limited precision, use appropriate tolerance
        assert len(result) == 3

    def test_sub_fp16(self):
        """Test - operator with FP16 tensors"""
        a = labneura.Tensor([10.0, 20.0, 30.0], labneura.QuantizationMode.FP16)
        b = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP16)
        
        c = a - b
        assert c.quantization_mode() == labneura.QuantizationMode.FP16
        assert c.size() == 3

    def test_mul_fp16(self):
        """Test * operator with FP16 tensors"""
        a = labneura.Tensor([2.0, 3.0, 4.0], labneura.QuantizationMode.FP16)
        b = labneura.Tensor([5.0, 6.0, 7.0], labneura.QuantizationMode.FP16)
        
        c = a * b
        assert c.quantization_mode() == labneura.QuantizationMode.FP16
        assert c.size() == 3

    def test_div_fp16(self):
        """Test / operator with FP16 tensors"""
        a = labneura.Tensor([10.0, 20.0, 30.0], labneura.QuantizationMode.FP16)
        b = labneura.Tensor([2.0, 5.0, 10.0], labneura.QuantizationMode.FP16)
        
        c = a / b
        assert c.quantization_mode() == labneura.QuantizationMode.FP16
        assert c.size() == 3

    def test_iadd_fp16(self):
        """Test += operator with FP16 tensors"""
        a = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP16)
        b = labneura.Tensor([4.0, 5.0, 6.0], labneura.QuantizationMode.FP16)
        
        a += b
        assert a.quantization_mode() == labneura.QuantizationMode.FP16
        assert a.size() == 3

    def test_isub_fp16(self):
        """Test -= operator with FP16 tensors"""
        a = labneura.Tensor([10.0, 20.0, 30.0], labneura.QuantizationMode.FP16)
        b = labneura.Tensor([1.0, 2.0, 3.0], labneura.QuantizationMode.FP16)
        
        a -= b
        assert a.quantization_mode() == labneura.QuantizationMode.FP16
        assert a.size() == 3

    def test_imul_fp16(self):
        """Test *= operator with FP16 tensors"""
        a = labneura.Tensor([2.0, 3.0, 4.0], labneura.QuantizationMode.FP16)
        b = labneura.Tensor([5.0, 6.0, 7.0], labneura.QuantizationMode.FP16)
        
        a *= b
        assert a.quantization_mode() == labneura.QuantizationMode.FP16
        assert a.size() == 3

    def test_itruediv_fp16(self):
        """Test /= operator with FP16 tensors"""
        a = labneura.Tensor([10.0, 20.0, 30.0], labneura.QuantizationMode.FP16)
        b = labneura.Tensor([2.0, 5.0, 10.0], labneura.QuantizationMode.FP16)
        
        a /= b
        assert a.quantization_mode() == labneura.QuantizationMode.FP16
        assert a.size() == 3


class TestOperatorsWithINT16:
    """Test operators with INT16 quantization mode"""

    def test_add_int16(self):
        """Test + operator with INT16 tensors"""
        a = labneura.Tensor([100, 200, 300], labneura.QuantizationMode.INT16)
        b = labneura.Tensor([400, 500, 600], labneura.QuantizationMode.INT16)
        
        c = a + b
        assert c.quantization_mode() == labneura.QuantizationMode.INT16
        result = c.data_int16()
        
        assert len(result) == 3
        assert result[0] == 500
        assert result[1] == 700
        assert result[2] == 900

    def test_sub_int16(self):
        """Test - operator with INT16 tensors"""
        a = labneura.Tensor([1000, 2000, 3000], labneura.QuantizationMode.INT16)
        b = labneura.Tensor([100, 200, 300], labneura.QuantizationMode.INT16)
        
        c = a - b
        assert c.quantization_mode() == labneura.QuantizationMode.INT16
        result = c.data_int16()
        
        assert result[0] == 900
        assert result[1] == 1800
        assert result[2] == 2700

    def test_mul_int16(self):
        """Test * operator with INT16 tensors"""
        a = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT16)
        b = labneura.Tensor([2, 3, 4], labneura.QuantizationMode.INT16)
        
        c = a * b
        assert c.quantization_mode() == labneura.QuantizationMode.INT16
        result = c.data_int16()
        
        assert result[0] == 20
        assert result[1] == 60
        assert result[2] == 120

    def test_div_int16(self):
        """Test / operator with INT16 tensors"""
        a = labneura.Tensor([100, 200, 300], labneura.QuantizationMode.INT16)
        b = labneura.Tensor([2, 4, 5], labneura.QuantizationMode.INT16)
        
        c = a / b
        assert c.quantization_mode() == labneura.QuantizationMode.INT16
        result = c.data_int16()
        
        assert result[0] == 50
        assert result[1] == 50
        assert result[2] == 60

    def test_iadd_int16(self):
        """Test += operator with INT16 tensors"""
        a = labneura.Tensor([100, 200, 300], labneura.QuantizationMode.INT16)
        b = labneura.Tensor([400, 500, 600], labneura.QuantizationMode.INT16)
        
        a += b
        assert a.quantization_mode() == labneura.QuantizationMode.INT16
        result = a.data_int16()
        
        assert result[0] == 500
        assert result[1] == 700
        assert result[2] == 900

    def test_isub_int16(self):
        """Test -= operator with INT16 tensors"""
        a = labneura.Tensor([1000, 2000, 3000], labneura.QuantizationMode.INT16)
        b = labneura.Tensor([100, 200, 300], labneura.QuantizationMode.INT16)
        
        a -= b
        assert a.quantization_mode() == labneura.QuantizationMode.INT16
        result = a.data_int16()
        
        assert result[0] == 900
        assert result[1] == 1800
        assert result[2] == 2700

    def test_imul_int16(self):
        """Test *= operator with INT16 tensors"""
        a = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT16)
        b = labneura.Tensor([2, 3, 4], labneura.QuantizationMode.INT16)
        
        a *= b
        assert a.quantization_mode() == labneura.QuantizationMode.INT16
        result = a.data_int16()
        
        assert result[0] == 20
        assert result[1] == 60
        assert result[2] == 120

    def test_itruediv_int16(self):
        """Test /= operator with INT16 tensors"""
        a = labneura.Tensor([100, 200, 300], labneura.QuantizationMode.INT16)
        b = labneura.Tensor([2, 4, 5], labneura.QuantizationMode.INT16)
        
        a /= b
        assert a.quantization_mode() == labneura.QuantizationMode.INT16
        result = a.data_int16()
        
        assert result[0] == 50
        assert result[1] == 50
        assert result[2] == 60

    def test_negative_values_int16(self):
        """Test operators with negative INT16 values"""
        a = labneura.Tensor([-100, -200, -300], labneura.QuantizationMode.INT16)
        b = labneura.Tensor([100, 200, 300], labneura.QuantizationMode.INT16)
        
        c = a + b
        result = c.data_int16()
        assert result[0] == 0
        assert result[1] == 0
        assert result[2] == 0


class TestOperatorsWithINT8:
    """Test operators with INT8 quantization mode"""

    def test_add_int8(self):
        """Test + operator with INT8 tensors"""
        a = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([40, 50, 60], labneura.QuantizationMode.INT8)
        
        c = a + b
        assert c.quantization_mode() == labneura.QuantizationMode.INT8
        result = c.data_int8()
        
        assert len(result) == 3
        assert result[0] == 50
        assert result[1] == 70
        assert result[2] == 90

    def test_sub_int8(self):
        """Test - operator with INT8 tensors"""
        a = labneura.Tensor([100, 80, 60], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT8)
        
        c = a - b
        assert c.quantization_mode() == labneura.QuantizationMode.INT8
        result = c.data_int8()
        
        assert result[0] == 90
        assert result[1] == 60
        assert result[2] == 30

    def test_mul_int8(self):
        """Test * operator with INT8 tensors"""
        a = labneura.Tensor([5, 10, 15], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([2, 3, 4], labneura.QuantizationMode.INT8)
        
        c = a * b
        assert c.quantization_mode() == labneura.QuantizationMode.INT8
        result = c.data_int8()
        
        assert result[0] == 10
        assert result[1] == 30
        assert result[2] == 60

    def test_div_int8(self):
        """Test / operator with INT8 tensors"""
        a = labneura.Tensor([100, 80, 60], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([2, 4, 5], labneura.QuantizationMode.INT8)
        
        c = a / b
        assert c.quantization_mode() == labneura.QuantizationMode.INT8
        result = c.data_int8()
        
        assert result[0] == 50
        assert result[1] == 20
        assert result[2] == 12

    def test_iadd_int8(self):
        """Test += operator with INT8 tensors"""
        a = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([40, 50, 60], labneura.QuantizationMode.INT8)
        
        a += b
        assert a.quantization_mode() == labneura.QuantizationMode.INT8
        result = a.data_int8()
        
        assert result[0] == 50
        assert result[1] == 70
        assert result[2] == 90

    def test_isub_int8(self):
        """Test -= operator with INT8 tensors"""
        a = labneura.Tensor([100, 80, 60], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT8)
        
        a -= b
        assert a.quantization_mode() == labneura.QuantizationMode.INT8
        result = a.data_int8()
        
        assert result[0] == 90
        assert result[1] == 60
        assert result[2] == 30

    def test_imul_int8(self):
        """Test *= operator with INT8 tensors"""
        a = labneura.Tensor([5, 10, 15], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([2, 3, 4], labneura.QuantizationMode.INT8)
        
        a *= b
        assert a.quantization_mode() == labneura.QuantizationMode.INT8
        result = a.data_int8()
        
        assert result[0] == 10
        assert result[1] == 30
        assert result[2] == 60

    def test_itruediv_int8(self):
        """Test /= operator with INT8 tensors"""
        a = labneura.Tensor([100, 80, 60], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([2, 4, 5], labneura.QuantizationMode.INT8)
        
        a /= b
        assert a.quantization_mode() == labneura.QuantizationMode.INT8
        result = a.data_int8()
        
        assert result[0] == 50
        assert result[1] == 20
        assert result[2] == 12

    def test_negative_values_int8(self):
        """Test operators with negative INT8 values"""
        a = labneura.Tensor([-10, -20, -30], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([10, 20, 30], labneura.QuantizationMode.INT8)
        
        c = a + b
        result = c.data_int8()
        assert result[0] == 0
        assert result[1] == 0
        assert result[2] == 0

    def test_saturation_int8(self):
        """Test INT8 saturation behavior"""
        a = labneura.Tensor([100, 100], labneura.QuantizationMode.INT8)
        b = labneura.Tensor([100, 100], labneura.QuantizationMode.INT8)
        
        c = a + b
        # Result should handle overflow (saturate or wrap depending on implementation)
        assert c.quantization_mode() == labneura.QuantizationMode.INT8
        assert c.size() == 2


class TestOperatorEdgeCases:
    """Test edge cases for operators"""

    def test_add_with_zeros(self):
        """Test addition with zero values"""
        a = labneura.Tensor([0.0, 0.0, 0.0])
        b = labneura.Tensor([1.0, 2.0, 3.0])
        
        c = a + b
        result = c.data_fp32()
        
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(3.0)

    def test_mul_with_zeros(self):
        """Test multiplication with zero values"""
        a = labneura.Tensor([0.0, 0.0, 0.0])
        b = labneura.Tensor([1.0, 2.0, 3.0])
        
        c = a * b
        result = c.data_fp32()
        
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.0)

    def test_mul_with_ones(self):
        """Test multiplication with ones"""
        a = labneura.Tensor([1.0, 1.0, 1.0])
        b = labneura.Tensor([5.0, 10.0, 15.0])
        
        c = a * b
        result = c.data_fp32()
        
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(10.0)
        assert result[2] == pytest.approx(15.0)

    def test_add_large_values(self):
        """Test addition with large values"""
        a = labneura.Tensor([1000.0, 2000.0, 3000.0])
        b = labneura.Tensor([4000.0, 5000.0, 6000.0])
        
        c = a + b
        result = c.data_fp32()
        
        assert result[0] == pytest.approx(5000.0)
        assert result[1] == pytest.approx(7000.0)
        assert result[2] == pytest.approx(9000.0)

    def test_add_small_values(self):
        """Test addition with small values"""
        a = labneura.Tensor([0.001, 0.002, 0.003])
        b = labneura.Tensor([0.004, 0.005, 0.006])
        
        c = a + b
        result = c.data_fp32()
        
        assert result[0] == pytest.approx(0.005)
        assert result[1] == pytest.approx(0.007)
        assert result[2] == pytest.approx(0.009)

    def test_div_by_one(self):
        """Test division by one"""
        a = labneura.Tensor([5.0, 10.0, 15.0])
        b = labneura.Tensor([1.0, 1.0, 1.0])
        
        c = a / b
        result = c.data_fp32()
        
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(10.0)
        assert result[2] == pytest.approx(15.0)

    def test_sub_equal_tensors(self):
        """Test subtraction of equal tensors"""
        a = labneura.Tensor([5.0, 10.0, 15.0])
        b = labneura.Tensor([5.0, 10.0, 15.0])
        
        c = a - b
        result = c.data_fp32()
        
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.0)

    def test_div_equal_tensors(self):
        """Test division of equal tensors"""
        a = labneura.Tensor([5.0, 10.0, 15.0])
        b = labneura.Tensor([5.0, 10.0, 15.0])
        
        c = a / b
        result = c.data_fp32()
        
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(1.0)
