#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/labneura/types.hpp"
#include "../include/labneura/tensor.h"
#include "../include/labneura/backends/backend_factory.h"

namespace py = pybind11;

PYBIND11_MODULE(labneura, m) {
    m.doc() = "LabNeura Python bindings with Tensor support";

    // Quantization functions
    m.def("quantize_int8", &labneura::quantize_int8,
          "Quantize a float to int8", py::arg("x"), py::arg("scale"), py::arg("zero_point") = 0);

    m.def("dequantize_int8", &labneura::dequantize_int8,
          "Dequantize an int8 to float", py::arg("q"), py::arg("scale"), py::arg("zero_point") = 0);

    // QuantizationMode enum
    py::enum_<labneura::QuantizationMode>(m, "QuantizationMode")
        .value("FP32", labneura::QuantizationMode::FP32)
        .value("INT8", labneura::QuantizationMode::INT8);

    // Tensor class
    py::class_<labneura::Tensor>(m, "Tensor")
        .def(py::init<>(), "Default constructor")
        .def(py::init<const std::vector<float>&, labneura::QuantizationMode>(),
             "Constructor from float vector", py::arg("data"), py::arg("mode") = labneura::QuantizationMode::FP32)
        .def(py::init<const std::vector<int>&, labneura::QuantizationMode>(),
             "Constructor from int vector", py::arg("data"), py::arg("mode") = labneura::QuantizationMode::FP32)
        .def(py::init<float, labneura::QuantizationMode>(),
             "Constructor from float scalar", py::arg("scalar"), py::arg("mode") = labneura::QuantizationMode::FP32)
        .def(py::init<int, labneura::QuantizationMode>(),
             "Constructor from int scalar", py::arg("scalar"), py::arg("mode") = labneura::QuantizationMode::FP32)
        
        // Methods
        .def("size", &labneura::Tensor::size, "Get tensor size")
        .def("numel", &labneura::Tensor::numel, "Get number of elements (alias for size)")
        .def("quantization_mode", &labneura::Tensor::quantization_mode, "Get quantization mode")
        
        // Data access
        .def("data_fp32", [](labneura::Tensor& self) {
            // Return as Python list for FP32 mode
            if (self.quantization_mode() != labneura::QuantizationMode::FP32) {
                throw std::runtime_error("Tensor is not in FP32 mode");
            }
            std::vector<float> result(self.data_fp32(), self.data_fp32() + self.size());
            return result;
        }, "Get FP32 data as list")
        
        .def("data_int8", [](labneura::Tensor& self) {
            // Return as Python list for INT8 mode
            if (self.quantization_mode() != labneura::QuantizationMode::INT8) {
                throw std::runtime_error("Tensor is not in INT8 mode");
            }
            std::vector<int> result(self.data_int8(), self.data_int8() + self.size());
            return result;
        }, "Get INT8 data as list")
        
        // Operations
        .def("add", &labneura::Tensor::add, "Add two tensors (returns new tensor)")
        .def("add_inplace", &labneura::Tensor::add_inplace, "In-place addition (modifies self)")
        .def("sub_inplace", &labneura::Tensor::sub_inplace, "In-place subtraction (modifies self)")
        .def("mul_inplace", &labneura::Tensor::mul_inplace, "In-place multiplication (modifies self)");

    // Backend detection
    m.def("detect_backend", &labneura::detect_backend, "Detect the preferred backend based on CPU features");
}

