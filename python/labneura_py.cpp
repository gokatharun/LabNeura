#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "labneura/types.hpp"
#include "labneura/tensor.h"
#include "labneura/backends/backend_factory.h"

namespace py = pybind11;

PYBIND11_MODULE(labneura, m) {
    m.doc() = "LabNeura Python bindings with Tensor support";

    // Surface package version at module import time
    m.attr("__version__") = "0.1.9";

    // Quantization functions
    m.def("quantize_int8", &labneura::quantize_int8,
          "Quantize a float to int8", py::arg("x"), py::arg("scale"), py::arg("zero_point") = 0);

    m.def("dequantize_int8", &labneura::dequantize_int8,
          "Dequantize an int8 to float", py::arg("q"), py::arg("scale"), py::arg("zero_point") = 0);

        m.def("quantize_int16", &labneura::quantize_int16,
            "Quantize a float to int16", py::arg("x"), py::arg("scale"), py::arg("zero_point") = 0);

        m.def("dequantize_int16", &labneura::dequantize_int16,
            "Dequantize an int16 to float", py::arg("q"), py::arg("scale"), py::arg("zero_point") = 0);

    m.def("quantize_fp16", 
          [](float x) { return labneura::quantize_fp16(x); },
          "Quantize a float32 to float16 (returns uint16_t as Python int)");

    m.def("dequantize_fp16",
          [](uint16_t h) { return labneura::dequantize_fp16(h); },
          "Dequantize a float16 to float32 (takes uint16_t as Python int)");

    // QuantizationMode enum
    py::enum_<labneura::QuantizationMode>(m, "QuantizationMode")
        .value("FP32", labneura::QuantizationMode::FP32)
        .value("FP16", labneura::QuantizationMode::FP16)
        .value("INT16", labneura::QuantizationMode::INT16)
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

        .def("data_int16", [](labneura::Tensor& self) {
            // Return as Python list for INT16 mode
            if (self.quantization_mode() != labneura::QuantizationMode::INT16) {
                throw std::runtime_error("Tensor is not in INT16 mode");
            }
            const int16_t* ptr = self.data_int16();
            std::vector<int> result(self.size());
            for (std::size_t i = 0; i < self.size(); ++i) {
                result[i] = static_cast<int>(ptr[i]);
            }
            return result;
        }, "Get INT16 data as list")

        .def("data_fp16", [](labneura::Tensor& self) {
            // Return as Python list of uint16 values for FP16 mode
            if (self.quantization_mode() != labneura::QuantizationMode::FP16) {
                throw std::runtime_error("Tensor is not in FP16 mode");
            }
            // Accept either int16_t* or uint16_t* depending on header signature
            const auto* ptr = self.data_fp16();
            std::vector<int> result(self.size());
            for (std::size_t i = 0; i < self.size(); ++i) {
                // Normalize to unsigned 16-bit for Python-facing API
                result[i] = static_cast<int>(static_cast<uint16_t>(ptr[i]));
            }
            return result;
        }, "Get FP16 data as list of uint16 values")
        
        // Operations
        .def("add", &labneura::Tensor::add, "Add two tensors (returns new tensor)")
        .def("mul", &labneura::Tensor::mul, "Multiply two tensors (returns new tensor)")
        .def("sub", &labneura::Tensor::sub, "Subtract two tensors (returns new tensor)")
        .def("add_inplace", &labneura::Tensor::add_inplace, "In-place addition (modifies self)")
        .def("mul_inplace", &labneura::Tensor::mul_inplace, "In-place multiplication (modifies self)")
        .def("sub_inplace", &labneura::Tensor::sub_inplace, "In-place subtraction (modifies self)");

    // Backend detection
    m.def("detect_backend", &labneura::detect_backend, "Detect the preferred backend based on CPU features");
}

