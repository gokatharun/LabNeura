#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "eda_bindings.h"
#include "labneura/types.hpp"
#include "labneura/tensor.h"
#include "labneura/detail/broadcast.h"
#include "labneura/runtime.h"
#include "labneura/backends/common/backend_factory.h"

#include <limits>
#include <vector>

namespace py = pybind11;

namespace {

std::vector<std::size_t> parse_tensor_shape(py::handle shape) {
    if (!PySequence_Check(shape.ptr()) || PyUnicode_Check(shape.ptr()) ||
        PyBytes_Check(shape.ptr())) {
        throw py::type_error("shape must be a sequence of non-negative integers");
    }

    py::object sequence = py::reinterpret_steal<py::object>(
        PySequence_Fast(shape.ptr(), "shape must be a sequence"));
    if (!sequence) {
        throw py::error_already_set();
    }

    const py::int_ zero(0);
    std::vector<std::size_t> dimensions;
    const auto count = PySequence_Fast_GET_SIZE(sequence.ptr());
    dimensions.reserve(static_cast<std::size_t>(count));
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(sequence.ptr(), i);
        if (PyBool_Check(item) || !PyIndex_Check(item)) {
            throw py::type_error(
                "shape dimensions must be non-negative integers");
        }

        py::object index = py::reinterpret_steal<py::object>(
            PyNumber_Index(item));
        if (!index) {
            throw py::error_already_set();
        }
        const int is_negative =
            PyObject_RichCompareBool(index.ptr(), zero.ptr(), Py_LT);
        if (is_negative < 0) {
            throw py::error_already_set();
        }
        if (is_negative) {
            throw py::value_error("shape dimensions cannot be negative");
        }

        const unsigned long long dimension =
            PyLong_AsUnsignedLongLong(index.ptr());
        if (PyErr_Occurred()) {
            throw py::error_already_set();
        }
        if (dimension > std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("shape dimension exceeds size_t");
        }
        dimensions.push_back(static_cast<std::size_t>(dimension));
    }
    return dimensions;
}

std::size_t parse_tensor_index(py::handle value, const char* argument_name) {
    if (PyBool_Check(value.ptr()) || !PyIndex_Check(value.ptr())) {
        throw py::type_error(std::string(argument_name) +
                             " must be a non-negative integer");
    }

    py::object index = py::reinterpret_steal<py::object>(
        PyNumber_Index(value.ptr()));
    if (!index) {
        throw py::error_already_set();
    }
    const py::int_ zero(0);
    const int is_negative =
        PyObject_RichCompareBool(index.ptr(), zero.ptr(), Py_LT);
    if (is_negative < 0) {
        throw py::error_already_set();
    }
    if (is_negative) {
        throw py::value_error(std::string(argument_name) +
                              " cannot be negative");
    }

    const unsigned long long parsed =
        PyLong_AsUnsignedLongLong(index.ptr());
    if (PyErr_Occurred()) {
        throw py::error_already_set();
    }
    if (parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(std::string(argument_name) +
                                  " exceeds size_t");
    }
    return static_cast<std::size_t>(parsed);
}

labneura::QuantizationMode parse_tensor_mode(py::handle mode) {
    if (py::isinstance<py::str>(mode)) {
        const std::string mode_name = py::cast<std::string>(mode);
        if (mode_name == "auto") {
            return labneura::QuantizationMode::FP32;
        }
        throw py::value_error(
            "mode must be 'auto' or a labneura.QuantizationMode value");
    }
    return py::cast<labneura::QuantizationMode>(mode);
}

labneura::Tensor tensor_from_flat(py::sequence data,
                                  py::handle shape,
                                  py::handle mode) {
    const std::vector<std::size_t> dimensions = parse_tensor_shape(shape);
    const auto count = PySequence_Size(data.ptr());
    if (count < 0) {
        throw py::error_already_set();
    }
    const bool integer_data = [&] {
        for (Py_ssize_t i = 0; i < count; ++i) {
            py::object item = py::reinterpret_steal<py::object>(
                PySequence_GetItem(data.ptr(), i));
            if (!item) {
                throw py::error_already_set();
            }
            if (!PyIndex_Check(item.ptr())) return false;
        }
        return true;
    }();
    const auto dtype = parse_tensor_mode(mode);
    if (integer_data) {
        return labneura::Tensor::from_flat(
            py::cast<std::vector<int>>(data), dimensions, dtype);
    }
    return labneura::Tensor::from_flat(
        py::cast<std::vector<float>>(data), dimensions, dtype);
}

}  // namespace

PYBIND11_MODULE(labneura, m) {
    m.doc() = "LabNeura Python bindings with Tensor support";

    // Surface package version at module import time
    m.attr("__version__") = "0.4.1";

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

    m.def("add_into",
          [](const labneura::Tensor& a, const labneura::Tensor& b,
             labneura::Tensor& out) {
              a.add_into(b, out);
          },
          "Write FP32 a + b into the caller-provided output tensor",
          py::arg("a"), py::arg("b"), py::arg("out"));

    m.def("_broadcast_plan",
          [](py::handle lhs_shape, py::handle lhs_strides,
             py::handle rhs_shape, py::handle rhs_strides) {
              const auto plan = labneura::detail::make_broadcast_plan(
                  parse_tensor_shape(lhs_shape),
                  parse_tensor_shape(lhs_strides),
                  parse_tensor_shape(rhs_shape),
                  parse_tensor_shape(rhs_strides));
              return py::make_tuple(py::tuple(py::cast(plan.shape)),
                                    py::tuple(py::cast(plan.lhs_strides)),
                                    py::tuple(py::cast(plan.rhs_strides)));
          },
          "Internal shape/stride planner; does not execute or allocate tensors",
          py::arg("lhs_shape"), py::arg("lhs_strides"),
          py::arg("rhs_shape"), py::arg("rhs_strides"));

    // QuantizationMode enum
    py::enum_<labneura::QuantizationMode>(m, "QuantizationMode")
        .value("FP32", labneura::QuantizationMode::FP32)
        .value("FP16", labneura::QuantizationMode::FP16)
        .value("INT16", labneura::QuantizationMode::INT16)
        .value("INT8", labneura::QuantizationMode::INT8);

    // Tensor class
    py::class_<labneura::Tensor>(m, "Tensor")
        .def(py::init<>(), "Default constructor")
        .def("__copy__", [](const labneura::Tensor& self) {
            return labneura::Tensor(self);
        })
        .def(py::init<const std::vector<float>&, labneura::QuantizationMode>(),
             "Constructor from float vector", py::arg("data"), py::arg("mode") = labneura::QuantizationMode::FP32)
        .def(py::init<const std::vector<int>&, labneura::QuantizationMode>(),
             "Constructor from int vector", py::arg("data"), py::arg("mode") = labneura::QuantizationMode::FP32)
        .def(py::init<const std::vector<std::vector<float>>&, labneura::QuantizationMode>(),
             "Constructor from a 2-D list of float rows", py::arg("rows"), py::arg("mode") = labneura::QuantizationMode::FP32)
        .def(py::init<float, labneura::QuantizationMode>(),
             "Constructor from float scalar", py::arg("scalar"), py::arg("mode") = labneura::QuantizationMode::FP32)
        .def(py::init<int, labneura::QuantizationMode>(),
             "Constructor from int scalar", py::arg("scalar"), py::arg("mode") = labneura::QuantizationMode::FP32)
        .def_static("from_flat", &tensor_from_flat,
             "Construct a contiguous tensor from flat row-major data and a shape",
             py::arg("data"), py::arg("shape"), py::arg("mode") = py::str("auto"))
        
        // Methods
        .def("size", &labneura::Tensor::size, "Get tensor size")
        .def_property_readonly("shape", [](const labneura::Tensor& self) {
            return py::tuple(py::cast(self.shape()));
        }, "Tensor shape as a tuple")
        .def_property_readonly("strides", [](const labneura::Tensor& self) {
            return py::tuple(py::cast(self.strides()));
        }, "Tensor strides in elements")
        .def_property_readonly("storage_offset",
                               &labneura::Tensor::storage_offset,
                               "Tensor offset in storage elements")
        .def("is_contiguous", &labneura::Tensor::is_contiguous,
             "Whether the tensor uses contiguous row-major storage")
        .def("ndim", &labneura::Tensor::ndim, "Number of dimensions")
        .def("rows", &labneura::Tensor::rows, "Number of rows (1 for 1-D)")
        .def("cols", &labneura::Tensor::cols, "Number of columns")
        .def("numel", &labneura::Tensor::numel, "Get number of elements (alias for size)")
        .def("quantization_mode", &labneura::Tensor::quantization_mode, "Get quantization mode")
        .def("reshape", [](const labneura::Tensor& self, py::handle shape) {
            return self.reshape(parse_tensor_shape(shape));
        }, "Return an independent copy reshaped to the requested dimensions",
           py::arg("shape"))
        .def("reshape_view", [](const labneura::Tensor& self, py::handle shape) {
            return self.reshape_view(parse_tensor_shape(shape));
        }, "Return a contiguous reshape that shares storage with this tensor",
           py::arg("shape"))
        .def("slice_first_axis",
             [](const labneura::Tensor& self, py::handle start,
                py::handle stop) {
                 return self.slice_first_axis(
                     parse_tensor_index(start, "start"),
                     parse_tensor_index(stop, "stop"));
             },
             "Return a first-axis range view that shares storage",
             py::arg("start"), py::arg("stop"))
        .def("select_first_axis",
             [](const labneura::Tensor& self, py::handle index) {
                 return self.select_first_axis(
                     parse_tensor_index(index, "index"));
             },
             "Return a view selecting one element along the first axis",
             py::arg("index"))
        .def("flatten", &labneura::Tensor::flatten,
             "Return an independent one-dimensional copy in row-major order")
        
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
        .def("div", &labneura::Tensor::div, "Divide two tensors (returns new tensor)")
        .def("add_inplace", &labneura::Tensor::add_inplace, "In-place addition (modifies self)")
        .def("mul_inplace", &labneura::Tensor::mul_inplace, "In-place multiplication (modifies self)")
        .def("sub_inplace", &labneura::Tensor::sub_inplace, "In-place subtraction (modifies self)")
        .def("div_inplace", &labneura::Tensor::div_inplace, "In-place division (modifies self)")
        
        // Operator overloading for binary operations
        .def("__add__", &labneura::Tensor::add, "Tensor addition using + operator")
        .def("__sub__", &labneura::Tensor::sub, "Tensor subtraction using - operator")
        .def("__mul__", &labneura::Tensor::mul, "Tensor multiplication using * operator")
        .def("__truediv__", &labneura::Tensor::div, "Tensor division using / operator")
        
        // Operator overloading for in-place operations
        .def("__iadd__", [](labneura::Tensor& self, const labneura::Tensor& other) {
            self.add_inplace(other);
            return self;
        }, "In-place addition using += operator")
        .def("__isub__", [](labneura::Tensor& self, const labneura::Tensor& other) {
            self.sub_inplace(other);
            return self;
        }, "In-place subtraction using -= operator")
        .def("__imul__", [](labneura::Tensor& self, const labneura::Tensor& other) {
            self.mul_inplace(other);
            return self;
        }, "In-place multiplication using *= operator")
        .def("__itruediv__", [](labneura::Tensor& self, const labneura::Tensor& other) {
            self.div_inplace(other);
            return self;
        }, "In-place division using /= operator");

    // Backend detection
    m.def("detect_backend", &labneura::detect_backend, "Detect the preferred backend based on CPU features");
    m.def("available_cpu_count", &labneura::available_cpu_count,
          "Get the CPU count available to this process");
    m.def("execution_thread_count", &labneura::execution_thread_count,
          "Get LabNeura's configured execution thread count");
    m.def("set_execution_thread_count", &labneura::set_execution_thread_count,
          "Set LabNeura's configured execution thread count",
          py::arg("count"));
    m.def("set_execution_thread_count_auto",
          &labneura::set_execution_thread_count_auto,
          "Set LabNeura's configured execution thread count to all available CPUs");

    bind_eda(m);
}
