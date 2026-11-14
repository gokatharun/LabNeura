#include <pybind11/pybind11.h>

#include "eda_bindings.h"
#include "tensor_bindings.h"

namespace py = pybind11;

PYBIND11_MODULE(labneura, m) {
    m.doc() = "LabNeura Python bindings with Tensor support";
    m.attr("__version__") = "0.4.5";

    bind_tensor(m);
    bind_eda(m);
}
