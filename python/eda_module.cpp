#include <pybind11/pybind11.h>

#include "eda_bindings.h"

PYBIND11_MODULE(_eda, module) {
    module.doc() = "Structural Verilog parsing and netlist interchange.";
    bind_eda_api(module);
}
