#pragma once

#include <pybind11/pybind11.h>

void bind_eda_api(pybind11::module_& eda_module);
void bind_eda(pybind11::module_& labneura_module);
