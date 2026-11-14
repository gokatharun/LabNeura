#include <pybind11/pybind11.h>

#include "tensor_bindings.h"

PYBIND11_MODULE(_tensor, m) {
    bind_tensor(m);
}
