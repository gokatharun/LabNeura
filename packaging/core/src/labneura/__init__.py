"""LabNeura's component-aware Python package namespace.

Tensor and EDA are optional native components.  This package intentionally
does not require either component so that ``labneura-core`` can be installed
on its own.
"""

from __future__ import annotations

import sys
from importlib import import_module
from importlib.util import find_spec


# This is the core package's version source of truth.  Component distributions
# should use the coordinated release version, but must not derive their
# availability from this optional-import module.
__version__ = "0.4.2"


def _load_optional_module(module_name):
    # Probe for the extension before importing it.  This deliberately avoids
    # treating an initializer failure as an absent optional component, even
    # when that failure reports the component's own fully qualified name.
    if find_spec(module_name) is None:
        return None
    return import_module(module_name)


_tensor_module = _load_optional_module(f"{__name__}._tensor")
if _tensor_module is not None:
    Tensor = _tensor_module.Tensor
    QuantizationMode = _tensor_module.QuantizationMode

    quantize_int8 = _tensor_module.quantize_int8
    dequantize_int8 = _tensor_module.dequantize_int8
    quantize_int16 = _tensor_module.quantize_int16
    dequantize_int16 = _tensor_module.dequantize_int16
    quantize_fp16 = _tensor_module.quantize_fp16
    dequantize_fp16 = _tensor_module.dequantize_fp16
    add_into = _tensor_module.add_into

    detect_backend = _tensor_module.detect_backend
    available_cpu_count = _tensor_module.available_cpu_count
    execution_thread_count = _tensor_module.execution_thread_count
    set_execution_thread_count = _tensor_module.set_execution_thread_count
    set_execution_thread_count_auto = _tensor_module.set_execution_thread_count_auto


_eda_module = _load_optional_module(f"{__name__}._eda")
if _eda_module is not None:
    eda = _eda_module
    # There is no Python source submodule.  Register the extension object so
    # both ``labneura.eda`` and ``import labneura.eda`` resolve identically.
    sys.modules[f"{__name__}.eda"] = eda


del _tensor_module
del _eda_module
del import_module
del find_spec
del _load_optional_module
del sys
