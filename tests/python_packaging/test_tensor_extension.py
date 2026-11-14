"""Regression checks for the independently built labneura._tensor extension.

Set LABNEURA_TENSOR_EXTENSION to a built _tensor shared library to run these
checks. See tests/python_packaging/README.md for the focused command.
"""

import os
import shutil
import subprocess
import sys
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import pytest


PACKAGE_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "packaging"
    / "core"
    / "src"
    / "labneura"
)
TENSOR_EXTENSION_ENV = "LABNEURA_TENSOR_EXTENSION"
PUBLIC_TENSOR_EXPORTS = (
    "Tensor",
    "QuantizationMode",
    "quantize_int8",
    "dequantize_int8",
    "quantize_int16",
    "dequantize_int16",
    "quantize_fp16",
    "dequantize_fp16",
    "add_into",
    "detect_backend",
    "available_cpu_count",
    "execution_thread_count",
    "set_execution_thread_count",
    "set_execution_thread_count_auto",
)


@pytest.fixture
def tensor_package(tmp_path):
    extension_value = os.environ.get(TENSOR_EXTENSION_ENV)
    if not extension_value:
        pytest.skip(
            f"set {TENSOR_EXTENSION_ENV} to a built _tensor extension; "
            "see tests/python_packaging/README.md"
        )

    extension = Path(extension_value).expanduser().resolve()
    if not extension.is_file():
        pytest.fail(
            f"{TENSOR_EXTENSION_ENV} does not name a file: {extension}",
            pytrace=False,
        )
    if not any(extension.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        pytest.fail(
            f"{extension} does not use a Python extension suffix for {sys.executable}",
            pytrace=False,
        )

    package = tmp_path / "labneura"
    package.mkdir()
    shutil.copy2(PACKAGE_SOURCE / "__init__.py", package / "__init__.py")
    copied_extension = package / extension.name
    shutil.copy2(extension, copied_extension)
    return package.parent, copied_extension.resolve()


def run_python(root, code):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def assert_python_succeeds(root, code):
    result = run_python(root, code)
    assert result.returncode == 0, result.stderr


def native_module_check(extension_path):
    return (
        "from importlib.machinery import EXTENSION_SUFFIXES\n"
        "from pathlib import Path\n"
        "import labneura._tensor as native_tensor\n"
        f"assert Path(native_tensor.__file__).resolve() == Path({str(extension_path)!r})\n"
        "assert any(native_tensor.__file__.endswith(suffix) "
        "for suffix in EXTENSION_SUFFIXES)\n"
    )


def test_package_initializer_loads_native_tensor_exports(tensor_package):
    root, extension = tensor_package
    code = native_module_check(extension) + "import labneura\n"
    code += "\n".join(
        f"assert hasattr(labneura, {symbol!r})" for symbol in PUBLIC_TENSOR_EXPORTS
    )
    code += "\nassert labneura.Tensor is native_tensor.Tensor\n"
    assert_python_succeeds(root, code)


def test_direct_tensor_submodule_import_uses_native_binary(tensor_package):
    root, extension = tensor_package
    code = (
        "from labneura._tensor import Tensor as ExtensionTensor\n"
        "import labneura\n"
        "from pathlib import Path\n"
        "assert labneura.Tensor is ExtensionTensor\n"
        + native_module_check(extension)
    )
    assert_python_succeeds(root, code)


def test_tensor_construction_and_arithmetic(tensor_package):
    root, extension = tensor_package
    code = native_module_check(extension) + (
        "from labneura import Tensor\n"
        "lhs = Tensor([1.0, 2.0])\n"
        "rhs = Tensor([3.0, 4.0])\n"
        "assert (lhs + rhs).data_fp32() == [4.0, 6.0]\n"
    )
    assert_python_succeeds(root, code)


def test_quantization_api_and_result(tensor_package):
    root, extension = tensor_package
    code = native_module_check(extension) + (
        "import labneura\n"
        "assert labneura.quantize_int8(1.0, 0.5) == 2\n"
        "assert labneura.dequantize_int8(2, 0.5) == 1.0\n"
    )
    assert_python_succeeds(root, code)


def test_runtime_and_backend_apis_are_available(tensor_package):
    root, extension = tensor_package
    runtime_exports = (
        "detect_backend",
        "available_cpu_count",
        "execution_thread_count",
        "set_execution_thread_count",
        "set_execution_thread_count_auto",
    )
    code = native_module_check(extension) + "import labneura\n"
    code += "\n".join(
        f"assert callable(labneura.{symbol})" for symbol in runtime_exports
    )
    code += (
        "\nassert isinstance(labneura.detect_backend(), str)"
        "\nassert labneura.available_cpu_count() > 0"
        "\nassert labneura.execution_thread_count() > 0\n"
    )
    assert_python_succeeds(root, code)


def test_eda_is_unavailable_in_tensor_only_package(tensor_package):
    root, extension = tensor_package
    code = native_module_check(extension) + (
        "import labneura\n"
        "assert not hasattr(labneura, 'eda')\n"
        "try:\n"
        "    import labneura.eda\n"
        "except ModuleNotFoundError as error:\n"
        "    assert error.name == 'labneura.eda'\n"
        "else:\n"
        "    raise AssertionError('EDA unexpectedly imported')\n"
    )
    assert_python_succeeds(root, code)
