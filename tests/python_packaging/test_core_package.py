import os
import subprocess
import sys
from pathlib import Path

import pytest


PACKAGE_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "packaging"
    / "core"
    / "src"
    / "labneura"
)

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

TENSOR_FIXTURE = """
class Tensor:
    pass


class QuantizationMode:
    pass


def quantize_int8(*args):
    pass


def dequantize_int8(*args):
    pass


def quantize_int16(*args):
    pass


def dequantize_int16(*args):
    pass


def quantize_fp16(*args):
    pass


def dequantize_fp16(*args):
    pass


def add_into(*args):
    pass


def detect_backend(*args):
    pass


def available_cpu_count(*args):
    pass


def execution_thread_count(*args):
    pass


def set_execution_thread_count(*args):
    pass


def set_execution_thread_count_auto(*args):
    pass


_broadcast_plan = object()
private_helper = object()
"""

EDA_FIXTURE = """
class Module:
    pass
"""


def make_package(tmp_path, *, tensor=None, eda=None):
    package = tmp_path / "labneura"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        (PACKAGE_SOURCE / "__init__.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if tensor is not None:
        (package / "_tensor.py").write_text(tensor, encoding="utf-8")
    if eda is not None:
        (package / "_eda.py").write_text(eda, encoding="utf-8")
    return tmp_path


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


def test_component_combinations_and_optional_imports(tmp_path):
    cases = {
        "neither": (None, None),
        "tensor_only": (TENSOR_FIXTURE, None),
        "eda_only": (None, EDA_FIXTURE),
        "both": (TENSOR_FIXTURE, EDA_FIXTURE),
    }

    for name, (tensor, eda) in cases.items():
        root = make_package(tmp_path / name, tensor=tensor, eda=eda)
        checks = [
            "import labneura",
            "assert labneura.__version__ == '0.4.3'",
        ]
        if tensor is None:
            checks.append("assert not hasattr(labneura, 'Tensor')")
            checks.append("assert not hasattr(labneura, 'QuantizationMode')")
        else:
            checks.extend(
                f"assert hasattr(labneura, {symbol!r})"
                for symbol in PUBLIC_TENSOR_EXPORTS
            )
            checks.append("assert not hasattr(labneura, '_broadcast_plan')")
            checks.append("assert not hasattr(labneura, '_tensor_module')")

        if eda is None:
            checks.append("assert not hasattr(labneura, 'eda')")
            checks.append(
                "try:\n"
                "    import labneura.eda\n"
                "except ModuleNotFoundError as exc:\n"
                "    assert exc.name == 'labneura.eda'\n"
                "else:\n"
                "    raise AssertionError('EDA unexpectedly available')"
            )
        else:
            checks.extend(
                [
                    "import labneura.eda",
                    "import labneura._eda as eda_module",
                    "assert labneura.eda is eda_module",
                    "assert labneura.eda.Module is eda_module.Module",
                    "import sys",
                    "assert sys.modules['labneura.eda'] is eda_module",
                ]
            )

        result = run_python(root, "\n".join(checks))
        assert result.returncode == 0, f"{name}:\n{result.stderr}"


def test_every_public_tensor_reexport_is_present(tmp_path):
    root = make_package(tmp_path, tensor=TENSOR_FIXTURE)
    code = "import labneura\n" + "\n".join(
        f"assert hasattr(labneura, {symbol!r})" for symbol in PUBLIC_TENSOR_EXPORTS
    )
    result = run_python(root, code)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("component", "source", "missing_name"),
    [
        (
            "_tensor",
            "raise ModuleNotFoundError('dependency missing', name='missing_tensor_dependency')",
            "missing_tensor_dependency",
        ),
        (
            "_eda",
            "raise ModuleNotFoundError('dependency missing', name='missing_eda_dependency')",
            "missing_eda_dependency",
        ),
        (
            "_tensor",
            "raise ModuleNotFoundError('component failure', name='component_tensor_failure')",
            "component_tensor_failure",
        ),
        (
            "_eda",
            "raise ModuleNotFoundError('component failure', name='component_eda_failure')",
            "component_eda_failure",
        ),
        (
            "_tensor",
            "raise ModuleNotFoundError('initializer failure', name='labneura._tensor')",
            "labneura._tensor",
        ),
        (
            "_eda",
            "raise ModuleNotFoundError('initializer failure', name='labneura._eda')",
            "labneura._eda",
        ),
    ],
)
def test_unrelated_module_not_found_errors_propagate(
    tmp_path, component, source, missing_name
):
    root = make_package(tmp_path, **{component[1:]: source})
    result = run_python(
        root,
        "try:\n"
        "    import labneura\n"
        "except ModuleNotFoundError as exc:\n"
        f"    assert exc.name == {missing_name!r}\n"
        "else:\n"
        "    raise AssertionError('component import unexpectedly succeeded')\n",
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("component", "message"),
    [("_tensor", "tensor ABI mismatch"), ("_eda", "EDA ABI mismatch")],
)
def test_non_module_not_found_import_errors_propagate(tmp_path, component, message):
    root = make_package(
        tmp_path,
        **{component[1:]: f"raise ImportError({message!r})\n"},
    )
    result = run_python(
        root,
        "try:\n"
        "    import labneura\n"
        "except ImportError as exc:\n"
        f"    assert str(exc) == {message!r}\n"
        "else:\n"
        "    raise AssertionError('component import unexpectedly succeeded')\n",
    )
    assert result.returncode == 0, result.stderr
