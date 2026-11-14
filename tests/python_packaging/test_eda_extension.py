"""Regression checks for the independently built labneura._eda extension.

Set LABNEURA_EDA_EXTENSION to a built _eda shared library to run these checks.
See tests/python_packaging/README.md for the focused command.
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
EDA_EXTENSION_ENV = "LABNEURA_EDA_EXTENSION"


@pytest.fixture
def eda_package(tmp_path):
    extension_value = os.environ.get(EDA_EXTENSION_ENV)
    if not extension_value:
        pytest.skip(
            f"set {EDA_EXTENSION_ENV} to a built _eda extension; "
            "see tests/python_packaging/README.md"
        )

    extension = Path(extension_value).expanduser().resolve()
    if not extension.is_file():
        pytest.fail(
            f"{EDA_EXTENSION_ENV} does not name a file: {extension}",
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
        "import labneura\n"
        "import labneura._eda as native_eda\n"
        f"assert Path(native_eda.__file__).resolve() == Path({str(extension_path)!r})\n"
        "assert any(native_eda.__file__.endswith(suffix) "
        "for suffix in EXTENSION_SUFFIXES)\n"
        "assert labneura.eda is native_eda\n"
    )


def test_eda_imports_and_public_symbols(eda_package):
    root, extension = eda_package
    code = native_module_check(extension) + (
        "import labneura.eda as eda\n"
        "assert eda is native_eda\n"
        "assert eda.PortDirection.INPUT.name == 'INPUT'\n"
        "assert eda.NetType.INTERNAL.name == 'INTERNAL'\n"
        "assert eda.ExpressionKind.IDENTIFIER.name == 'IDENTIFIER'\n"
        "assert eda.Module.__module__ == 'labneura._eda'\n"
        "assert eda.Project.__module__ == 'labneura._eda'\n"
        "assert eda.StandardCellLibrary.__module__ == 'labneura._eda'\n"
        "assert not hasattr(labneura, 'Tensor')\n"
        "try:\n"
        "    import labneura._tensor\n"
        "except ModuleNotFoundError as error:\n"
        "    assert error.name == 'labneura._tensor'\n"
        "else:\n"
        "    raise AssertionError('Tensor unexpectedly imported')\n"
    )
    assert_python_succeeds(root, code)


def test_eda_extension_parses_verilog(eda_package):
    root, extension = eda_package
    code = native_module_check(extension) + (
        "from pathlib import Path\n"
        "source = Path('split_extension_fixture.v')\n"
        "source.write_text("
        "'module top(input wire a, output wire y);\\n'"
        "'  assign y = a;\\n'"
        "'endmodule\\n', encoding='utf-8')\n"
        "module = native_eda.parse_verilog(str(source))\n"
        "assert module.name == 'top'\n"
        "assert [port.name for port in module.ports] == ['a', 'y']\n"
    )
    assert_python_succeeds(root, code)


def test_eda_extension_has_default_fpga_library(eda_package):
    root, extension = eda_package
    code = native_module_check(extension) + (
        "library = native_eda.default_fpga_library()\n"
        "assert library.cell_count > 0\n"
        "assert library.cells\n"
    )
    assert_python_succeeds(root, code)
