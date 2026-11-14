#!/usr/bin/env python3
"""Verify LabNeura wheel ownership and isolated install matrices."""

from __future__ import annotations

import argparse
import email
import importlib.machinery
import json
import os
import platform
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

from packaging.requirements import Requirement


VERSION = "0.4.2"
COMPONENTS = ("labneura-core", "labneura-tensor", "labneura-eda")
NATIVE_COMPONENTS = ("labneura-tensor", "labneura-eda")
METADISTRIBUTIONS = ("labneura", "labneura-all")
EXPECTED_DEPENDENCIES = {
    "labneura-core": (),
    "labneura-tensor": ("labneura-core>=0.4.2,<0.5",),
    "labneura-eda": ("labneura-core>=0.4.2,<0.5",),
    "labneura": (
        "labneura-core>=0.4.2,<0.5",
        "labneura-tensor>=0.4.2,<0.5",
        "labneura-eda>=0.4.2,<0.5",
    ),
    "labneura-all": (
        "labneura-core>=0.4.2,<0.5",
        "labneura-tensor>=0.4.2,<0.5",
        "labneura-eda>=0.4.2,<0.5",
    ),
}


def normalized_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def normalized_dependencies(dependencies):
    """Canonicalize requirement syntax and ignore metadata ordering."""
    return tuple(sorted(str(Requirement(dependency)) for dependency in dependencies))


def check_wheel_tags(name, wheel_tags, artifact_name):
    if not wheel_tags:
        raise AssertionError(f"{artifact_name}: missing wheel platform tags")
    if name in NATIVE_COMPONENTS and any(
        tag.endswith("-none-any") for tag in wheel_tags
    ):
        raise AssertionError(
            f"{artifact_name}: native wheel has pure-Python tag {wheel_tags}"
        )
    if name not in NATIVE_COMPONENTS and not all(
        tag.endswith("-none-any") for tag in wheel_tags
    ):
        raise AssertionError(
            f"{artifact_name}: pure-Python wheel has platform tags {wheel_tags}"
        )


def run(command, *, cwd=None, env=None, expected=0):
    result = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != expected:
        raise RuntimeError(
            f"command returned {result.returncode}, expected {expected}: "
            f"{' '.join(map(str, command))}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def wheel_index(wheelhouse: Path):
    wheels = {}
    for path in wheelhouse.glob("*.whl"):
        with zipfile.ZipFile(path) as archive:
            metadata_paths = [
                name for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_paths) != 1:
                raise AssertionError(f"{path.name}: expected one METADATA file")
            metadata = email.message_from_bytes(archive.read(metadata_paths[0]))
            name = normalized_name(metadata["Name"])
            if name in wheels:
                raise AssertionError(f"duplicate wheel for {name}")
            wheels[name] = (path.resolve(), metadata)
    expected = set(EXPECTED_DEPENDENCIES)
    if set(wheels) != expected:
        raise AssertionError(
            f"wheelhouse distributions differ: expected {sorted(expected)}, "
            f"found {sorted(wheels)}"
        )
    return wheels


def check_wheel_artifacts(wheelhouse: Path):
    wheels = wheel_index(wheelhouse)
    owners = {}
    native_paths = []
    extension_suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    for name, (path, metadata) in wheels.items():
        with zipfile.ZipFile(path) as archive:
            files = [item for item in archive.namelist() if not item.endswith("/")]
            metadata_root = next(
                item.split("/", 1)[0] for item in files
                if item.endswith(".dist-info/METADATA")
            )
            payload = [item for item in files if not item.startswith(metadata_root + "/")]
            if metadata["Version"] != VERSION:
                raise AssertionError(f"{path.name}: unexpected version {metadata['Version']}")
            actual_dependencies = tuple(metadata.get_all("Requires-Dist", []))
            if normalized_dependencies(actual_dependencies) != normalized_dependencies(
                EXPECTED_DEPENDENCIES[name]
            ):
                raise AssertionError(
                    f"{path.name}: dependencies {actual_dependencies!r} do not match "
                    f"{EXPECTED_DEPENDENCIES[name]!r}"
                )

            wheel_metadata_path = metadata_root + "/WHEEL"
            wheel_metadata = email.message_from_bytes(archive.read(wheel_metadata_path))
            pure = wheel_metadata.get("Root-Is-Purelib", "").lower() == "true"

            if name == "labneura-core":
                if payload != ["labneura/__init__.py"]:
                    raise AssertionError(f"core wheel payload is unexpected: {payload}")
                if not pure:
                    raise AssertionError("core wheel must be pure Python")
            elif name in METADISTRIBUTIONS:
                if payload:
                    raise AssertionError(
                        f"{name} must be metadata-only; found payload {payload}"
                    )
                if not pure:
                    raise AssertionError(f"{name} wheel must be pure Python")
            else:
                if pure:
                    raise AssertionError(f"{name} wheel unexpectedly marked pure")
                expected_stem = "labneura/_tensor" if name.endswith("tensor") else "labneura/_eda"
                native = [
                    item for item in payload
                    if item.startswith(expected_stem)
                    and any(item.endswith(suffix) for suffix in extension_suffixes)
                ]
                if len(native) != 1 or len(payload) != 1:
                    raise AssertionError(
                        f"{name} must own only its native extension; found {payload}"
                    )
                if any(item.endswith((".a", ".h", ".hpp")) for item in payload):
                    raise AssertionError(f"{name} wheel contains C++ development files")
                native_paths.append((name, path, native[0]))

            for item in payload:
                if item in owners:
                    raise AssertionError(
                        f"package path {item} is owned by both {owners[item]} and {name}"
                    )
                owners[item] = name

        wheel_tags = wheel_metadata.get_all("Tag", [])
        check_wheel_tags(name, wheel_tags, path.name)
        print(f"ARTIFACT PASS {path.name}: payload={payload}; tags={wheel_tags}")

    for name, wheel, archive_path in native_paths:
        inspect_native_dependencies(name, wheel, archive_path, wheelhouse)
    return wheels


def inspect_native_dependencies(name, wheel, archive_path, wheelhouse):
    inspect_dir = wheelhouse / ".native-inspect" / name
    inspect_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        binary = inspect_dir / Path(archive_path).name
        binary.write_bytes(archive.read(archive_path))

    if sys.platform == "darwin":
        dependencies = run(["otool", "-L", binary]).stdout
        load_commands = run(["otool", "-l", binary]).stdout
        combined = dependencies + load_commands
        if any(token in combined for token in ("/Users/runner/work", str(Path.cwd()))):
            raise AssertionError(f"{name} contains a build-machine runtime path")
        rpaths = [line.strip() for line in load_commands.splitlines() if "path " in line]
        if any("/tmp/" in line for line in rpaths):
            raise AssertionError(f"{name} contains a temporary RPATH: {rpaths}")
    elif sys.platform.startswith("linux"):
        dependencies = run(["ldd", binary]).stdout
        if "not found" in dependencies:
            raise AssertionError(f"{name} has an unavailable runtime dependency:\n{dependencies}")
        dynamic = run(["readelf", "-d", binary]).stdout
        if any(token in dynamic for token in ("/home/runner/work", str(Path.cwd()), "/tmp/")):
            raise AssertionError(f"{name} contains a build-machine RPATH/RUNPATH")
    print(f"RUNTIME PASS {name}: native dependencies resolved on {platform.platform()}")


def venv_python(path: Path):
    if os.name == "nt":
        return path / "Scripts" / "python.exe"
    return path / "bin" / "python"


def create_environment(workdir: Path, name: str):
    environment = workdir / name
    venv.EnvBuilder(with_pip=True).create(environment)
    python = venv_python(environment)
    return environment, python


def clean_python_environment():
    environment = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
    ):
        environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def install_wheels(python: Path, wheelhouse: Path, wheels, direct_names, *, cwd):
    command = [
        python,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-index",
        "--find-links",
        wheelhouse,
    ]
    command.extend(wheels[name][0] for name in direct_names)
    run(command, cwd=cwd, env=clean_python_environment())


def installed_labneura_packages(python: Path, cwd: Path):
    result = run(
        [python, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
        cwd=cwd,
        env=clean_python_environment(),
    )
    return {
        normalized_name(item["name"])
        for item in json.loads(result.stdout)
        if normalized_name(item["name"]).startswith("labneura")
    }


def pip_check(python: Path, cwd: Path, *, expected=0):
    result = run(
        [python, "-m", "pip", "check", "--disable-pip-version-check"],
        cwd=cwd,
        env=clean_python_environment(),
        expected=expected,
    )
    return result.stdout + result.stderr


COMMON_CHECKS = r'''
import importlib
import importlib.util
import sys
from pathlib import Path
import sysconfig

import labneura
site = Path(sysconfig.get_paths()["purelib"]).resolve()
assert labneura.__file__ is not None
Path(labneura.__file__).resolve().relative_to(site)
assert labneura.__version__ == "0.4.2"

def in_site(module):
    assert module.__file__ is not None
    Path(module.__file__).resolve().relative_to(site)
    print(f"PASS installed module path: {module.__name__}={module.__file__}")

def unavailable(name):
    assert importlib.util.find_spec(name) is None, name
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as error:
        assert error.name == name, (name, error.name)
    else:
        raise AssertionError(f"{name} unexpectedly imported")
    print(f"PASS absent optional component: {name} (discovery and import)")

def tensor_smoke():
    import labneura._tensor as native_tensor
    from labneura import Tensor
    in_site(native_tensor)
    assert any(
        native_tensor.__file__.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )
    assert Tensor is native_tensor.Tensor
    left = Tensor([1.0, 2.0])
    right = Tensor([3.0, 4.0])
    assert (left + right).data_fp32() == [4.0, 6.0]
    assert callable(labneura.quantize_int8)
    assert labneura.quantize_int8(1.0, 0.5) == 2

def eda_smoke():
    import labneura._eda as native_eda
    import labneura.eda as eda
    assert eda is native_eda
    assert labneura.eda is native_eda
    assert eda.Module is native_eda.Module
    in_site(native_eda)
    assert any(
        native_eda.__file__.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )
    source = Path("wheel_matrix_fixture.v")
    source.write_text(
        "module top(input wire a, output wire y);\n"
        "  assign y = a;\n"
        "endmodule\n",
        encoding="utf-8",
    )
    parsed = eda.parse_verilog(str(source))
    assert parsed.name == "top"
    assert [port.name for port in parsed.ports] == ["a", "y"]
'''


def scenario_code(scenario):
    if scenario == "W1":
        return COMMON_CHECKS + r'''
assert not hasattr(labneura, "Tensor")
assert not hasattr(labneura, "eda")
print("PASS W1 Tensor and eda attributes are absent")
unavailable("labneura._tensor")
unavailable("labneura._eda")
unavailable("labneura.eda")
'''
    if scenario == "W2":
        return COMMON_CHECKS + r'''
assert importlib.util.find_spec("labneura._tensor") is not None
tensor_smoke()
assert not hasattr(labneura, "eda")
print("PASS W2 eda attribute is absent")
unavailable("labneura._eda")
unavailable("labneura.eda")
'''
    if scenario == "W3":
        return COMMON_CHECKS + r'''
assert importlib.util.find_spec("labneura._eda") is not None
eda_smoke()
assert not hasattr(labneura, "Tensor")
print("PASS W3 Tensor attribute is absent")
unavailable("labneura._tensor")
try:
    from labneura import Tensor
except ImportError:
    pass
else:
    raise AssertionError("Tensor unexpectedly exported")
'''
    if scenario in (
        "W4",
        "W5",
        "W6",
        "O1",
        "O2",
        "O3",
        "keep-after-metadata-uninstall",
    ):
        return COMMON_CHECKS + r'''
assert importlib.util.find_spec("labneura._tensor") is not None
assert importlib.util.find_spec("labneura._eda") is not None
tensor_smoke()
eda_smoke()
'''
    if scenario == "U1":
        return COMMON_CHECKS + r'''
assert importlib.util.find_spec("labneura._eda") is not None
eda_smoke()
assert not hasattr(labneura, "Tensor")
unavailable("labneura._tensor")
'''
    if scenario == "U2":
        return COMMON_CHECKS + r'''
assert importlib.util.find_spec("labneura._tensor") is not None
tensor_smoke()
assert not hasattr(labneura, "eda")
unavailable("labneura._eda")
unavailable("labneura.eda")
'''
    if scenario == "U3":
        return r'''
import sysconfig
from pathlib import Path
site = Path(sysconfig.get_paths()["purelib"]).resolve()
initializer = site / "labneura" / "__init__.py"
assert not initializer.exists(), initializer
'''
    raise ValueError(f"unknown scenario {scenario}")


def run_python_checks(python: Path, cwd: Path, scenario: str):
    result = run(
        [python, "-c", scenario_code(scenario)],
        cwd=cwd,
        env=clean_python_environment(),
    )
    if result.stdout:
        print(result.stdout, end="")
    print(f"{scenario} PASS imports and public APIs")


def install_scenario(
    name, direct_names, expected_packages, wheels, wheelhouse, workdir, *, check_name=None
):
    environment, python = create_environment(workdir, name)
    cwd = workdir / (name + "-cwd")
    cwd.mkdir()
    install_wheels(python, wheelhouse, wheels, direct_names, cwd=cwd)
    actual = installed_labneura_packages(python, cwd)
    if actual != set(expected_packages):
        raise AssertionError(
            f"{name}: installed {sorted(actual)}, "
            f"expected {sorted(expected_packages)}"
        )
    pip_check(python, cwd)
    run_python_checks(python, cwd, check_name or name)
    return environment, python, cwd


def verify_matrix(wheels, wheelhouse, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    install_scenario("W1", ("labneura-core",), ("labneura-core",), wheels, wheelhouse, workdir)
    install_scenario("W2", COMPONENTS[:2], COMPONENTS[:2], wheels, wheelhouse, workdir)
    install_scenario(
        "W3",
        ("labneura-core", "labneura-eda"),
        ("labneura-core", "labneura-eda"),
        wheels,
        wheelhouse,
        workdir,
    )
    install_scenario("W4", COMPONENTS, COMPONENTS, wheels, wheelhouse, workdir)
    install_scenario("W5", ("labneura",), (*COMPONENTS, "labneura"), wheels, wheelhouse, workdir)
    install_scenario(
        "W6",
        ("labneura-all",),
        (*COMPONENTS, "labneura-all"),
        wheels,
        wheelhouse,
        workdir,
    )

    orders = {
        "O1": ("labneura-core", "labneura-tensor", "labneura-eda"),
        "O2": ("labneura-core", "labneura-eda", "labneura-tensor"),
        "O3": ("labneura-tensor", "labneura-eda", "labneura-core"),
    }
    for name, order in orders.items():
        _, python = create_environment(workdir, name)
        cwd = workdir / (name + "-cwd")
        cwd.mkdir()
        for distribution in order:
            install_wheels(python, wheelhouse, wheels, (distribution,), cwd=cwd)
        if installed_labneura_packages(python, cwd) != set(COMPONENTS):
            raise AssertionError(f"{name}: unexpected installed distributions")
        pip_check(python, cwd)
        run_python_checks(python, cwd, name)

    for name, distribution, scenario in (
        ("U1", "labneura-tensor", "U1"),
        ("U2", "labneura-eda", "U2"),
        ("U3", "labneura-core", "U3"),
    ):
        _, python, cwd = install_scenario(
            name + "-base", COMPONENTS, COMPONENTS, wheels, wheelhouse, workdir,
            check_name="W4",
        )
        run(
            [python, "-m", "pip", "uninstall", "-y", distribution],
            cwd=cwd,
            env=clean_python_environment(),
        )
        run_python_checks(python, cwd, scenario)
        if name == "U3":
            check_output = pip_check(python, cwd, expected=1)
            if "labneura-core" not in check_output:
                raise AssertionError("U3 pip check did not report the removed core dependency")
            print("U3 PASS core initializer removed; pip check reports expected dependent packages")
        else:
            pip_check(python, cwd)
            print(f"{name} PASS uninstalled component did not remove sibling files")

    for name in METADISTRIBUTIONS:
        env_name = "U-meta-compat" if name == "labneura" else "U-meta-all"
        _, python = create_environment(workdir, env_name)
        cwd = workdir / (env_name + "-cwd")
        cwd.mkdir()
        install_wheels(python, wheelhouse, wheels, (name,), cwd=cwd)
        run([python, "-m", "pip", "uninstall", "-y", name], cwd=cwd, env=clean_python_environment())
        expected = set(COMPONENTS)
        actual = installed_labneura_packages(python, cwd)
        if actual != expected:
            raise AssertionError(f"{env_name}: dependencies were removed: {sorted(actual)}")
        pip_check(python, cwd)
        run_python_checks(python, cwd, "keep-after-metadata-uninstall")
        print(f"{env_name} PASS dependencies remain installed after metadata uninstall")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wheelhouse", type=Path)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    wheelhouse = args.wheelhouse.resolve()
    workdir = args.workdir.resolve()
    if workdir.exists():
        raise SystemExit(f"refusing to overwrite existing matrix directory: {workdir}")
    wheels = check_wheel_artifacts(wheelhouse)
    verify_matrix(wheels, wheelhouse, workdir)
    print("All LabNeura wheel artifact and installation matrix checks passed.")


if __name__ == "__main__":
    main()
