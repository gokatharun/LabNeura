#!/usr/bin/env python3
"""Verify the three LabNeura wheels and their isolated install contracts."""

from __future__ import annotations

import argparse
import email
import json
import os
import platform
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

from packaging.requirements import Requirement

DISTRIBUTIONS = ("labneura", "labneura-tensor", "labneura-eda")
NATIVE_DISTRIBUTIONS = ("labneura-tensor", "labneura-eda")
EXPECTED_DEPENDENCIES = {
    "labneura": ("labneura-tensor>=0.5,<0.6", "labneura-eda>=0.5,<0.6"),
    "labneura-tensor": (),
    "labneura-eda": (),
}


def normalized_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def run(command, *, cwd=None, env=None, expected=0):
    result = subprocess.run([str(part) for part in command], cwd=cwd, env=env,
                            text=True, capture_output=True, check=False)
    if result.returncode != expected:
        raise RuntimeError(
            f"command returned {result.returncode}, expected {expected}: "
            f"{' '.join(map(str, command))}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def wheel_index(wheelhouse: Path):
    wheels = {}
    for path in wheelhouse.glob("*.whl"):
        with zipfile.ZipFile(path) as archive:
            metadata_paths = [item for item in archive.namelist()
                              if item.endswith(".dist-info/METADATA")]
            if len(metadata_paths) != 1:
                raise AssertionError(f"{path.name}: expected one METADATA file")
            metadata = email.message_from_bytes(archive.read(metadata_paths[0]))
            name = normalized_name(metadata["Name"])
            if name in wheels:
                raise AssertionError(f"duplicate wheel for {name}")
            wheels[name] = (path.resolve(), metadata)
    if set(wheels) != set(DISTRIBUTIONS):
        raise AssertionError(
            f"wheelhouse distributions differ: expected {list(DISTRIBUTIONS)}, "
            f"found {sorted(wheels)}"
        )
    return wheels


def dependencies(values):
    return tuple(sorted(str(Requirement(value)) for value in values))


def inspect_native(name: str, wheel: Path, member: str, workdir: Path):
    output = workdir / ".native-inspect" / name / Path(member).name
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        output.write_bytes(archive.read(member))
    if sys.platform == "darwin":
        details = run(["otool", "-L", output]).stdout
    elif sys.platform.startswith("linux"):
        details = run(["ldd", output]).stdout
        if "not found" in details:
            raise AssertionError(f"{name} has an unavailable dependency:\n{details}")
    elif sys.platform == "win32":
        details = run(["dumpbin", "/dependents", output]).stdout
    else:
        details = "dependency inspection unavailable"
    if "slang" in details.lower():
        raise AssertionError(f"{name} links Slang:\n{details}")
    print(f"RUNTIME PASS {name}: dependencies checked on {platform.platform()}")


def check_artifacts(wheelhouse: Path, expected_version: str):
    wheels = wheel_index(wheelhouse)
    owners = {}
    for name, (path, metadata) in wheels.items():
        with zipfile.ZipFile(path) as archive:
            files = [item for item in archive.namelist() if not item.endswith("/")]
            metadata_root = next(item.split("/", 1)[0] for item in files
                                 if item.endswith(".dist-info/METADATA"))
            payload = [item for item in files if not item.startswith(metadata_root + "/")]
            if metadata["Version"] != expected_version:
                raise AssertionError(f"{path.name}: version {metadata['Version']} != {expected_version}")
            actual = metadata.get_all("Requires-Dist", [])
            if dependencies(actual) != dependencies(EXPECTED_DEPENDENCIES[name]):
                raise AssertionError(f"{path.name}: unexpected dependencies {actual}")
            wheel_metadata = email.message_from_bytes(archive.read(metadata_root + "/WHEEL"))
            tags = wheel_metadata.get_all("Tag", [])
            pure = wheel_metadata.get("Root-Is-Purelib", "").lower() == "true"
            if name == "labneura":
                if payload != ["labneura/__init__.py"] or not pure:
                    raise AssertionError(f"umbrella payload is unexpected: {payload}")
                if not all(tag.endswith("-none-any") for tag in tags):
                    raise AssertionError(f"umbrella tags are not platform-independent: {tags}")
            else:
                stem = "labneura/_tensor" if name.endswith("tensor") else "labneura/_eda"
                native = [item for item in payload if item.startswith(stem) and
                          item.endswith((".so", ".pyd", ".dylib"))]
                if pure or len(native) != 1 or len(payload) != 1:
                    raise AssertionError(f"{name} must own one native extension: {payload}")
                if any(tag.endswith("-none-any") for tag in tags):
                    raise AssertionError(f"{name} has a pure-Python tag: {tags}")
                inspect_native(name, path, native[0], wheelhouse)
            for item in payload:
                if item in owners:
                    raise AssertionError(f"{item} owned by {owners[item]} and {name}")
                owners[item] = name
            print(f"ARTIFACT PASS {path.name}: payload={payload}; tags={tags}")
    return wheels


def clean_environment():
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def environment_python(path: Path) -> Path:
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def verify_install(name: str, direct: tuple[str, ...], expected: set[str], wheels,
                   wheelhouse: Path, workdir: Path, code: str):
    env_dir = workdir / name
    venv.EnvBuilder(with_pip=True).create(env_dir)
    python = environment_python(env_dir)
    cwd = workdir / f"{name}-cwd"
    cwd.mkdir()
    command = [python, "-m", "pip", "install", "--no-index", "--find-links", wheelhouse]
    command.extend(wheels[item][0] for item in direct)
    run(command, cwd=cwd, env=clean_environment())
    listed = json.loads(run([python, "-m", "pip", "list", "--format=json"],
                            cwd=cwd, env=clean_environment()).stdout)
    actual = {normalized_name(item["name"]) for item in listed
              if normalized_name(item["name"]).startswith("labneura")}
    if actual != expected:
        raise AssertionError(f"{name}: installed {sorted(actual)}, expected {sorted(expected)}")
    run([python, "-m", "pip", "check"], cwd=cwd, env=clean_environment())
    run([python, "-c", code], cwd=cwd, env=clean_environment())
    print(f"INSTALL PASS {name}: {sorted(actual)}")


def verify_matrix(wheels, wheelhouse: Path, workdir: Path):
    workdir.mkdir(parents=True)
    verify_install("tensor-only", ("labneura-tensor",), {"labneura-tensor"}, wheels,
                   wheelhouse, workdir, "import labneura._tensor as t; assert t.Tensor")
    verify_install("eda-only", ("labneura-eda",), {"labneura-eda"}, wheels,
                   wheelhouse, workdir, "import labneura._eda as e; assert e.Module")
    verify_install(
        "umbrella", ("labneura",), set(DISTRIBUTIONS), wheels, wheelhouse, workdir,
        "import labneura, labneura.eda; "
        "assert hasattr(labneura, 'Tensor'); assert labneura.eda.Module",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wheelhouse", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    if args.workdir.exists():
        raise SystemExit(f"refusing to overwrite existing matrix directory: {args.workdir}")
    wheels = check_artifacts(args.wheelhouse.resolve(), args.version)
    verify_matrix(wheels, args.wheelhouse.resolve(), args.workdir.resolve())
    print("All three LabNeura wheel artifact and installation checks passed.")


if __name__ == "__main__":
    main()
