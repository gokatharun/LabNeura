#!/usr/bin/env python3
"""Enforce LabNeura native component ownership and build boundaries."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}
INCLUDE_RE = re.compile(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]", re.MULTILINE)
PATH_RE = re.compile(r"(?P<path>(?:src|include)/labneura/[A-Za-z0-9_./+-]+\.(?:cxx|cpp|cc|c))")
ISA_FLAGS = ("-mavx", "-mf16c", "/arch:AVX")


@dataclass(frozen=True)
class Violation:
    rule: str
    path: str
    detail: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.path}: {self.detail}"


def native_files(root: Path, relative: str):
    base = root / relative
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("*") if path.is_file() and path.suffix in CPP_SUFFIXES)


def includes(path: Path) -> tuple[str, ...]:
    return tuple(INCLUDE_RE.findall(path.read_text(encoding="utf-8", errors="ignore")))


def check_dependency_boundaries(root: Path) -> list[Violation]:
    violations = []
    boundaries = (
        ("src/labneura/eda", "labneura/tensor/", "EDA must not depend on Tensor"),
        ("src/labneura/tensor", "labneura/eda/", "Tensor must not depend on EDA"),
        ("src/labneura/core", "labneura/tensor/", "Core must be domain independent"),
        ("src/labneura/core", "labneura/eda/", "Core must be domain independent"),
    )
    for directory, forbidden, detail in boundaries:
        for path in native_files(root, directory):
            if any(item.startswith(forbidden) for item in includes(path)):
                violations.append(Violation("component-dependency", str(path.relative_to(root)), detail))
    return violations


def check_component_ownership(root: Path) -> list[Violation]:
    violations = []
    tensor_root = root / "src/labneura/tensor"
    core_root = tensor_root / "core"
    kernel_markers = ("backend", "kernel", "ops_", "tensor_storage", "parallel", "runtime_internal")
    for path in native_files(root, "src/labneura"):
        relative = path.relative_to(root)
        lowered = path.name.lower()
        if any(marker in lowered for marker in kernel_markers):
            if str(relative).startswith("src/labneura/core/") and "cpu_features" in lowered:
                continue
            if tensor_root in path.parents and core_root not in path.parents:
                violations.append(Violation(
                    "tensor-kernel-ownership", str(relative),
                    "Tensor computational infrastructure belongs under tensor/core",
                ))
            elif tensor_root not in path.parents and "src/labneura/eda/" not in str(relative):
                violations.append(Violation(
                    "tensor-kernel-ownership", str(relative),
                    "numerical backend/kernel is outside Tensor Core",
                ))
    for path in native_files(root, "src/labneura/core"):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(word in text for word in ("quantiz", "netlist", "verilog", "systemverilog")):
            violations.append(Violation(
                "core-domain-independence", str(path.relative_to(root)),
                "domain-specific numerical or EDA terminology found in Core",
            ))
    return violations


def check_duplicate_trees(root: Path) -> list[Violation]:
    violations = []
    for relative in ("python/src", "python/include"):
        for path in native_files(root, relative):
            violations.append(Violation(
                "duplicate-native-tree", str(path.relative_to(root)),
                "python packaging must stage the authoritative src/include trees",
            ))
    return violations


def cmake_text(root: Path) -> str:
    files = [root / "CMakeLists.txt", *sorted((root / "cmake").rglob("*.cmake")),
             *sorted((root / "cmake").rglob("CMakeLists.txt"))]
    return "\n".join(path.read_text(encoding="utf-8", errors="ignore")
                      for path in files if path.exists())


def check_source_registration(root: Path) -> list[Violation]:
    registered = {match.group("path") for match in PATH_RE.finditer(cmake_text(root))}
    violations = []
    for path in native_files(root, "src/labneura"):
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        if relative not in registered:
            violations.append(Violation(
                "cmake-source-registration", relative,
                "production source is absent from component CMake inventories",
            ))
    return violations


def check_isa_isolation(root: Path) -> list[Violation]:
    violations = []
    # Source-level CMake properties are the supported isolation mechanism. A
    # target-wide AVX/F16C option makes generic dispatch unsafe on older x86.
    block_re = re.compile(r"target_compile_options\s*\((.*?)\)", re.DOTALL)
    for cmake in [root / "CMakeLists.txt", *sorted((root / "cmake").rglob("*.cmake")),
                  *sorted((root / "cmake").rglob("CMakeLists.txt"))]:
        if not cmake.exists():
            continue
        text = cmake.read_text(encoding="utf-8", errors="ignore")
        for match in block_re.finditer(text):
            block = match.group(1)
            if any(flag in block for flag in ISA_FLAGS):
                target = block.split()[0] if block.split() else "<unknown>"
                violations.append(Violation(
                    "portable-isa-flags", str(cmake.relative_to(root)),
                    f"target-wide ISA option on {target}; use per-source SIMD properties",
                ))
    return violations


def validate(root: Path) -> list[Violation]:
    checks = (
        check_dependency_boundaries, check_component_ownership,
        check_duplicate_trees, check_source_registration, check_isa_isolation,
    )
    return [violation for check in checks for violation in check(root)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    violations = validate(args.root.resolve())
    if violations:
        print("Architecture validation failed:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1
    print("PASS: native component ownership, dependencies, source registration, and ISA isolation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
