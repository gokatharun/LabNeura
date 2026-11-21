#!/usr/bin/env python3
"""Build a self-contained labneura-tensor source distribution."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TREES = ("assets", "cmake", "include", "python", "src")
FILES = ("CMakeLists.txt", "LICENSE")


def stage_project(repository: Path, destination: Path) -> None:
    ignored = shutil.ignore_patterns(
        "__pycache__", "*.pyc", "*.pyo", "*.so", "*.dylib", "*.dll",
        "*.pyd", "*.a", "*.o", ".DS_Store", "*.bak",
    )
    for name in TREES:
        shutil.copytree(repository / name, destination / name, ignore=ignored)
    for name in FILES:
        shutil.copy2(repository / name, destination / name)
    shutil.copy2(repository / "packaging/tensor/README.md", destination / "README.md")
    pyproject = (repository / "packaging/tensor/pyproject.toml").read_text(encoding="utf-8")
    source_setting = 'cmake.source-dir = "../.."'
    if pyproject.count(source_setting) != 1:
        raise RuntimeError("Tensor pyproject does not contain the expected CMake source-dir")
    (destination / "pyproject.toml").write_text(
        pyproject.replace(source_setting, 'cmake.source-dir = "."'), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repository, outdir = args.repository.resolve(), args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="labneura-tensor-sdist-") as temporary:
        stage = Path(temporary) / "labneura-tensor"
        stage.mkdir()
        stage_project(repository, stage)
        subprocess.run([
            sys.executable, "-m", "build", "--sdist", "--no-isolation",
            "--outdir", str(outdir), str(stage),
        ], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
