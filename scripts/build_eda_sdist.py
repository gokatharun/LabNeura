#!/usr/bin/env python3
"""Build a self-contained labneura-eda source distribution.

The split EDA pyproject normally points at the repository root.  A release
sdist therefore has to stage the native source tree beside that pyproject;
otherwise the backend emits a metadata-only archive that cannot be rebuilt.
"""

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
    for name in TREES:
        shutil.copytree(
            repository / name,
            destination / name,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                "*.pyo",
                "*.so",
                "*.dylib",
                "*.dll",
                "*.pyd",
                "*.a",
                "*.o",
                ".DS_Store",
                "*.bak",
            ),
        )
    for name in FILES:
        shutil.copy2(repository / name, destination / name)
    shutil.copy2(repository / "packaging/eda/README.md", destination / "README.md")
    pyproject = (repository / "packaging/eda/pyproject.toml").read_text(
        encoding="utf-8"
    )
    source_setting = 'cmake.source-dir = "../.."'
    if pyproject.count(source_setting) != 1:
        raise RuntimeError("EDA pyproject does not contain the expected CMake source-dir")
    (destination / "pyproject.toml").write_text(
        pyproject.replace(source_setting, 'cmake.source-dir = "."'),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--repository", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    repository = args.repository.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="labneura-eda-sdist-") as temporary:
        stage = Path(temporary) / "labneura-eda"
        stage.mkdir()
        stage_project(repository, stage)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--sdist",
                "--no-isolation",
                "--outdir",
                str(outdir),
                str(stage),
            ],
            check=True,
        )
    archives = sorted(outdir.glob("labneura_eda-*.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one EDA sdist, found {archives}")
    print(archives[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
