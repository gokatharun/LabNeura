#!/usr/bin/env python3
"""Validate the Windows core/EDA wheel pair before runtime qualification."""

from __future__ import annotations

import argparse
import email
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename


def _metadata(archive: zipfile.ZipFile, wheel: Path):
    paths = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    if len(paths) != 1:
        raise AssertionError(f"{wheel.name}: expected one METADATA file, found {paths}")
    return email.message_from_bytes(archive.read(paths[0]))


def _requirements(metadata) -> set[str]:
    return {str(Requirement(value)) for value in metadata.get_all("Requires-Dist", [])}


def validate(dist: Path, expected_version: str, extract_dir: Path) -> Path:
    wheels = sorted(dist.glob("*.whl"))
    indexed = {}
    for wheel in wheels:
        name, version, build, tags = parse_wheel_filename(wheel.name)
        canonical = canonicalize_name(name)
        if canonical in indexed:
            raise AssertionError(f"duplicate {canonical} wheel: {wheel.name}")
        indexed[canonical] = (wheel, str(version), build, tags)

    expected = {"labneura-core", "labneura-eda"}
    if set(indexed) != expected:
        raise AssertionError(
            f"expected only {sorted(expected)}, found {sorted(indexed)} in {dist}"
        )

    core, core_version, _, core_tags = indexed["labneura-core"]
    eda, eda_version, _, eda_tags = indexed["labneura-eda"]
    if {core_version, eda_version} != {expected_version}:
        raise AssertionError(
            f"wheel versions differ from {expected_version}: "
            f"{core.name}={core_version}, {eda.name}={eda_version}"
        )
    if not any(
        tag.interpreter == "py3" and tag.abi == "none" and tag.platform == "any"
        for tag in core_tags
    ):
        raise AssertionError(f"{core.name}: expected a py3-none-any core wheel")
    if not any(
        tag.interpreter == "cp311"
        and tag.abi == "cp311"
        and tag.platform == "win_amd64"
        for tag in eda_tags
    ):
        raise AssertionError(f"{eda.name}: expected cp311-cp311-win_amd64, got {eda_tags}")

    with zipfile.ZipFile(core) as archive:
        metadata = _metadata(archive, core)
        if canonicalize_name(metadata["Name"]) != "labneura-core":
            raise AssertionError(f"{core.name}: incorrect Name metadata")
        if _requirements(metadata):
            raise AssertionError(f"{core.name}: core wheel must have no dependencies")

    with zipfile.ZipFile(eda) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        metadata = _metadata(archive, eda)
        if canonicalize_name(metadata["Name"]) != "labneura-eda":
            raise AssertionError(f"{eda.name}: incorrect Name metadata")
        if _requirements(metadata) != {"labneura-core<0.6,>=0.5"}:
            raise AssertionError(
                f"{eda.name}: unexpected dependencies {_requirements(metadata)}"
            )
        forbidden = [name for name in names if "slang" in name.lower()]
        if forbidden:
            raise AssertionError(f"{eda.name}: bundled Slang content: {forbidden}")
        bundled_dlls = [name for name in names if name.lower().endswith(".dll")]
        if bundled_dlls:
            raise AssertionError(f"{eda.name}: unexpected bundled DLLs: {bundled_dlls}")
        extensions = [name for name in names if name.lower().endswith(".pyd")]
        if len(extensions) != 1 or not Path(extensions[0]).name.startswith("_eda"):
            raise AssertionError(f"{eda.name}: expected one _eda PYD, found {extensions}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        extension = extract_dir / Path(extensions[0]).name
        extension.write_bytes(archive.read(extensions[0]))

    print(f"PASS {core.name}: version={core_version}, tags={sorted(map(str, core_tags))}")
    print(f"PASS {eda.name}: version={eda_version}, tags={sorted(map(str, eda_tags))}")
    print(f"EDA_EXTENSION={extension.resolve()}")
    return extension.resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--extract-dir", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    extension = validate(args.dist, args.version, args.extract_dir)
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"extension={extension}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
