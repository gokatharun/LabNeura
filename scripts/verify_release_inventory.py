#!/usr/bin/env python3
"""Fail release qualification when any required LabNeura artifact is absent."""

from __future__ import annotations

import argparse
from pathlib import Path

from packaging.utils import canonicalize_name, parse_wheel_filename

try:
    from scripts.release_inventory import CPYTHON_TAGS, NATIVE_DISTRIBUTIONS, PLATFORMS
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from release_inventory import CPYTHON_TAGS, NATIVE_DISTRIBUTIONS, PLATFORMS


def platform_family(tag: str) -> str | None:
    if tag == "win_amd64":
        return "windows-amd64"
    if tag.startswith("macosx_") and tag.endswith("_arm64"):
        return "macos-arm64"
    if tag.startswith("macosx_") and tag.endswith("_x86_64"):
        return "macos-x86_64"
    if tag.startswith(("manylinux", "musllinux")) and tag.endswith("_aarch64"):
        return "linux-aarch64"
    if tag.startswith(("manylinux", "musllinux")) and tag.endswith("_x86_64"):
        return "linux-x86_64"
    return None


def validate(directory: Path) -> tuple[int, int]:
    expected = {(distribution, python, platform)
                for distribution in NATIVE_DISTRIBUTIONS
                for python in CPYTHON_TAGS for platform in PLATFORMS}
    found = set()
    umbrella = []
    unexpected = []
    for wheel in directory.glob("*.whl"):
        name, _, _, tags = parse_wheel_filename(wheel.name)
        distribution = canonicalize_name(name)
        if distribution == "labneura":
            if any(tag.interpreter == "py3" and tag.abi == "none" and tag.platform == "any"
                   for tag in tags):
                umbrella.append(wheel.name)
            else:
                unexpected.append(wheel.name)
            continue
        if distribution not in NATIVE_DISTRIBUTIONS:
            unexpected.append(wheel.name)
            continue
        for tag in tags:
            family = platform_family(tag.platform)
            if tag.interpreter in CPYTHON_TAGS and tag.abi == tag.interpreter and family:
                found.add((distribution, tag.interpreter, family))
    missing = expected - found
    extra = found - expected
    if len(umbrella) != 1 or missing or extra or unexpected:
        details = [f"umbrella={umbrella}", f"unexpected={unexpected}"]
        details += [f"missing {item}" for item in sorted(missing)]
        details += [f"extra {item}" for item in sorted(extra)]
        raise AssertionError("release artifact inventory failed:\n" + "\n".join(details))
    return len(found), len(umbrella)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    native, pure = validate(args.directory)
    print(f"PASS release inventory: {native} native ABI/platform entries, {pure} umbrella wheel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
