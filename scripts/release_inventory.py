"""Authoritative Python and platform inventory for LabNeura releases."""

from __future__ import annotations

import argparse
import json

PYTHON_VERSIONS = ("3.8", "3.9", "3.10", "3.11", "3.12", "3.13")
CPYTHON_TAGS = tuple("cp" + version.replace(".", "") for version in PYTHON_VERSIONS)
NATIVE_DISTRIBUTIONS = ("labneura-tensor", "labneura-eda")
PURE_DISTRIBUTIONS = ("labneura",)
PLATFORMS = ("linux-x86_64", "linux-aarch64", "macos-arm64", "macos-x86_64", "windows-amd64")


def inventory() -> dict[str, object]:
    return {
        "python_versions": list(PYTHON_VERSIONS),
        "cpython_tags": list(CPYTHON_TAGS),
        "native_distributions": list(NATIVE_DISTRIBUTIONS),
        "pure_distributions": list(PURE_DISTRIBUTIONS),
        "platforms": list(PLATFORMS),
    }


def required_native_wheels() -> set[tuple[str, str, str]]:
    return {(dist, tag, platform) for dist in NATIVE_DISTRIBUTIONS for tag in CPYTHON_TAGS for platform in PLATFORMS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    print(json.dumps(inventory(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
