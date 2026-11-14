#!/usr/bin/env python3
"""Keep the package metadata and native module version in sync."""

import argparse
import re
import sys
from pathlib import Path


VERSION_PATTERN = re.compile(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?")
PYPROJECT_VERSION = re.compile(r'(?m)^version = "(?P<version>[^"]+)"$')
CPP_VERSION = re.compile(
    r'(m\.attr\("__version__"\) = ")(?P<version>[^"]+)(";)'
)
CORE_VERSION = re.compile(r'(?m)^__version__ = "(?P<version>[^"]+)"$')
CMAKE_PACKAGE_VERSION = re.compile(
    r'(?m)^\s+VERSION (?P<version>[^\s]+)$'
)

VERSION_DECLARATIONS = (
    (Path("pyproject.toml"), PYPROJECT_VERSION),
    (Path("CMakeLists.txt"), CMAKE_PACKAGE_VERSION),
    (Path("python/labneura_py.cpp"), CPP_VERSION),
    (Path("packaging/core/src/labneura/__init__.py"), CORE_VERSION),
    (Path("packaging/tensor/pyproject.toml"), PYPROJECT_VERSION),
    (Path("packaging/eda/pyproject.toml"), PYPROJECT_VERSION),
    (Path("packaging/compat/pyproject.toml"), PYPROJECT_VERSION),
    (Path("packaging/all/pyproject.toml"), PYPROJECT_VERSION),
)


def version_files(root):
    return tuple((root / relative, pattern) for relative, pattern in VERSION_DECLARATIONS)


def read_versions(root):
    declarations = []
    for path, pattern in version_files(root):
        source = path.read_text(encoding="utf-8")
        matches = list(pattern.finditer(source))
        if len(matches) != 1:
            raise ValueError(
                f"could not uniquely locate version declaration in {path.relative_to(root)}"
            )
        match = matches[0]
        declarations.append((path, source, match, match.group("version")))
    return declarations


def check_version(root):
    declarations = read_versions(root)
    expected = declarations[0][3]
    mismatches = [
        f"{path.relative_to(root)} has {version}"
        for path, _, _, version in declarations
        if version != expected
    ]
    if mismatches:
        raise ValueError(
            f"version mismatch: pyproject.toml has {expected}; "
            + "; ".join(mismatches)
        )
    return expected


def update_version(root, version):
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"invalid version: {version!r}; expected MAJOR.MINOR.PATCH")
    for path, source, match, _ in read_versions(root):
        updated = (
            source[:match.start("version")]
            + version
            + source[match.end("version"):]
        )
        path.write_text(updated, encoding="utf-8")


def bump_patch(root):
    current = check_version(root)
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", current)
    if not match:
        raise ValueError(
            f"cannot automatically patch-bump non-stable version {current!r}; "
            "pass VERSION=MAJOR.MINOR.PATCH explicitly"
        )
    major, minor, patch = (int(part) for part in match.groups())
    return f"{major}.{minor}.{patch + 1}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    version_action = parser.add_mutually_exclusive_group(required=True)
    version_action.add_argument("--version", help="set an explicit package version")
    version_action.add_argument(
        "--bump-patch", action="store_true", help="increment the stable patch version"
    )
    version_action.add_argument(
        "--check", action="store_true", help="check all package version declarations match"
    )
    version_action.add_argument(
        "--get-version", action="store_true", help="print the matching current version"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    try:
        if args.check:
            print(check_version(args.root))
        elif args.get_version:
            print(check_version(args.root))
        elif args.bump_patch:
            version = bump_patch(args.root)
            update_version(args.root, version)
            print(version)
        else:
            update_version(args.root, args.version)
            print(args.version)
    except (OSError, ValueError) as error:
        print(f"version update failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
