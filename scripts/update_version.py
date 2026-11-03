#!/usr/bin/env python3
"""Keep the package metadata and native module version in sync."""

import argparse
import re
import sys
from pathlib import Path


VERSION_PATTERN = re.compile(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?")
PYPROJECT_VERSION = re.compile(r'(?m)^version = "([^"]+)"$')
CPP_VERSION = re.compile(r'(m\.attr\("__version__"\) = ")([^"]+)(";)')


def version_files(root):
    return root / "pyproject.toml", root / "python" / "labneura_py.cpp"


def read_versions(root):
    pyproject_path, cpp_path = version_files(root)
    pyproject = pyproject_path.read_text(encoding="utf-8")
    cpp_source = cpp_path.read_text(encoding="utf-8")
    pyproject_matches = list(PYPROJECT_VERSION.finditer(pyproject))
    cpp_matches = list(CPP_VERSION.finditer(cpp_source))
    if len(pyproject_matches) != 1 or len(cpp_matches) != 1:
        raise ValueError("could not uniquely locate both version declarations")
    return (
        pyproject,
        cpp_source,
        pyproject_matches[0],
        cpp_matches[0],
        pyproject_matches[0].group(1),
        cpp_matches[0].group(2),
    )


def check_version(root):
    _, _, _, _, pyproject_version, cpp_version = read_versions(root)
    if pyproject_version != cpp_version:
        raise ValueError(
            "version mismatch: pyproject.toml has "
            f"{pyproject_version}, but python/labneura_py.cpp has {cpp_version}"
        )
    return pyproject_version


def update_version(root, version):
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"invalid version: {version!r}; expected MAJOR.MINOR.PATCH")
    pyproject, cpp_source, pyproject_match, cpp_match, _, _ = read_versions(root)
    pyproject = (
        pyproject[:pyproject_match.start(1)]
        + version
        + pyproject[pyproject_match.end(1):]
    )
    cpp_source = (
        cpp_source[:cpp_match.start(2)]
        + version
        + cpp_source[cpp_match.end(2):]
    )
    pyproject_path, cpp_path = version_files(root)
    pyproject_path.write_text(pyproject, encoding="utf-8")
    cpp_path.write_text(cpp_source, encoding="utf-8")


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
        "--check", action="store_true", help="check both version declarations match"
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
