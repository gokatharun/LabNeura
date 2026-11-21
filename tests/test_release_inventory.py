from pathlib import Path

import pytest

from scripts.release_inventory import CPYTHON_TAGS, NATIVE_DISTRIBUTIONS, PLATFORMS
from scripts.verify_release_inventory import platform_family, validate


PLATFORM_TAGS = {
    "linux-x86_64": "manylinux_2_17_x86_64",
    "linux-aarch64": "manylinux_2_17_aarch64",
    "macos-arm64": "macosx_11_0_arm64",
    "macos-x86_64": "macosx_10_15_x86_64",
    "windows-amd64": "win_amd64",
}


def populate(directory: Path):
    (directory / "labneura-0.5.2-py3-none-any.whl").touch()
    for distribution in NATIVE_DISTRIBUTIONS:
        stem = distribution.replace("-", "_")
        for python in CPYTHON_TAGS:
            for platform in PLATFORMS:
                (directory / f"{stem}-0.5.2-{python}-{python}-{PLATFORM_TAGS[platform]}.whl").touch()


def test_authoritative_inventory_requires_windows_for_every_python():
    assert CPYTHON_TAGS == ("cp38", "cp39", "cp310", "cp311", "cp312", "cp313")
    assert set(PLATFORMS) == set(PLATFORM_TAGS)


def test_complete_three_distribution_inventory(tmp_path):
    populate(tmp_path)
    assert validate(tmp_path) == (60, 1)


def test_missing_windows_wheel_fails_qualification(tmp_path):
    populate(tmp_path)
    (tmp_path / "labneura_eda-0.5.2-cp38-cp38-win_amd64.whl").unlink()
    with pytest.raises(AssertionError, match="windows-amd64"):
        validate(tmp_path)


def test_platform_family_rejects_unknown_tags():
    assert platform_family("linux_ppc64le") is None
