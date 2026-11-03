import subprocess
import sys
from pathlib import Path

import pytest


VERSION_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "update_version.py"
)


def version_root(tmp_path, pyproject_version="1.2.3", cpp_version="1.2.3"):
    (tmp_path / "python").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "labneura"\nversion = "{pyproject_version}"\n',
        encoding="utf-8",
    )
    (tmp_path / "python" / "labneura_py.cpp").write_text(
        f'm.attr("__version__") = "{cpp_version}";\n',
        encoding="utf-8",
    )
    return tmp_path


def run_version_script(root, *arguments):
    return subprocess.run(
        [sys.executable, str(VERSION_SCRIPT), "--root", str(root), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )


def test_version_script_patch_bump_keeps_declarations_in_sync(tmp_path):
    root = version_root(tmp_path)

    result = run_version_script(root, "--bump-patch")

    assert result.returncode == 0
    assert result.stdout.strip() == "1.2.4"
    assert 'version = "1.2.4"' in (root / "pyproject.toml").read_text()
    assert '"1.2.4"' in (root / "python" / "labneura_py.cpp").read_text()
    assert run_version_script(root, "--check").stdout.strip() == "1.2.4"


def test_version_script_sets_release_tag_version(tmp_path):
    root = version_root(tmp_path, "1.2.3", "1.2.3")

    result = run_version_script(root, "--version", "2.0.0")

    assert result.returncode == 0
    assert result.stdout.strip() == "2.0.0"
    assert run_version_script(root, "--get-version").stdout.strip() == "2.0.0"


def test_version_script_rejects_mismatched_declarations_for_patch_bump(tmp_path):
    root = version_root(tmp_path, "1.2.3", "1.2.2")

    result = run_version_script(root, "--bump-patch")

    assert result.returncode != 0
    assert "version mismatch" in result.stderr
    assert 'version = "1.2.3"' in (root / "pyproject.toml").read_text()


def test_version_script_rejects_invalid_explicit_version(tmp_path):
    root = version_root(tmp_path)

    result = run_version_script(root, "--version", "v2.0")

    assert result.returncode != 0
    assert "invalid version" in result.stderr


@pytest.mark.parametrize("version", ["1.2.3rc1", "1.2.3.dev1"])
def test_version_script_requires_explicit_version_for_prerelease_patch_bump(
    tmp_path, version
):
    root = version_root(tmp_path, version, version)

    result = run_version_script(root, "--bump-patch")

    assert result.returncode != 0
    assert "non-stable version" in result.stderr
