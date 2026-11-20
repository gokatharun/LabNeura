import importlib.util
import zipfile
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_windows_wheels.py"
SPEC = importlib.util.spec_from_file_location("verify_windows_wheels", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _wheel(path: Path, name: str, requires=(), payload=()):
    distribution, version, *_ = path.name.removesuffix(".whl").split("-")
    metadata_root = f"{distribution}-{version}.dist-info"
    metadata = [
        "Metadata-Version: 2.1",
        f"Name: {name}",
        f"Version: {version}",
        *(f"Requires-Dist: {requirement}" for requirement in requires),
        "",
        "",
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{metadata_root}/METADATA", "\n".join(metadata))
        archive.writestr(f"{metadata_root}/WHEEL", "Wheel-Version: 1.0\n")
        for name, contents in payload:
            archive.writestr(name, contents)


def test_validates_exact_cp311_windows_core_eda_pair(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    _wheel(
        dist / "labneura_core-0.4.9-py3-none-any.whl",
        "labneura-core",
        payload=(("labneura/__init__.py", ""),),
    )
    _wheel(
        dist / "labneura_eda-0.4.9-cp311-cp311-win_amd64.whl",
        "labneura-eda",
        requires=("labneura-core>=0.5,<0.6",),
        payload=(("labneura/_eda.cp311-win_amd64.pyd", b"PE fixture"),),
    )

    extension = MODULE.validate(dist, "0.4.9", tmp_path / "unpacked")

    assert extension.name == "_eda.cp311-win_amd64.pyd"
    assert extension.read_bytes() == b"PE fixture"


@pytest.mark.parametrize(
    "payload",
    [
        (("labneura/_eda.cp311-win_amd64.pyd", b"PE"), ("bin/slang.dll", b"")),
        (("labneura/_eda.cp311-win_amd64.pyd", b"PE"), ("bin/helper.dll", b"")),
    ],
)
def test_rejects_slang_or_bundled_dlls(tmp_path, payload):
    dist = tmp_path / "dist"
    dist.mkdir()
    _wheel(dist / "labneura_core-0.4.9-py3-none-any.whl", "labneura-core")
    _wheel(
        dist / "labneura_eda-0.4.9-cp311-cp311-win_amd64.whl",
        "labneura-eda",
        requires=("labneura-core>=0.5,<0.6",),
        payload=payload,
    )

    with pytest.raises(AssertionError):
        MODULE.validate(dist, "0.4.9", tmp_path / "unpacked")
