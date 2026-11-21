from pathlib import Path

from scripts.check_architecture import (
    check_dependency_boundaries,
    check_duplicate_trees,
    check_isa_isolation,
    validate,
)

ROOT = Path(__file__).resolve().parents[2]


def write(root: Path, relative: str, contents: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def test_repository_component_architecture_is_valid():
    assert validate(ROOT) == []


def test_eda_tensor_dependency_is_rejected(tmp_path):
    write(tmp_path, "src/labneura/eda/parser.cpp", '#include <labneura/tensor/tensor.h>\n')
    assert any(item.rule == "component-dependency" for item in check_dependency_boundaries(tmp_path))


def test_python_native_mirror_is_rejected(tmp_path):
    write(tmp_path, "python/src/labneura/tensor.cpp", "// duplicate\n")
    assert any(item.rule == "duplicate-native-tree" for item in check_duplicate_trees(tmp_path))


def test_target_wide_avx2_is_rejected(tmp_path):
    write(tmp_path, "CMakeLists.txt", "target_compile_options(portable PRIVATE -mavx2)\n")
    assert any(item.rule == "portable-isa-flags" for item in check_isa_isolation(tmp_path))


def test_per_source_avx2_property_is_allowed(tmp_path):
    write(
        tmp_path,
        "CMakeLists.txt",
        "set_source_files_properties(src/labneura/tensor/core/kernels/cpu/avx2/op.cpp "
        "PROPERTIES COMPILE_OPTIONS -mavx2)\n",
    )
    assert check_isa_isolation(tmp_path) == []


def test_f16c_kernel_uses_the_cmake_feature_contract():
    component = (ROOT / "cmake/components/tensor/CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    implementation = (
        ROOT
        / "src/labneura/tensor/core/kernels/cpu/avx2/avx2_ops_fp16_f16c.cpp"
    ).read_text(encoding="utf-8")
    caller = (
        ROOT / "src/labneura/tensor/core/kernels/cpu/avx2/avx2_ops_fp16.cpp"
    ).read_text(encoding="utf-8")
    declaration = (
        ROOT / "src/labneura/tensor/core/kernels/cpu/avx2/avx2_fp16_f16c.h"
    ).read_text(encoding="utf-8")

    assert (
        "target_sources(labneura_tensor_objects PRIVATE "
        "${LABNEURA_TENSOR_F16C_SOURCE})" in component
    )
    assert "PRIVATE LABNEURA_ENABLE_AVX2_F16C=1" in component
    assert implementation.startswith("#if defined(LABNEURA_ENABLE_AVX2_F16C)")
    assert "#if defined(LABNEURA_ENABLE_AVX2_F16C)" in caller
    assert "#if defined(LABNEURA_ENABLE_AVX2_F16C)" in declaration
