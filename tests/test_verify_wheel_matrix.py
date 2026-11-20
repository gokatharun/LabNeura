from email import message_from_string

import pytest

from scripts.verify_wheel_matrix import (
    check_wheel_tags,
    check_wheel_readme,
    normalized_dependencies,
    otool_load_information,
)


def test_dependency_bound_order_is_ignored():
    expected = ("labneura-core>=0.5,<0.6",)
    wheel_metadata = ("labneura-core<0.6,>=0.5",)

    assert normalized_dependencies(wheel_metadata) == normalized_dependencies(expected)


def test_dependency_name_and_bounds_are_still_checked():
    expected = ("labneura-core>=0.5,<0.6",)

    assert normalized_dependencies(("labneura-core>=0.5,<0.7",)) != normalized_dependencies(expected)
    assert normalized_dependencies(("labneura-tensor>=0.5,<0.6",)) != normalized_dependencies(expected)


def test_core_wheel_accepts_pure_python_tag():
    check_wheel_tags("labneura-core", ["py3-none-any"], "core.whl")


def test_metadata_distribution_accepts_pure_python_tag():
    check_wheel_tags("labneura-all", ["py3-none-any"], "all.whl")


def test_core_wheel_rejects_platform_tag():
    with pytest.raises(AssertionError, match="pure-Python wheel has platform tags"):
        check_wheel_tags("labneura-core", ["cp311-cp311-macosx_14_0_arm64"], "core.whl")


def test_native_component_rejects_pure_python_tag():
    with pytest.raises(AssertionError, match="native wheel has pure-Python tag"):
        check_wheel_tags("labneura-eda", ["py3-none-any"], "eda.whl")


def test_native_component_accepts_platform_tag():
    check_wheel_tags("labneura-tensor", ["cp311-cp311-macosx_14_0_arm64"], "tensor.whl")


def test_wheel_requires_nonempty_markdown_description():
    metadata = message_from_string(
        "Name: labneura-core\n"
        "Description-Content-Type: text/markdown; charset=UTF-8\n\n"
        "# LabNeura Core\n"
    )

    check_wheel_readme(metadata, "core.whl")


@pytest.mark.parametrize(
    "headers",
    (
        "Name: labneura-core\n\n",
        "Name: labneura-core\nDescription-Content-Type: text/plain\n\nDescription\n",
        "Name: labneura-core\nDescription-Content-Type: text/markdown\n\n   \n",
    ),
)
def test_wheel_rejects_missing_or_non_markdown_description(headers):
    with pytest.raises(AssertionError, match="missing non-empty Markdown"):
        check_wheel_readme(message_from_string(headers), "core.whl")


def test_otool_output_header_is_not_mistaken_for_a_runtime_path():
    output = (
        "/Users/runner/work/_temp/native.so:\n"
        "Load command 0\n\tcmd LC_RPATH\n\tpath @loader_path\n"
    )

    body = otool_load_information(output)

    assert "/Users/runner/work" not in body
    assert "path @loader_path" in body


def test_otool_load_path_remains_visible_for_leak_detection():
    output = (
        "/tmp/native.so:\n"
        "Load command 0\n\tcmd LC_RPATH\n"
        "\tpath /Users/runner/work/LabNeura (offset 12)\n"
    )

    assert "/Users/runner/work/LabNeura" in otool_load_information(output)
