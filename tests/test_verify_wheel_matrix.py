import pytest

from scripts.verify_wheel_matrix import check_wheel_tags, normalized_dependencies


def test_dependency_bound_order_is_ignored():
    expected = ("labneura-core>=0.4.2,<0.5",)
    wheel_metadata = ("labneura-core<0.5,>=0.4.2",)

    assert normalized_dependencies(wheel_metadata) == normalized_dependencies(expected)


def test_dependency_name_and_bounds_are_still_checked():
    expected = ("labneura-core>=0.4.2,<0.5",)

    assert normalized_dependencies(("labneura-core>=0.4.2,<0.6",)) != normalized_dependencies(expected)
    assert normalized_dependencies(("labneura-tensor>=0.4.2,<0.5",)) != normalized_dependencies(expected)


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
