from scripts.verify_wheel_matrix import normalized_dependencies


def test_dependency_bound_order_is_ignored():
    expected = ("labneura-core>=0.4.2,<0.5",)
    wheel_metadata = ("labneura-core<0.5,>=0.4.2",)

    assert normalized_dependencies(wheel_metadata) == normalized_dependencies(expected)


def test_dependency_name_and_bounds_are_still_checked():
    expected = ("labneura-core>=0.4.2,<0.5",)

    assert normalized_dependencies(("labneura-core>=0.4.2,<0.6",)) != normalized_dependencies(expected)
    assert normalized_dependencies(("labneura-tensor>=0.4.2,<0.5",)) != normalized_dependencies(expected)
