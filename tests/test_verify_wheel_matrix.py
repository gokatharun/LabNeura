from scripts.verify_wheel_matrix import DISTRIBUTIONS, EXPECTED_DEPENDENCIES, dependencies


def test_exactly_three_release_distributions():
    assert DISTRIBUTIONS == ("labneura", "labneura-tensor", "labneura-eda")


def test_native_components_are_independently_installable():
    assert EXPECTED_DEPENDENCIES["labneura-tensor"] == ()
    assert EXPECTED_DEPENDENCIES["labneura-eda"] == ()


def test_umbrella_owns_both_native_dependencies():
    assert dependencies(EXPECTED_DEPENDENCIES["labneura"]) == dependencies(
        ("labneura-tensor<0.6,>=0.5", "labneura-eda>=0.5,<0.6")
    )
