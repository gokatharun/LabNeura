import pytest

import labneura


def test_execution_thread_configuration():
    available = labneura.available_cpu_count()
    assert available >= 1
    assert labneura.execution_thread_count() >= 1

    original = labneura.execution_thread_count()
    try:
        labneura.set_execution_thread_count(1)
        assert labneura.execution_thread_count() == 1

        with pytest.raises(ValueError):
            labneura.set_execution_thread_count(0)
        with pytest.raises(ValueError):
            labneura.set_execution_thread_count(available + 1)

        labneura.set_execution_thread_count_auto()
        assert labneura.execution_thread_count() == available
    finally:
        labneura.set_execution_thread_count(original)
