import os

import numpy as np
import pytest

from benchmarking.benchmark_runner import (
    BenchmarkRunner,
    _integer_result,
    _statistics,
    backend_override,
    generate_inputs,
    reference_result,
)


@pytest.mark.parametrize("dtype_name", ["FP32", "INT8", "FP16", "INT16"])
def test_input_generation_is_deterministic_and_divisors_are_nonzero(dtype_name):
    lhs, rhs = generate_inputs(dtype_name, 257, seed=41)
    lhs_again, rhs_again = generate_inputs(dtype_name, 257, seed=41)

    np.testing.assert_array_equal(lhs, lhs_again)
    np.testing.assert_array_equal(rhs, rhs_again)
    assert np.all(rhs != 0)


@pytest.mark.parametrize(
    ("dtype", "lhs", "rhs", "operation", "expected"),
    [
        (np.int8, [127, -128, 40, -40], [1, -1, 4, -4], "add",
         [127, -128, 44, -44]),
        (np.int8, [-128, 127, 40, -40], [1, -1, 4, -4], "sub",
         [-128, 127, 36, -36]),
        (np.int8, [127, -128, 20, -20], [2, -2, -10, -10], "mul",
         [127, 127, -128, 127]),
        (np.int8, [-127, 127, 7, -7], [2, -2, -3, 3], "div",
         [-63, -63, -2, -2]),
        (np.int16, [32767, -32768, 200, -200], [1, -1, 100, -100], "add",
         [32767, -32768, 300, -300]),
        (np.int16, [32767, -32768, 200, -200], [-1, 1, 100, -100], "sub",
         [32767, -32768, 100, -100]),
        (np.int16, [30000, -30000, 200, -200], [2, -2, -200, -200], "mul",
         [32767, 32767, -32768, 32767]),
        (np.int16, [-32768, 32767, 7, -7], [-1, -2, -3, 3], "div",
         [32767, -16383, -2, -2]),
    ],
)
def test_integer_reference_uses_saturating_arithmetic(dtype, lhs, rhs, operation, expected):
    actual = _integer_result(
        np.asarray(lhs, dtype=dtype),
        np.asarray(rhs, dtype=dtype),
        operation,
        np.dtype(dtype),
    )
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=dtype))


def test_reference_fp32_matches_independent_elementwise_result():
    lhs = np.array([1.5, -2.0], dtype=np.float32)
    rhs = np.array([2.0, 4.0], dtype=np.float32)

    np.testing.assert_array_equal(reference_result(lhs, rhs, "mul", "FP32"),
                                  np.array([3.0, -8.0], dtype=np.float32))


def test_backend_override_restores_set_and_unset_environment(monkeypatch):
    monkeypatch.setenv("LABNEURA_BACKEND", "AVX2")
    with pytest.raises(RuntimeError):
        with backend_override("GENERIC"):
            assert os.environ["LABNEURA_BACKEND"] == "GENERIC"
            raise RuntimeError("exercise cleanup")
    assert os.environ["LABNEURA_BACKEND"] == "AVX2"

    monkeypatch.delenv("LABNEURA_BACKEND")
    with pytest.raises(RuntimeError):
        with backend_override("GENERIC"):
            raise RuntimeError("exercise cleanup")
    assert "LABNEURA_BACKEND" not in os.environ


def test_statistics_reports_median_iqr_and_p95():
    result = _statistics([1e-6, 2e-6, 3e-6, 4e-6, 5e-6])

    assert result["median_us"] == pytest.approx(3.0)
    assert result["iqr_us"] == pytest.approx(2.0)
    assert result["p95_us"] == pytest.approx(4.8)


def test_speedup_comparisons_use_matching_backend_case():
    rows = [
        {"comparison_backend": "GENERIC", "dtype": "FP32", "operation": "ADD",
         "size": 10, "layer": "allocating", "library": "numpy",
         "correctness": "PASS", "median_us": 4.0},
        {"comparison_backend": "GENERIC", "dtype": "FP32", "operation": "ADD",
         "size": 10, "layer": "allocating", "library": "pytorch",
         "correctness": "PASS", "median_us": 8.0},
        {"comparison_backend": "GENERIC", "dtype": "FP32", "operation": "ADD",
         "size": 10, "layer": "allocating", "library": "labneura",
         "correctness": "PASS", "median_us": 2.0},
    ]

    BenchmarkRunner._add_speedups(rows)
    labneura_row = rows[-1]
    assert labneura_row["speedup_vs_numpy"] == pytest.approx(2.0)
    assert labneura_row["speedup_vs_pytorch"] == pytest.approx(4.0)
