import pytest

from benchmarks.eda.aggregate_frontend_results import (
    ComparisonError,
    compare,
    markdown_report,
    ranked_phase_contributions,
)


def row(frontend, *, digest="abc", correctness="passed", median=0.01, nets=4):
    return {
        "frontend": frontend,
        "category": "syntax",
        "execution_backend": "scalar",
        "workload_identity": "boolean-10k",
        "workload_content_hash": digest,
        "correctness_verification_status": correctness,
        "median_seconds": median,
        "port_count": 3,
        "net_count": nets,
        "instance_count": 2,
        "cell_types": ["AND2"],
        "metrics": {"peak_rss_bytes": None},
    }


def test_comparison_requires_hash_category_and_accepted_correctness():
    with pytest.raises(ComparisonError, match="missing slang"):
        compare([row("native"), row("slang", digest="different")])

    with pytest.raises(ComparisonError, match="unaccepted correctness"):
        compare([row("native"), row("slang", correctness="unavailable")])


def test_nullable_metrics_and_structural_differences_are_preserved():
    result = compare([row("native", median=None), row("slang", nets=5)])[0]

    assert result["native_speedup_ratio"] is None
    assert result["native_metrics"]["peak_rss_bytes"] is None
    assert result["structurally_equal"] is False
    assert result["native_structure"]["net_count"] == 4
    assert result["slang_structure"]["net_count"] == 5


def test_report_calculates_ratio_and_only_ranks_measured_phases():
    rows = [row("native", median=0.005), row("slang", median=0.02)]
    rows[0]["phase_timings_seconds"] = {"parse": 0.003, "elaborate": 0.001}
    rows[1]["phase_timings_seconds"] = {"parse": 0.012, "deferred": None}
    comparisons = compare(rows)
    phases = ranked_phase_contributions(rows)
    report = markdown_report(comparisons, phases)

    assert comparisons[0]["native_speedup_ratio"] == pytest.approx(4.0)
    assert [phase["phase"] for phase in phases] == ["parse", "parse", "elaborate"]
    assert all(phase["evidence"] == "measured" for phase in phases)
    assert "4x" in report
    assert "different (see JSON)" not in report
    assert "deferred" not in report


def test_report_does_not_invent_hotspots_when_phase_data_is_absent():
    comparisons = compare([row("native"), row("slang")])
    report = markdown_report(comparisons, [])
    assert "no hotspot ranking is claimed" in report
