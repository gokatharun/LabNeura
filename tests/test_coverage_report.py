import json
from pathlib import Path


def test_native_coverage_schema_is_versioned_and_excludes_nonproduction_sources():
    source = Path(__file__).resolve().parents[1] / "scripts" / "coverage_report.py"
    text = source.read_text(encoding="utf-8")
    assert '"schema_version": 1' in text
    for excluded in ("tests", "build", "generated", "third_party"):
        assert excluded in text


def test_percent_contract_is_explicit():
    namespace = {}
    source = Path(__file__).resolve().parents[1] / "scripts" / "coverage_report.py"
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
    assert namespace["percent"]({"covered": 3, "count": 4}) == "75.00%"
    assert namespace["percent"]({"covered": 0, "count": 0}) == "N/A"
