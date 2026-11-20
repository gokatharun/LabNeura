import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lexer_benchmark_report.py"


def sample_log(platform: str, optimized: str) -> str:
    return f"""sources=3 input_bytes=1000 corpus_tokens=200 warmups=2 repetitions=9 hardware_threads=8
case=lexer_end_to_end backend=scalar input_bytes=1000 corpus_tokens=200 median_seconds=0.001 iqr_seconds=0.0001 p95_seconds=0.0012 median_MB_per_second=1.0 tokens_per_pass=200 retained_token_and_error_capacity_bytes=4096
case=lexer_end_to_end backend={optimized} input_bytes=1000 corpus_tokens=200 median_seconds=0.0005 iqr_seconds=0.0001 p95_seconds=0.0007 median_MB_per_second=2.0 tokens_per_pass=200 retained_token_and_error_capacity_bytes=4096
case=lexer_end_to_end backend=slang input_bytes=1000 corpus_tokens=210 median_seconds=0.002 iqr_seconds=0.0002 p95_seconds=0.0022 median_MB_per_second=0.5 tokens_per_pass=210 retained_token_and_error_capacity_bytes=unavailable slang_version=11.0.0+0
architecture={platform} cpu_model=Example CPU Model
os=Linux
build_type=Release
compiler=gcc 14.2
"""


def test_parse_and_aggregate_lexer_reports(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    for platform, optimized in (("linux-x86_64", "avx2"), ("linux-arm64", "neon")):
        log = tmp_path / f"{platform}.log"
        output = reports / f"lexer-benchmark-{platform}.json"
        log.write_text(sample_log(platform, optimized), encoding="utf-8")
        subprocess.run(
            [sys.executable, str(SCRIPT), "parse", "--input", str(log),
             "--output", str(output), "--platform", platform],
            check=True,
        )
        parsed = json.loads(output.read_text(encoding="utf-8"))
        assert parsed["correctness"] == "passed"
        assert parsed["metadata"]["cpu_model"] == "Example CPU Model"
        assert len(parsed["measurements"]) == 3
        optimized_row = next(
            row for row in parsed["measurements"] if row["backend"] == optimized
        )
        assert optimized_row["speedup_vs_scalar"] == 2.0
        assert optimized_row["ratio_vs_slang"] == 4.0

    summary_json = tmp_path / "summary.json"
    summary_markdown = tmp_path / "summary.md"
    subprocess.run(
        [sys.executable, str(SCRIPT), "aggregate", "--input-dir", str(reports),
         "--json", str(summary_json), "--markdown", str(summary_markdown)],
        check=True,
    )
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert len(summary["platforms"]) == 2
    assert all(row["best_native_over_slang"] == 4.0 for row in summary["platforms"])
    markdown = summary_markdown.read_text(encoding="utf-8")
    assert "Linux" not in markdown
    assert "linux-x86_64" in markdown
    assert "4.00x" in markdown
