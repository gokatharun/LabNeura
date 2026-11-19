from pathlib import Path

from benchmarks.eda.profile_frontends import (
    PERF_EVENTS,
    parse_peak_rss,
    parse_perf_stat,
    peak_rss_command,
    perf_stat_command,
    sample_command,
    xctrace_command,
)


def test_platform_commands_keep_profiling_separate_from_benchmark(tmp_path):
    benchmark = ["frontend-benchmark", "--workload", "basic.sv"]
    assert peak_rss_command("Darwin", benchmark) == ["/usr/bin/time", "-l", *benchmark]
    assert peak_rss_command("Linux", benchmark) == ["/usr/bin/time", "-v", *benchmark]
    assert peak_rss_command("Plan9", benchmark) is None

    perf = perf_stat_command(benchmark, tmp_path / "stat.csv")
    assert perf[:3] == ["perf", "stat", "-x"]
    assert perf[-len(benchmark):] == benchmark
    assert set(perf[perf.index("-e") + 1].split(",")) == set(PERF_EVENTS)

    trace = xctrace_command(benchmark, tmp_path / "result.trace")
    assert trace[:4] == ["xcrun", "xctrace", "record", "--template"]
    assert trace[-len(benchmark):] == benchmark
    allocations = xctrace_command(
        benchmark, tmp_path / "allocations.trace", "Allocations")
    assert allocations[allocations.index("--template") + 1] == "Allocations"
    assert sample_command(123, tmp_path / "sample.txt") == [
        "sample", "123", "1", "-file", str(tmp_path / "sample.txt")]


def test_perf_parser_preserves_unavailable_counters_as_null():
    parsed = parse_perf_stat(
        "12345,,cycles,100.00,100.00\n"
        "<not supported>,,cache-misses,0.00,0.00\n"
        "2.5,msec,task-clock,100.00,100.00\n")

    assert parsed["cycles"] == {"status": "available", "value": 12345, "unit": "count"}
    assert parsed["task-clock"] == {
        "status": "available", "value": 2.5, "unit": "msec"}
    assert parsed["cache-misses"]["status"] == "unavailable"
    assert parsed["cache-misses"]["value"] is None
    assert parsed["instructions"]["value"] is None


def test_peak_rss_parser_handles_macos_linux_and_missing_values():
    assert parse_peak_rss("Darwin", "  987654  maximum resident set size\n") == {
        "status": "available", "value": 987654, "unit": "bytes"}
    assert parse_peak_rss(
        "Linux", "Maximum resident set size (kbytes): 4321\n") == {
            "status": "available", "value": 4321, "unit": "KiB"}

    missing = parse_peak_rss("Linux", "no measurement")
    assert missing["status"] == "unavailable"
    assert missing["value"] is None


def test_source_contains_no_zero_substitute_for_unavailable_metrics():
    source = Path("benchmarks/eda/profile_frontends.py").read_text()
    assert '"status": "unavailable", "value": None' in source
