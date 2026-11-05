#!/usr/bin/env python3
"""Compare Generic and AVX2 INT8 in-place multiplication on Linux x86_64."""

import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import labneura as ln


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "benchmarking" / "avx2_int8_comparison.md"
BENCHMARK_SIZES = [32, 256, 4_096, 65_536, 1_048_576]
CORRECTNESS_SIZES = [1, 8, 31, 32, 33, 63, 64, 65, 96, 97, 256]
WARMUP_ITERATIONS = 5
SAMPLE_COUNT = 7


def with_backend(name, lhs, rhs):
    previous = os.environ.get("LABNEURA_BACKEND")
    os.environ["LABNEURA_BACKEND"] = name
    try:
        left = ln.Tensor(lhs, ln.QuantizationMode.INT8)
        right = ln.Tensor(rhs, ln.QuantizationMode.INT8)
    finally:
        if previous is None:
            os.environ.pop("LABNEURA_BACKEND", None)
        else:
            os.environ["LABNEURA_BACKEND"] = previous
    return left, right


def verify_correctness():
    cases = (
        lambda i: ((i % 32) + 1, 2),
        lambda i: ((i % 32) - 16, (1, 2, -1, -2)[i % 4]),
        lambda i: ((127, -128, 64, -64, 0)[i % 5], (2, 2, 2, 2, -128)[i % 5]),
    )

    for size in CORRECTNESS_SIZES:
        for case in cases:
            lhs, rhs = zip(*(case(i) for i in range(size)))
            expected = [max(-128, min(127, a * b)) for a, b in zip(lhs, rhs)]
            for backend in ("GENERIC", "AVX2"):
                left, right = with_backend(backend, list(lhs), list(rhs))
                left.mul_inplace(right)
                actual = left.data_int8()
                if actual != expected:
                    raise AssertionError(
                        f"{backend} incorrect for size {size}: "
                        f"expected {expected}, got {actual}"
                    )


def benchmark_backend(backend, size):
    lhs = [(i % 127) - 63 for i in range(size)]
    rhs = [1] * size
    left, right = with_backend(backend, lhs, rhs)
    iterations = max(50, min(5_000, 5_000_000 // size))

    for _ in range(WARMUP_ITERATIONS):
        left.mul_inplace(right)

    sample_times = []
    for _ in range(SAMPLE_COUNT):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            left.mul_inplace(right)
        sample_times.append((time.perf_counter_ns() - start) / iterations)

    if left.data_int8() != lhs:
        raise AssertionError(f"{backend} benchmark changed its input values")
    return statistics.median(sample_times), iterations


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "Unavailable"


def current_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main():
    if platform.system() != "Linux" or platform.machine().lower() not in {
        "x86_64",
        "amd64",
    }:
        raise RuntimeError("This comparison must run on Linux x86_64")
    if ln.detect_backend() != "AVX2":
        raise RuntimeError(
            "AVX2 backend is unavailable; refusing to run AVX2 benchmark"
        )

    verify_correctness()

    rows = []
    for size in BENCHMARK_SIZES:
        generic_ns, iterations = benchmark_backend("GENERIC", size)
        avx2_ns, _ = benchmark_backend("AVX2", size)
        rows.append(
            (
                size,
                iterations,
                generic_ns,
                avx2_ns,
                generic_ns / avx2_ns,
                size * 1_000_000_000 / generic_ns,
                size * 1_000_000_000 / avx2_ns,
            )
        )

    report_lines = [
        "# Linux x86_64 INT8 Multiplication: Generic vs AVX2",
        "",
        f"- Run time (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Commit: `{current_commit()}`",
        f"- OS / architecture: {platform.platform()} / {platform.machine()}",
        f"- CPU: {cpu_model()}",
        f"- Runtime-selected backend: {ln.detect_backend()}",
        "- Operation: in-place INT8 elementwise multiply with multiplier values of 1",
        "- Timing includes Python binding call overhead; tensors are allocated before timing.",
        f"- Samples: {SAMPLE_COUNT} per backend and size; median reported; "
        f"{WARMUP_ITERATIONS} warmups per backend.",
        "- Correctness was checked against independently clamped expected values "
        f"and Generic results for sizes {', '.join(map(str, CORRECTNESS_SIZES))}.",
        "",
        "| Elements | Iterations/sample | Generic median (ns/op) | AVX2 median (ns/op) | AVX2 speedup | Generic M elements/s | AVX2 M elements/s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for size, iterations, generic_ns, avx2_ns, speedup, generic_rate, avx2_rate in rows:
        report_lines.append(
            f"| {size:,} | {iterations:,} | {generic_ns:,.1f} | "
            f"{avx2_ns:,.1f} | {speedup:.2f}x | "
            f"{generic_rate / 1_000_000:,.2f} | {avx2_rate / 1_000_000:,.2f} |"
        )

    report_lines.extend(
        [
            "",
            "Speedup is Generic median latency divided by AVX2 median latency; "
            "values above 1 indicate AVX2 was faster. Results are specific to "
            "this GitHub-hosted runner and include Python-call overhead, so they "
            "are not a hardware-independent performance guarantee.",
            "",
        ]
    )
    REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
