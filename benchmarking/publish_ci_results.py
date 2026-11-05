#!/usr/bin/env python3
"""Turn native backend benchmark CSVs into a commit-specific Markdown report."""

import argparse
import csv
from pathlib import Path


def load_results(
    directory: Path,
) -> dict[tuple[str, str, str, int], dict[str, tuple[float, float]]]:
    results: dict[
        tuple[str, str, str, int], dict[str, tuple[float, float]]
    ] = {}
    for path in sorted(directory.glob("benchmark-*.csv")):
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                key = (
                    row["architecture"],
                    row["dtype"],
                    row["operation"],
                    int(row["size"]),
                )
                results.setdefault(key, {})[row["backend"]] = (
                    float(row["median_ns"]),
                    float(row["p95_ns"]),
                )
    if not results:
        raise SystemExit(f"No benchmark CSV files found in {directory}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run-url", required=True)
    args = parser.parse_args()

    results = load_results(args.results_dir)
    architectures = sorted({key[0] for key in results})
    lines = [
        "# Native CPU Backend Benchmark",
        "",
        f"- Commit: [`{args.commit[:12]}`](https://github.com/gokatharun/LabNeura/commit/{args.commit})",
        f"- GitHub Actions run: {args.run_url}",
        "- Measurements: median and p95 native in-place operation latency; tensor allocation and input initialization are outside the timed interval.",
        "- Correctness: each backend is checked against independent scalar expected values before timing.",
        "- Scope: FP32 and saturating INT8 ADD, SUB, MUL, and DIV. Division inputs are nonzero.",
        "- Runner hardware varies between GitHub-hosted executions; treat these as per-run observations, not stable performance guarantees.",
        "",
    ]
    for architecture in architectures:
        vector_backend = "AVX2" if architecture == "x86_64" else "NEON"
        lines.extend(
            [
                f"## {architecture}: Generic vs {vector_backend}",
                "",
                "| Dtype | Operation | Elements | Generic median (ns) | Generic p95 (ns) | "
                "SIMD median (ns) | SIMD p95 (ns) | Speedup |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        arch_results = sorted(
            (
                (key, values)
                for key, values in results.items()
                if key[0] == architecture
            ),
            key=lambda item: (item[0][1], item[0][2], item[0][3]),
        )
        for (_, dtype, operation, size), backends in arch_results:
            generic = backends.get("GENERIC")
            simd = backends.get(vector_backend)
            if generic is None or simd is None:
                raise SystemExit(
                    f"Missing Generic or {vector_backend} result for "
                    f"{architecture}/{dtype}/{operation}/{size}"
                )
            generic_median, generic_p95 = generic
            simd_median, simd_p95 = simd
            speedup = generic_median / simd_median if simd_median else float("inf")
            lines.append(
                f"| {dtype} | {operation} | {size:,} | {generic_median:.2f} | "
                f"{generic_p95:.2f} | {simd_median:.2f} | {simd_p95:.2f} | "
                f"{speedup:.2f}x |"
            )
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
