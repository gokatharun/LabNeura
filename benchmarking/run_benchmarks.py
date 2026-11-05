#!/usr/bin/env python3
"""Run the Python API comparison and write a unique, self-describing report."""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarking.benchmark_runner import (  # noqa: E402
    REQUIRED_SIZES,
    BenchmarkRunner,
    collect_metadata,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Run a quick correctness and timing smoke test")
    parser.add_argument("--sizes", type=str,
                        help="Comma-separated tensor sizes (defaults to the required suite)")
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--threads", type=int, default=1,
                        help="NumPy/PyTorch CPU threads; default is one")
    parser.add_argument("--cmake-build-dir", type=Path,
                        help="CMake build directory used to build the installed extension")
    parser.add_argument("--output-dir", type=Path,
                        help="New output directory; existing directories are never overwritten")
    return parser.parse_args()


def _prepare_output_dir(requested: Optional[Path], commit: str) -> Path:
    if requested is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        requested = ROOT / "benchmarking" / "results" / f"python-{stamp}-{commit}"
    requested = requested.resolve()
    requested.mkdir(parents=True, exist_ok=False)
    return requested


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _plot_results(output_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plot generation requires the benchmark dependencies; install "
            "`pip install -r benchmarking/requirements.txt`"
        ) from error

    plots_dir = output_dir / "plots"
    plots_dir.mkdir()
    comparisons = set(row["comparison_backend"] for row in rows)
    simd_backend = next(name for name in comparisons if name != "GENERIC")

    for dtype in ("FP32", "INT8", "FP16", "INT16"):
        for operation in ("ADD", "SUB", "MUL", "DIV"):
            figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
            for axis, layer in zip(axes, ("allocating", "inplace")):
                selected_rows = [
                    row for row in rows
                    if row["dtype"] == dtype
                    and row["operation"] == operation
                    and row["layer"] == layer
                    and row["correctness"] == "PASS"
                    and (
                        (row["library"] == "labneura")
                        or row["comparison_backend"] == simd_backend
                    )
                ]
                labels = {
                    ("labneura", "GENERIC"): "LabNeura Generic",
                    ("labneura", simd_backend): f"LabNeura {simd_backend}",
                    ("numpy", simd_backend): "NumPy",
                    ("pytorch", simd_backend): "PyTorch CPU",
                }
                for (library, backend), label in labels.items():
                    points = sorted(
                        (row["size"], row["median_us"])
                        for row in selected_rows
                        if row["library"] == library
                        and row["comparison_backend"] == backend
                    )
                    if points:
                        axis.plot(
                            [point[0] for point in points],
                            [point[1] for point in points],
                            marker="o",
                            label=label,
                        )
                axis.set_xscale("log")
                axis.set_yscale("log")
                axis.set_title(layer.replace("_", " ").title())
                axis.set_xlabel("Elements")
                axis.grid(True, which="both", alpha=0.25)
            axes[0].set_ylabel("Median latency (µs)")
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                figure.legend(handles, labels, loc="upper center", ncol=4)
            figure.suptitle(f"{dtype} {operation}: Python/API latency")
            figure.tight_layout(rect=(0, 0, 1, 0.88))
            figure.savefig(plots_dir / f"{dtype.lower()}_{operation.lower()}.png", dpi=150)
            plt.close(figure)


def _write_report(path: Path, metadata: Dict[str, Any], rows: Sequence[Dict[str, Any]],
                  sizes: Sequence[int], simd_backend: str) -> None:
    passed = sum(row["correctness"] == "PASS" for row in rows)
    unsupported = sum(row["correctness"] == "UNSUPPORTED" for row in rows)
    lines = [
        "# LabNeura Python/API Benchmark",
        "",
        f"- Timestamp (UTC): {metadata['timestamp_utc']}",
        f"- Commit: `{metadata['git_commit']}`",
        f"- CPU: {metadata['cpu_model']} ({metadata['architecture']})",
        f"- OS: {metadata['os']}",
        f"- Python / NumPy / PyTorch: {metadata['python']} / {metadata['numpy']} / "
        f"{metadata['pytorch']}",
        f"- LabNeura: {metadata['labneura']}",
        f"- Actual SIMD backend selected: **{simd_backend}**",
        f"- Correctness-checked results: {passed} passed; {unsupported} explicitly unsupported",
        "",
        "## Method",
        "",
        f"Sizes: {', '.join(f'{size:,}' for size in sizes)} elements. Inputs are generated "
        "deterministically from the recorded seed, and correctness is checked before each "
        "timed case. The primary metric is the median of independent batch averages; IQR "
        "and p95 are also included in the CSV and JSON.",
        "",
        "Allocating timings include the operation and output allocation. In-place timings "
        "include only the API operation; input objects are prepared outside each timed "
        "batch and each receives one operation, so repeated timing does not mutate and "
        "reuse the same input. Input generation and dtype conversion are outside timing. "
        "Input-object construction is measured separately. These operation timings are "
        "end-to-end Python API measurements, not isolated native kernel timings.",
        "",
        "INT8 and INT16 use saturating add/subtract/multiply and truncating division in all "
        "libraries; division denominators are nonzero. FP16 operands and results use "
        "16-bit floating-point values. Generic FP16/INT16 rows are explicitly unsupported "
        "because the Generic backend has no kernels for those dtypes. NumPy/PyTorch rows "
        "are measured once per comparison backend context for consistent tables.",
        "",
        "The repeated-input working set is cache-resident. This report does not claim "
        "streaming/DRAM bandwidth performance. NumPy and PyTorch run on CPU with one "
        "thread. See `benchmark.json` for the full metadata and `benchmark.csv` for every "
        "case.",
        "",
        "The native C++ benchmark is a separate layer and remains documented in the "
        "[benchmark README](../../README.md); native timings must not be compared directly "
        "with the Python/API results here.",
        "",
        "## Sample comparison points",
        "",
        "| Dtype | Op | Layer | Size | Backend | LabNeura (µs) | NumPy (µs) | PyTorch (µs) | "
        "Lab/NumPy | Lab/PyTorch |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for dtype in ("FP32", "INT8", "FP16", "INT16"):
        for operation in ("ADD", "SUB", "MUL", "DIV"):
            for layer in ("allocating", "inplace"):
                for size in (min(sizes), max(sizes)):
                    for backend in ("GENERIC", simd_backend):
                        lookup = {
                            row["library"]: row
                            for row in rows
                            if row["dtype"] == dtype
                            and row["operation"] == operation
                            and row["layer"] == layer
                            and row["size"] == size
                            and row["comparison_backend"] == backend
                        }
                        lab = lookup.get("labneura", {})
                        numpy = lookup.get("numpy", {})
                        pytorch = lookup.get("pytorch", {})

                        def fmt(record: Dict[str, Any], field: str) -> str:
                            value = record.get(field)
                            return f"{value:.3f}" if isinstance(value, (int, float)) else "—"

                        lines.append(
                            f"| {dtype} | {operation} | {layer} | {size:,} | {backend} | "
                            f"{fmt(lab, 'median_us')} | {fmt(numpy, 'median_us')} | "
                            f"{fmt(pytorch, 'median_us')} | "
                            f"{fmt(lab, 'speedup_vs_numpy')}x | "
                            f"{fmt(lab, 'speedup_vs_pytorch')}x |"
                        )
    lines.extend(["", "## Visualization", ""])
    lines.extend(
        f"- `{dtype.lower()}_{operation.lower()}.png`"
        for dtype in ("FP32", "INT8", "FP16", "INT16")
        for operation in ("ADD", "SUB", "MUL", "DIV")
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _arguments()
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    if args.sizes:
        try:
            sizes = [int(value) for value in args.sizes.split(",")]
        except ValueError as error:
            raise SystemExit("--sizes must be comma-separated positive integers") from error
        if not sizes or any(size <= 0 for size in sizes):
            raise SystemExit("--sizes must contain positive integers")
    elif args.smoke:
        sizes = [10, 64, 1024]
    else:
        sizes = REQUIRED_SIZES

    batches = 3 if args.smoke else args.batches
    iterations = 5 if args.smoke else args.iterations
    try:
        import torch
    except ImportError as error:
        raise SystemExit(
            "PyTorch is required for the requested comparison. Install the optional "
            "dependencies with `pip install -r benchmarking/requirements.txt`; no library will be "
            "silently omitted."
        ) from error
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError as error:
        raise SystemExit(
            "threadpoolctl is required to record/control CPU thread counts; install "
            "`pip install -r benchmarking/requirements.txt`."
        ) from error

    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        thread_info = threadpool_info()
        runner = BenchmarkRunner(
            torch=torch, sizes=sizes, batches=batches,
            iterations=iterations, seed=args.seed, threads=args.threads,
        )
        metadata = collect_metadata(torch, thread_info, args.cmake_build_dir)
        metadata["pytorch_threads"] = torch.get_num_threads()
        metadata["seed"] = args.seed
        metadata["batches"] = batches
        metadata["iterations_per_batch"] = iterations
        metadata["simd_backend"] = runner.simd_backend
        output_dir = _prepare_output_dir(args.output_dir, metadata["git_commit"])
        rows = runner.run()

        payload = {
            "metadata": metadata,
            "methodology": {
                "dtypes": ["FP32", "INT8", "FP16", "INT16"],
                "operations": ["ADD", "SUB", "MUL", "DIV"],
                "layers": ["construction", "allocating", "inplace"],
                "sizes": sizes,
                "correctness_checked_before_timing": True,
                "working_set": "cache-resident repeated deterministic inputs",
                "native_benchmark_included": False,
            },
            "results": rows,
        }
        (output_dir / "benchmark.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        _write_csv(output_dir / "benchmark.csv", rows)
        _write_report(output_dir / "report.md", metadata, rows, sizes, runner.simd_backend)
        _plot_results(output_dir, rows)

    print(f"Selected SIMD backend: {runner.simd_backend}")
    print(f"Results: {output_dir}")
    print(f"CSV: {output_dir / 'benchmark.csv'}")
    print(f"JSON: {output_dir / 'benchmark.json'}")
    print(f"Report: {output_dir / 'report.md'}")
    print(f"Plots: {output_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
