#!/usr/bin/env python3
"""Benchmark LabNeura FP32 ADD thread scaling in hot and rotating regimes."""

import argparse
import csv
import gc
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import labneura  # noqa: E402
from benchmarking.benchmark_runner import (  # noqa: E402
    _cpu_model,
    _compiler_metadata,
    _statistics,
    backend_override,
)

DEFAULT_SIZES = (1024, 16384, 65536, 262144, 1000000)


class Case:
    def __init__(self, name: str, backend: str, layer: str,
                 pairs: List, is_numpy: bool = False):
        self.name = name
        self.backend = backend
        self.layer = layer
        self.pairs = pairs
        self.is_numpy = is_numpy
        self.samples: List[float] = []

    def schedule(self, indices: List[int]) -> List:
        return [(None, *self.pairs[index]) for index in indices]

    def run(self, schedule: List) -> None:
        for _, left, right in schedule:
            if self.is_numpy:
                np.add(left, right, out=left if self.layer == "inplace" else None)
            elif self.layer == "inplace":
                left.add_inplace(right)
            else:
                left.add(right)

    def read(self, tensor: Any) -> np.ndarray:
        if self.is_numpy:
            return tensor
        return np.asarray(tensor.data_fp32(), dtype=np.float32)


def _make_operands(size: int, set_index: int, seed: int):
    rng = np.random.default_rng([seed, size, set_index])
    left = (np.float32(1.0) + rng.random(size, dtype=np.float32)).astype(np.float32)
    right = (np.float32(5e-4) +
             np.float32(1e-3) * rng.random(size, dtype=np.float32))
    return left, right.astype(np.float32)


def _pool_size(regime: str, size: int, working_set_mb: int,
               max_sets: int) -> int:
    if regime == "hot":
        return 1
    pair_bytes = 2 * size * 4
    return int(min(max_sets, max(2, -(-working_set_mb * 1024 * 1024 // pair_bytes))))


def _build_cases(layer: str, arrays: List, simd: str) -> List[Case]:
    cases = []
    method = "add_inplace" if layer == "inplace" else "add"
    for backend in ("GENERIC", simd):
        with backend_override(backend):
            if labneura.detect_backend() != backend:
                raise RuntimeError(f"Requested {backend}, got {labneura.detect_backend()}")
            pairs = [
                (labneura.Tensor(left.tolist(), labneura.QuantizationMode.FP32),
                 labneura.Tensor(right.tolist(), labneura.QuantizationMode.FP32))
                for left, right in arrays
            ]
        cases.append(Case(f"labneura-{backend}", backend, layer, pairs))
    numpy_pairs = [(left.copy(), right.copy()) for left, right in arrays]
    cases.append(Case("numpy", "CPU", layer, numpy_pairs, is_numpy=True))
    return cases


def _verify_case(case: Case, arrays: List) -> None:
    for index in {0, len(arrays) - 1}:
        np.testing.assert_array_equal(case.read(case.pairs[index][0]),
                                      arrays[index][0])
        np.testing.assert_array_equal(case.read(case.pairs[index][1]),
                                      arrays[index][1])
    left, right = arrays[0]
    expected = np.add(left, right)
    if case.is_numpy:
        actual_left, actual_right = left.copy(), right.copy()
    else:
        with backend_override(case.backend):
            actual_left = labneura.Tensor(left.tolist(), labneura.QuantizationMode.FP32)
            actual_right = labneura.Tensor(right.tolist(), labneura.QuantizationMode.FP32)
    if case.layer == "inplace":
        if case.is_numpy:
            np.add(actual_left, actual_right, out=actual_left)
            actual = actual_left
        else:
            actual_left.add_inplace(actual_right)
            actual = case.read(actual_left)
    else:
        actual = (np.add(actual_left, actual_right) if case.is_numpy
                  else case.read(actual_left.add(actual_right)))
    np.testing.assert_array_equal(actual, expected)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--target-batch-ms", type=float, default=10.0)
    parser.add_argument("--min-iterations", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=4096)
    parser.add_argument("--working-set-mb", type=int, default=64)
    parser.add_argument("--max-sets", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--cmake-build-dir", type=Path,
                        default=ROOT / "build" / "local-python")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _run(case: Case, schedule: List) -> None:
    case.run(schedule)


def _thread_counts(available: int) -> List[int]:
    return sorted({count for count in (1, 2, 4, 6, available)
                   if 1 <= count <= available})


def run_combination(regime: str, layer: str, size: int, thread_count: int,
                    args: argparse.Namespace, simd: str, torch: Any
                    ) -> List[Dict[str, Any]]:
    sets = _pool_size(regime, size, args.working_set_mb, args.max_sets)
    arrays = [_make_operands(size, index, args.seed) for index in range(sets)]
    cases = _build_cases(layer, arrays, simd)
    labneura.set_execution_thread_count(thread_count)
    for case in cases:
        _verify_case(case, arrays)

    measured_cases = [
        case for case in cases if case.backend != "CPU" or thread_count == 1
    ]
    calibration_indices = [i % sets for i in range(max(64, 2 * sets))]
    per_call_seconds = []
    for case in measured_cases:
        schedule = case.schedule(calibration_indices)
        _run(case, schedule)
        start = time.perf_counter()
        _run(case, schedule)
        per_call_seconds.append((time.perf_counter() - start) / len(schedule))

    iterations = int(args.target_batch_ms * 1e-3 /
                     max(min(per_call_seconds), 1e-9))
    iterations = min(args.max_iterations, max(args.min_iterations, iterations))
    iterations = -(-iterations // sets) * sets
    indices = [i % sets for i in range(iterations)]
    schedules = [case.schedule(indices) for case in measured_cases]
    for case, schedule in zip(measured_cases, schedules):
        _run(case, schedule)

    for case in measured_cases:
        case.samples.clear()
    gc.collect()
    gc.disable()
    try:
        for batch in range(args.batches):
            shift = batch % len(measured_cases)
            for position in range(len(measured_cases)):
                index = (position + shift) % len(measured_cases)
                case = measured_cases[index]
                start = time.perf_counter()
                _run(case, schedules[index])
                case.samples.append((time.perf_counter() - start) / iterations)
    finally:
        gc.enable()

    rows = []
    for case in measured_cases:
        if layer == "inplace":
            for index in {0, sets - 1}:
                values = case.read(case.pairs[index][0])
                if not np.isfinite(values).all():
                    raise RuntimeError(f"{case.name}: in-place values became non-finite")
        row = {
            "regime": regime,
            "layer": layer,
            "operation": "ADD",
            "size": size,
            "threads": thread_count if case.backend != "CPU" else 1,
            "implementation": case.name,
            "backend": case.backend,
            "correctness": "PASS",
            "batches": args.batches,
            "iterations_per_batch": iterations,
            "operand_sets": sets,
            "working_set_bytes": sets * 2 * size * 4,
        }
        row.update(_statistics(case.samples))
        row["samples_us"] = [sample * 1e6 for sample in case.samples]
        rows.append(row)
    return rows


def write_report(path: Path, metadata: Dict[str, Any],
                 rows: List[Dict[str, Any]], sizes: List[int]) -> None:
    by_key = {(row["regime"], row["layer"], row["size"], row["threads"],
               row["implementation"]): row for row in rows}
    lines = [
        "# LabNeura FP32 ADD thread-scaling benchmark", "",
        f"- Timestamp (UTC): {metadata['timestamp_utc']}",
        f"- Commit: `{metadata['git_commit']}`; CPU: {metadata['cpu_model']}",
        f"- OS: {metadata['os']}; Python: {metadata['python']}; "
        f"NumPy: {metadata['numpy']}",
        f"- Compiler: {metadata['compiler']['compiler']}; build type: "
        f"{metadata['compiler']['build_type']}; flags: "
        f"`{metadata['compiler']['configured_flags']}`",
        f"- Available CPUs: {metadata['available_cpus']}; tested thread counts: "
        f"`{metadata['thread_counts']}`",
        f"- Batches: {metadata['batches']}; seed: {metadata['seed']}; output is "
        "Python API latency, not isolated kernel latency.",
        "NumPy is included once per cache regime/mode/size as a single-thread "
        "reference. PyTorch is not included in this thread-scaling run.",
        "- Allocating calls include output creation and release. In-place calls reuse "
        "the prepared operands and are not reset inside timed batches.",
        "- Hot reuses one pair; rotating cycles a prebuilt operand pool targeting "
        f"{metadata['working_set_mb']} MiB (not claimed DRAM-cold).",
        "- Speedups are computed against the one-thread LabNeura result from this run. "
        "NumPy is included as a single-thread reference and is not duplicated "
        "for each configured LabNeura thread count.", "",
        "Each latency cell is `median µs (IQR µs / p95 µs)`.", "",
    ]
    for regime in ("hot", "rotating"):
        lines += [f"## {regime}", ""]
        for layer in ("allocating", "inplace"):
            lines += [f"### {layer}", "",
                      "| Size | Backend | Threads | Median (IQR / p95), µs | Speedup vs 1T |",
                      "|---:|---|---:|---:|---:|"]
            for size in sizes:
                for backend in ("GENERIC", metadata["simd_backend"]):
                    impl = f"labneura-{backend}"
                    baseline = by_key[(regime, layer, size, 1, impl)]
                    for count in metadata["thread_counts"]:
                        row = by_key[(regime, layer, size, count, impl)]
                        speedup = baseline["median_us"] / row["median_us"]
                        lines.append(
                            f"| {size} | {backend} | {count} | "
                            f"{row['median_us']:.3f} "
                            f"({row['iqr_us']:.3f} / {row['p95_us']:.3f}) | "
                            f"{speedup:.2f}x |")
            lines += ["", "| Size | NumPy median (IQR / p95), µs |",
                      "|---:|---:|"]
            for size in sizes:
                numpy_row = by_key[(regime, layer, size, 1, "numpy")]
                lines.append(f"| {size} | {numpy_row['median_us']:.3f} "
                             f"({numpy_row['iqr_us']:.3f} / "
                             f"{numpy_row['p95_us']:.3f}) |")
            lines.append("")
    lines += [
        "## Interpretation notes", "",
        "- IQR above 20% of median indicates a noisy row; consult raw `samples_us` "
        "in `benchmark.json`.",
        "- The backend is selected explicitly and verified while building tensors. "
        "NumPy is timed once as the one-thread reference.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _arguments()
    if args.batches < 30:
        raise SystemExit("--batches must be at least 30")
    if args.target_batch_ms <= 0 or args.min_iterations < 1:
        raise SystemExit("batch duration and minimum iterations must be positive")
    sizes = [int(value) for value in args.sizes.split(",")]
    if not sizes or any(size < 1 for size in sizes):
        raise SystemExit("--sizes must contain positive integers")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    available = int(labneura.available_cpu_count())
    counts = _thread_counts(available)
    with backend_override(None):
        simd = labneura.detect_backend()
    if simd not in ("NEON", "AVX2"):
        raise SystemExit("A supported SIMD backend is required")
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_model": _cpu_model(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "compiler": _compiler_metadata(args.cmake_build_dir),
        "available_cpus": available,
        "thread_counts": counts,
        "simd_backend": simd,
        "batches": args.batches,
        "seed": args.seed,
        "working_set_mb": args.working_set_mb,
        "target_batch_ms": args.target_batch_ms,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip(),
    }
    rows: List[Dict[str, Any]] = []
    for size in sizes:
        for regime in ("hot", "rotating"):
            for layer in ("allocating", "inplace"):
                for count in counts:
                    print(f"{regime:8s} {layer:10s} {size:>8} "
                          f"threads={count}", flush=True)
                    rows.extend(run_combination(
                        regime, layer, size, count, args, simd, None))
    columns = [key for key in rows[0] if key != "samples_us"]
    with (output / "benchmark.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items()
                          if key != "samples_us"} for row in rows)
    (output / "benchmark.json").write_text(json.dumps({
        "metadata": metadata, "results": rows,
    }, indent=2) + "\n", encoding="utf-8")
    write_report(output / "report.md", metadata, rows, sizes)
    print(f"Results: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
