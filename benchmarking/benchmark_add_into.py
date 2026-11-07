#!/usr/bin/env python3
"""Compare allocating FP32 ADD with reusable-output ADD across CPU libraries."""

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
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import labneura  # noqa: E402
from benchmarking.benchmark_runner import (  # noqa: E402
    _compiler_metadata,
    _cpu_model,
    backend_override,
)

DEFAULT_SIZES = (1024, 4096, 16384, 65536, 262144, 1000000)


class Case:
    def __init__(self, library: str, backend: str, mode: str,
                 calls: List[Callable[[], None]], pairs: List[Tuple[Any, Any, Any]],
                 size: int):
        self.library = library
        self.backend = backend
        self.mode = mode
        self.calls = calls
        self.pairs = pairs
        self.pair_count = len(pairs)
        self.size = size
        self.samples_us: List[float] = []

    def run(self, indices: List[int]) -> None:
        for index in indices:
            self.calls[index]()


def _statistics(samples: List[float]) -> Dict[str, float]:
    ordered = sorted(samples)
    count = len(ordered)

    def percentile(fraction: float) -> float:
        position = (count - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, count - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (
            position - lower)

    q1, median, q3 = (percentile(value) for value in (0.25, 0.5, 0.75))
    return {
        "median_us": median,
        "iqr_us": q3 - q1,
        "p95_us": percentile(0.95),
    }


def _make_arrays(size: int, set_count: int, seed: int
                 ) -> List[Tuple[np.ndarray, np.ndarray]]:
    return [
        (
            np.random.default_rng([seed, size, index, 0])
            .normal(size=size).astype(np.float32),
            np.random.default_rng([seed, size, index, 1])
            .normal(size=size).astype(np.float32),
        )
        for index in range(set_count)
    ]


def _build_cases(arrays: List[Tuple[np.ndarray, np.ndarray]], mode: str,
                 torch: Any, simd_backend: str) -> List[Case]:
    size = int(arrays[0][0].size)
    cases: List[Case] = []

    for backend in ("GENERIC", simd_backend):
        with backend_override(backend):
            selected = labneura.detect_backend()
            if selected != backend:
                raise RuntimeError(f"Requested {backend}, selected {selected}")
            pairs = []
            for left, right in arrays:
                a = labneura.Tensor(left.tolist(), labneura.QuantizationMode.FP32)
                b = labneura.Tensor(right.tolist(), labneura.QuantizationMode.FP32)
                out = (labneura.Tensor([0.0] * size,
                                       labneura.QuantizationMode.FP32)
                       if mode == "reusable_output" else None)
                pairs.append((a, b, out))

        calls = []
        for a, b, out in pairs:
            if mode == "reusable_output":
                calls.append(lambda a=a, b=b, out=out:
                             labneura.add_into(a, b, out))
            else:
                def call(a=a, b=b):
                    result = a.add(b)
                    del result
                calls.append(call)
        cases.append(Case("LabNeura", backend, mode, calls, pairs, size))

    numpy_pairs = [(left.copy(), right.copy(),
                    np.zeros(size, dtype=np.float32)
                    if mode == "reusable_output" else None)
                   for left, right in arrays]
    numpy_calls = []
    for a, b, out in numpy_pairs:
        if mode == "reusable_output":
            numpy_calls.append(lambda a=a, b=b, out=out: np.add(a, b, out=out))
        else:
            def call(a=a, b=b):
                result = np.add(a, b)
                del result
            numpy_calls.append(call)
    cases.append(Case("NumPy", "CPU", mode, numpy_calls, numpy_pairs, size))

    torch_pairs = []
    for left, right in arrays:
        a = torch.from_numpy(left.copy())
        b = torch.from_numpy(right.copy())
        out = torch.empty_like(a) if mode == "reusable_output" else None
        torch_pairs.append((a, b, out))
    torch_calls = []
    for a, b, out in torch_pairs:
        if mode == "reusable_output":
            torch_calls.append(lambda a=a, b=b, out=out:
                               torch.add(a, b, out=out))
        else:
            def call(a=a, b=b):
                result = torch.add(a, b)
                del result
            torch_calls.append(call)
    cases.append(Case("PyTorch", "CPU", mode, torch_calls, torch_pairs, size))
    return cases


def _check_cases(cases: List[Case], arrays: List[Tuple[np.ndarray, np.ndarray]],
                 mode: str) -> None:
    expected = arrays[0][0] + arrays[0][1]
    for case in cases:
        case.run([0])
        left, right, out = case.pairs[0]
        if case.library == "LabNeura":
            result = out if mode == "reusable_output" else left.add(right)
            actual = np.asarray(result.data_fp32(), dtype=np.float32)
        elif case.library == "NumPy":
            actual = out if mode == "reusable_output" else left + right
        else:
            actual = (out.numpy() if mode == "reusable_output"
                      else torch_result(left, right))
        np.testing.assert_array_equal(actual, expected,
                                      err_msg=f"{case.library}/{case.backend}")
        if case.library == "LabNeura":
            actual_left = np.asarray(left.data_fp32(), dtype=np.float32)
            actual_right = np.asarray(right.data_fp32(), dtype=np.float32)
        elif case.library == "NumPy":
            actual_left, actual_right = left, right
        else:
            actual_left, actual_right = left.numpy(), right.numpy()
        np.testing.assert_array_equal(actual_left, arrays[0][0])
        np.testing.assert_array_equal(actual_right, arrays[0][1])


def torch_result(left: Any, right: Any) -> np.ndarray:
    import torch
    return torch.add(left, right).numpy()


def _pool_size(regime: str, size: int, target_mb: int,
               max_sets: int) -> int:
    if regime == "hot":
        return 1
    bytes_per_set = 3 * size * np.dtype(np.float32).itemsize
    return min(max_sets, max(2, (target_mb * 1024 * 1024 + bytes_per_set - 1)
                            // bytes_per_set))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    parser.add_argument("--regimes", default="hot,rotating")
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--target-batch-ms", type=float, default=5.0)
    parser.add_argument("--min-iterations", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=2048)
    parser.add_argument("--working-set-mb", type=int, default=64)
    parser.add_argument("--max-sets", type=int, default=128)
    parser.add_argument("--seed", type=int, default=202603)
    parser.add_argument("--cmake-build-dir", type=Path,
                        default=ROOT / "build" / "local-python")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _measure(cases: List[Case], regime: str, layer: str, size: int,
             args: argparse.Namespace) -> List[Dict[str, Any]]:
    calibration_count = max(32, len(cases[0].calls) * 2)
    calibration_indices = [i % len(cases[0].calls)
                           for i in range(calibration_count)]
    per_call_seconds = []
    for case in cases:
        case.run(calibration_indices)
        start = time.perf_counter()
        case.run(calibration_indices)
        per_call_seconds.append((time.perf_counter() - start) /
                                len(calibration_indices))
    iterations = int(args.target_batch_ms * 1e-3 /
                     max(min(per_call_seconds), 1e-9))
    iterations = min(args.max_iterations,
                     max(args.min_iterations, iterations))
    iterations = -(-iterations // len(cases[0].calls)) * len(cases[0].calls)
    indices = [index % len(cases[0].calls) for index in range(iterations)]

    for case in cases:
        case.run(indices)
        case.samples_us.clear()
    gc.collect()
    gc.disable()
    try:
        for batch in range(args.batches):
            order = list(range(len(cases)))
            shift = batch % len(order)
            order = order[shift:] + order[:shift]
            for case_index in order:
                case = cases[case_index]
                start = time.perf_counter()
                case.run(indices)
                case.samples_us.append(
                    (time.perf_counter() - start) * 1e6 / iterations)
    finally:
        gc.enable()

    rows = []
    for case in cases:
        row = {
            "regime": regime,
            "operation": "FP32_ADD",
            "mode": layer,
            "size": size,
            "library": case.library,
            "backend": case.backend,
            "threads": 1,
            "batches": args.batches,
            "iterations_per_batch": iterations,
            "operand_sets": case.pair_count,
            "input_working_set_bytes": case.pair_count * size * 4 * 2,
            "preallocated_output_bytes": (
                case.pair_count * size * 4
                if layer == "reusable_output" else 0
            ),
            "correctness": "PASS",
        }
        row["working_set_bytes"] = (
            row["input_working_set_bytes"] + row["preallocated_output_bytes"])
        row["peak_estimated_working_set_bytes"] = (
            row["working_set_bytes"] +
            (size * 4 if layer == "allocating" else 0))
        row.update(_statistics(case.samples_us))
        row["samples_us"] = case.samples_us
        rows.append(row)
    return rows


def _write_report(path: Path, metadata: Dict[str, Any],
                  rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Reusable-output FP32 ADD benchmark", "",
        f"- Timestamp UTC: {metadata['timestamp_utc']}",
        f"- CPU: {metadata['cpu_model']}; OS: {metadata['os']}",
        f"- Python: {metadata['python']}; NumPy: {metadata['numpy']}; "
        f"PyTorch: {metadata['pytorch']}",
        f"- LabNeura backend(s): {', '.join(metadata['labneura_backends'])}; "
        f"execution threads: {metadata['execution_threads']}",
        f"- Compiler: {metadata['compiler']['compiler']}; build type: "
        f"{metadata['compiler']['build_type']}; flags: "
        f"`{metadata['compiler']['configured_flags']}`",
        f"- Rotating pool target: {metadata['working_set_mb']} MiB; pool capped "
        f"at {metadata['max_sets']} sets. The target includes input and reusable "
        "output pools; allocating mode has input pools and a transient output. "
        "This is not a claim of DRAM-cold execution.",
        "- Allocating mode times result creation and release. Reusable-output "
        "mode reuses a preallocated output for every call.",
        "- All inputs are constructed before timing. Reusable-output destinations "
        "are also preallocated; allocating results are created and released inside "
        "the timed calls. Each library cycles through the same deterministic "
        "operand-set indices. Inputs are not mutated during timed calls.",
        "- NumPy native thread pools and PyTorch CPU threads are set to one. "
        "LabNeura execution thread count is explicitly set to one.", "",
        "Median, IQR and p95 are per-call Python API latency in microseconds. "
        "Raw batch samples are in `benchmark.json`.", "",
    ]
    for regime in metadata["regimes"]:
        lines += [f"## {regime.title()} cache regime", ""]
        for mode in ("allocating", "reusable_output"):
            lines += [f"### {mode.replace('_', ' ').title()}", "",
                      "| Elements | Library / backend | Median µs | IQR µs | p95 µs | "
                      "vs NumPy | vs PyTorch | Generic / NEON | "
                      "Iterations/batch | Pool working set MiB |",
                      "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for row in rows:
                if row["regime"] != regime or row["mode"] != mode:
                    continue
                generic_over_neon = row.get("generic_over_neon")
                generic_neon_text = (
                    f"{generic_over_neon:.2f}x"
                    if generic_over_neon is not None else "n/a")
                lines.append(
                    f"| {row['size']} | {row['library']} / {row['backend']} | "
                    f"{row['median_us']:.3f} | {row['iqr_us']:.3f} | "
                    f"{row['p95_us']:.3f} | "
                    f"{row['speedup_vs_numpy']:.2f}x | "
                    f"{row['speedup_vs_pytorch']:.2f}x | "
                    f"{generic_neon_text} | {row['iterations_per_batch']} | "
                    f"{row['working_set_bytes'] / 1048576:.2f} |")
            lines.append("")
    lines += [
        "## Interpretation", "",
        "Compare reusable-output rows only with each other, and allocating rows "
        "only with each other. Results include Python binding and dispatch "
        "overhead; they are not isolated-kernel measurements. Backend selection "
        "is explicit and checked before input construction. Results are specific "
        "to the recorded machine and software versions. Speedup ratios are "
        "competitor median divided by the row median; values above 1.0x mean the "
        "row was faster. `Generic / NEON` is Generic median divided by NEON median.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _arguments()
    if args.batches < 30:
        raise SystemExit("--batches must be at least 30")
    if args.target_batch_ms <= 0 or args.min_iterations < 1:
        raise SystemExit("batch duration and minimum iterations must be positive")
    sizes = [int(value) for value in args.sizes.split(",")]
    regimes = [value.strip() for value in args.regimes.split(",")]
    if not sizes or any(size < 1 for size in sizes):
        raise SystemExit("--sizes must contain positive integers")
    if not regimes or any(regime not in ("hot", "rotating") for regime in regimes):
        raise SystemExit("--regimes must contain hot and/or rotating")
    try:
        import torch
        from threadpoolctl import threadpool_limits
    except ImportError as error:
        raise SystemExit(
            "This benchmark requires installed PyTorch and threadpoolctl; "
            "no dependencies were installed."
        ) from error

    if args.max_iterations < args.min_iterations:
        raise SystemExit("--max-iterations must be at least --min-iterations")
    if args.working_set_mb < 1 or args.max_sets < 1:
        raise SystemExit("working-set size and maximum operand sets must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    threadpool_context = threadpool_limits(limits=1)
    labneura.set_execution_thread_count(1)
    simd_backend = labneura.detect_backend()
    if simd_backend not in ("NEON", "AVX2"):
        raise SystemExit("A supported SIMD backend is required")
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_model": _cpu_model(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pytorch": torch.__version__,
        "labneura_version": getattr(labneura, "__version__", "unknown"),
        "labneura_module": str(Path(labneura.__file__).resolve()),
        "labneura_backends": ["GENERIC", simd_backend],
        "execution_threads": labneura.execution_thread_count(),
        "compiler": _compiler_metadata(args.cmake_build_dir),
        "working_set_mb": args.working_set_mb,
        "max_sets": args.max_sets,
        "batches": args.batches,
        "target_batch_ms": args.target_batch_ms,
        "seed": args.seed,
        "regimes": regimes,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip(),
    }
    threadpool_context.__enter__()
    results: List[Dict[str, Any]] = []
    try:
        for size in sizes:
            for regime in regimes:
                set_count = _pool_size(
                    regime, size, args.working_set_mb, args.max_sets)
                arrays = _make_arrays(size, set_count, args.seed)
                for mode in ("allocating", "reusable_output"):
                    cases = _build_cases(arrays, mode, torch, simd_backend)
                    _check_cases(cases, arrays, mode)
                    print(f"{regime:8s} {mode:15s} {size:>8} "
                          f"sets={set_count}", flush=True)
                    results.extend(_measure(cases, regime, mode, size, args))
    finally:
        threadpool_context.__exit__(None, None, None)

    indexed = {
        (row["regime"], row["mode"], row["size"], row["library"], row["backend"]): row
        for row in results
    }
    for row in results:
        peers = (row["regime"], row["mode"], row["size"])
        numpy_row = indexed[(*peers, "NumPy", "CPU")]
        pytorch_row = indexed[(*peers, "PyTorch", "CPU")]
        generic_row = indexed[(*peers, "LabNeura", "GENERIC")]
        neon_row = indexed[(*peers, "LabNeura", simd_backend)]
        row["speedup_vs_numpy"] = numpy_row["median_us"] / row["median_us"]
        row["speedup_vs_pytorch"] = pytorch_row["median_us"] / row["median_us"]
        row["generic_over_neon"] = (
            generic_row["median_us"] / neon_row["median_us"]
            if row["library"] == "LabNeura" and row["backend"] == "GENERIC"
            else None
        )

    columns = [key for key in results[0] if key != "samples_us"]
    with (output / "benchmark.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items()
                          if key != "samples_us"} for row in results)
    (output / "benchmark.json").write_text(json.dumps({
        "metadata": metadata, "results": results,
    }, indent=2) + "\n", encoding="utf-8")
    _write_report(output / "report.md", metadata, results)
    print(f"Results: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
