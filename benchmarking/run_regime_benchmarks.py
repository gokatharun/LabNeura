#!/usr/bin/env python3
"""Fair FP32 Python-API benchmark with explicit cache regimes.

LabNeura (Generic and SIMD), NumPy and PyTorch CPU all run the same operand contents,
operand sizes and reuse schedule. Two regimes are measured:

* hot:      one operand pair reused on every call;
* rotating: a pool of independent operand pairs visited cyclically. The pool working set
            is sized to exceed the CPU caches, but this is NOT claimed to be DRAM-cold.

Input construction is outside the timed section. NumPy and PyTorch are measured once per
regime. Within each (regime, layer, operation, size) the implementations are timed in
interleaved, rotating batch order so drift affects all of them equally.
"""

import argparse
import csv
import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import labneura  # noqa: E402
from benchmarking.benchmark_runner import (  # noqa: E402
    REQUIRED_SIZES,
    _statistics,
    backend_override,
    collect_metadata,
)

OPERATIONS = ("ADD", "SUB", "MUL", "DIV")
LAYERS = ("allocating", "inplace")
REGIMES = ("hot", "rotating")
_NUMPY = {"ADD": np.add, "SUB": np.subtract, "MUL": np.multiply, "DIV": np.divide}
_METHOD = {"ADD": "add", "SUB": "sub", "MUL": "mul", "DIV": "div"}


def make_operands(operation: str, size: int, set_index: int, seed: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic FP32 operands. The right operand is chosen per operation so that
    repeatedly applying the in-place operation keeps the left operand finite and in the
    normal FP32 range (no overflow, underflow or denormals)."""
    rng = np.random.default_rng([seed, size, OPERATIONS.index(operation), set_index])
    left = (np.float32(1.0) + rng.random(size, dtype=np.float32)).astype(np.float32)
    unit = rng.random(size, dtype=np.float32)
    if operation in ("ADD", "SUB"):
        right = np.float32(5e-4) + np.float32(1e-3) * unit
    else:
        right = np.float32(1.0) + np.float32(2e-6) * (unit - np.float32(0.5))
    return left, right.astype(np.float32)


def pool_size(regime: str, size: int, working_set_mb: int, max_sets: int) -> int:
    if regime == "hot":
        return 1
    pair_bytes = 2 * size * 4
    return int(min(max_sets, max(2, -(-working_set_mb * 1024 * 1024 // pair_bytes))))


class Case:
    """One implementation's prepared operand pool and timed call."""

    def __init__(self, name: str, library: str, backend: str, layer: str,
                 fn: Callable, pairs: List[Tuple[Any, Any]], read: Callable):
        self.name = name
        self.library = library
        self.backend = backend
        self.layer = layer
        self.fn = fn
        self.pairs = pairs
        self.read = read
        self.samples: List[float] = []

    def schedule(self, indices: Sequence[int]) -> List[Tuple[Callable, Any, Any]]:
        return [(self.fn, *self.pairs[i]) for i in indices]

    def run(self, schedule: List[Tuple[Callable, Any, Any]]) -> None:
        # NumPy's in-place form passes the destination as the positional `out` argument.
        if self.library == "numpy" and self.layer == "inplace":
            for call, x, y in schedule:
                call(x, y, x)
        else:
            for call, x, y in schedule:
                call(x, y)


def build_cases(operation: str, layer: str, arrays: List[Tuple[np.ndarray, np.ndarray]],
                simd_backend: str, torch: Any) -> List[Case]:
    cases: List[Case] = []
    suffix = "_inplace" if layer == "inplace" else ""
    for backend in ("GENERIC", simd_backend):
        with backend_override(backend):
            if labneura.detect_backend() != backend:
                raise RuntimeError(f"Requested {backend}, got {labneura.detect_backend()}")
            pairs = [
                (labneura.Tensor(a.tolist(), labneura.QuantizationMode.FP32),
                 labneura.Tensor(b.tolist(), labneura.QuantizationMode.FP32))
                for a, b in arrays
            ]
        cases.append(Case(
            f"labneura-{backend}", "labneura", backend, layer,
            getattr(labneura.Tensor, _METHOD[operation] + suffix), pairs,
            lambda t: np.asarray(t.data_fp32(), dtype=np.float32),
        ))
    cases.append(Case(
        "numpy", "numpy", "CPU", layer, _NUMPY[operation],
        [(a.copy(), b.copy()) for a, b in arrays], lambda t: np.array(t, dtype=np.float32),
    ))
    torch_method = _METHOD[operation] + ("_" if layer == "inplace" else "")
    cases.append(Case(
        "pytorch", "pytorch", "CPU", layer, getattr(torch.Tensor, torch_method),
        [(torch.from_numpy(a.copy()), torch.from_numpy(b.copy())) for a, b in arrays],
        lambda t: t.detach().cpu().numpy().astype(np.float32),
    ))
    return cases


def verify_case(case: Case, operation: str, arrays: List[Tuple[np.ndarray, np.ndarray]],
                simd_backend: str, torch: Any) -> None:
    """Pool contents match the reference operands, and one application of the exact timed
    callable is bit-identical to NumPy float32 arithmetic."""
    last = len(arrays) - 1
    for index in {0, last}:
        np.testing.assert_array_equal(case.read(case.pairs[index][0]), arrays[index][0])
        np.testing.assert_array_equal(case.read(case.pairs[index][1]), arrays[index][1])
    a, b = arrays[0]
    expected = _NUMPY[operation](a, b)
    check = build_cases(operation, case.layer, [(a, b)], simd_backend, torch)
    probe = next(c for c in check if c.name == case.name)
    x, y = probe.pairs[0]
    if case.layer == "inplace":
        probe.run([(probe.fn, x, y)])
        actual = probe.read(x)
    else:
        actual = probe.read(probe.fn(x, y))
    np.testing.assert_array_equal(actual, expected)


def _digest(values: Sequence[int]) -> str:
    return hashlib.sha1(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()[:12]


def run_combination(regime: str, layer: str, operation: str, size: int, args: Any,
                    simd_backend: str, torch: Any) -> List[Dict[str, Any]]:
    sets = pool_size(regime, size, args.working_set_mb, args.max_sets)
    arrays = [make_operands(operation, size, k, args.seed) for k in range(sets)]
    cases = build_cases(operation, layer, arrays, simd_backend, torch)
    for case in cases:
        verify_case(case, operation, arrays, simd_backend, torch)

    # Identical pre-timing work for every implementation (calibration, then a warm-up
    # batch), so every pool has seen the same number of operations when timing starts.
    calibration = [i % sets for i in range(max(64, 2 * sets))]
    per_call = []
    for case in cases:
        schedule = case.schedule(calibration)
        case.run(schedule)
        start = time.perf_counter()
        case.run(schedule)
        per_call.append((time.perf_counter() - start) / len(calibration))
    wanted = int(args.target_batch_ms * 1e-3 / max(min(per_call), 1e-9))
    iterations = min(args.max_iterations, max(args.min_iterations, wanted))
    iterations = -(-iterations // sets) * sets
    indices = [i % sets for i in range(iterations)]
    schedules = [case.schedule(indices) for case in cases]
    for case, schedule in zip(cases, schedules):
        case.run(schedule)

    gc.collect()
    gc.disable()
    try:
        for batch in range(args.batches):
            shift = batch % len(cases)
            for position in range(len(cases)):
                case_index = (position + shift) % len(cases)
                case = cases[case_index]
                if case.library == "labneura":
                    with backend_override(case.backend):
                        if labneura.detect_backend() != case.backend:
                            raise RuntimeError("backend changed during timing")
                start = time.perf_counter()
                case.run(schedules[case_index])
                case.samples.append((time.perf_counter() - start) / iterations)
    finally:
        gc.enable()

    if layer == "inplace":
        for case in cases:
            for index in {0, sets - 1}:
                values = case.read(case.pairs[index][0])
                if not (np.isfinite(values).all() and np.abs(values).min() > 1e-30
                        and np.abs(values).max() < 1e30):
                    raise RuntimeError(f"{case.name}: in-place operands left the normal range")

    rows = []
    operand_bytes = sets * 2 * size * 4
    for case in cases:
        row = {
            "regime": regime, "layer": layer, "operation": operation, "size": size,
            "implementation": case.name, "library": case.library,
            "labneura_backend": case.backend if case.library == "labneura" else None,
            "correctness": "PASS", "batches": args.batches,
            "iterations_per_batch": iterations, "operand_sets": sets,
            "working_set_bytes": operand_bytes,
            "output_bytes_per_call": size * 4 if layer == "allocating" else 0,
            "schedule_digest": _digest(indices),
            "threads": 1,
        }
        row.update(_statistics(case.samples))
        row["samples_us"] = [s * 1e6 for s in case.samples]
        rows.append(row)
    by_name = {row["implementation"]: row for row in rows}
    neon = by_name[f"labneura-{simd_backend}"]
    for row in rows:
        row["median_over_neon_median"] = row["median_us"] / neon["median_us"]
    return rows


def _fmt(row: Dict[str, Any]) -> str:
    flag = "~" if row["iqr_us"] > 0.2 * row["median_us"] else ""
    return f"{row['median_us']:.3f}{flag} ({row['iqr_us']:.3f}/{row['p95_us']:.3f})"


def _index(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple, Dict[str, Any]]:
    return {(r["regime"], r["layer"], r["operation"], r["size"], r["implementation"]): r
            for r in rows}


def write_report(path: Path, metadata: Dict[str, Any], rows: List[Dict[str, Any]],
                 sizes: Sequence[int], simd: str, regimes: Sequence[str],
                 operations: Sequence[str]) -> None:
    idx = _index(rows)
    neon_name = f"labneura-{simd}"
    lines = [
        "# LabNeura FP32 Python-API benchmark (fair operand reuse, cache regimes)", "",
        f"- Timestamp (UTC): {metadata['timestamp_utc']}",
        f"- Commit: `{metadata['git_commit']}` (worktree has {metadata['git_dirty_files']} "
        "modified/untracked paths)",
        f"- CPU: {metadata['cpu_model']} ({metadata['architecture']}); macOS "
        f"{metadata['macos_version']}; {metadata['os']}",
        f"- Python {metadata['python']}, NumPy {metadata['numpy']}, PyTorch "
        f"{metadata['pytorch']}",
        f"- Compiler: {metadata['compiler']['compiler']}; build type "
        f"{metadata['compiler']['build_type']}; configured flags: "
        f"`{metadata['compiler']['configured_flags']}`",
        "- Project compile options (CMakeLists.txt): `-O3 -ffast-math -funroll-loops`, "
        "with `-fno-fast-math` on the FP32 Generic/NEON/AVX2 sources",
        f"- LabNeura module: `{metadata['labneura_module']}` "
        f"(sha256 `{metadata['labneura_module_sha256'][:16]}…`, built "
        f"{metadata['labneura_module_mtime']})",
        f"- Selected SIMD backend: **{simd}** (verified with `labneura.detect_backend()` "
        "inside each backend context, before building tensors and before each timed batch)",
        f"- Threads: NumPy/PyTorch/LabNeura single-threaded "
        f"(PyTorch reports {metadata['pytorch_threads']})",
        f"- Batches: {metadata['batches']}; seed {metadata['seed']}; rows: {len(rows)}",
        "", "## Method", "",
        "- Operands are generated once per (operation, size, set) from the seed and converted "
        "to every library, so contents, sizes and reuse schedule are identical. NumPy no "
        "longer shares one right-hand array; every library uses a distinct right-hand "
        "operand per operand set.",
        "- **hot**: one operand pair reused on every call. **rotating**: a pool of "
        f"independent operand pairs (target working set {metadata['working_set_mb']} MiB, "
        f"at most {metadata['max_sets']} sets) visited cyclically; each set is visited "
        "equally often per batch. The working set is larger than the M1 caches for large "
        "sizes but this is **not** measured as DRAM-cold; for small sizes the pool still "
        "fits in cache (see the working-set column).",
        "- **allocating**: the timed call returns a new tensor/array that is released "
        "immediately (allocation and release are inside the timer for every library). "
        "LabNeura NEON writes directly into the new output; LabNeura Generic still clones "
        "then operates in place.",
        "- **in-place**: `a op= b` on the pool's left operand (LabNeura `*_inplace`, NumPy "
        "`out=a`, PyTorch `a.op_(b)`). Operands persist, so they are mutated across calls "
        "by the same schedule in every library. Right operands are chosen per operation "
        "(ADD/SUB ≈1e-3, MUL/DIV ≈1±1e-6) so the left operand stays finite and in the "
        "normal FP32 range; this was verified after timing.",
        "- Input construction is outside the timers. Correctness is checked before timing: "
        "pool contents equal the reference, and one application of the exact timed callable "
        "is bit-identical to NumPy float32 arithmetic for every implementation.",
        "- Per case: calibration + one warm-up batch (identical for all implementations), "
        "then batches in rotating interleaved order; GC is disabled during timing. "
        "Each sample is the batch time divided by iterations per batch.",
        "- API surfaces differ slightly (LabNeura/PyTorch methods, NumPy ufunc). These are "
        "end-to-end Python API timings and must not be compared with native kernel timings.",
        "- Cells are `median (IQR/p95)` in µs per call. `~` marks IQR > 20% of the median.",
        "- Ratios are time ratios: `X/NEON` > 1 means LabNeura NEON is faster than X.",
        "",
    ]
    for regime in regimes:
        for layer in LAYERS:
            lines += [f"## {regime} — {layer}", ""]
            for operation in operations:
                lines += [
                    f"### {operation}", "",
                    "| Size | Sets | Working set | Iters | Generic | NEON | NumPy | PyTorch "
                    "| Generic/NEON | NumPy/NEON | PyTorch/NEON |",
                    "|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|",
                ]
                for size in sizes:
                    try:
                        g = idx[(regime, layer, operation, size, "labneura-GENERIC")]
                        n = idx[(regime, layer, operation, size, neon_name)]
                        p = idx[(regime, layer, operation, size, "numpy")]
                        t = idx[(regime, layer, operation, size, "pytorch")]
                    except KeyError:
                        continue
                    ws = n["working_set_bytes"] / 1024
                    ws_text = f"{ws / 1024:.1f} MiB" if ws >= 1024 else f"{ws:.1f} KiB"
                    lines.append(
                        f"| {size:,} | {n['operand_sets']} | {ws_text} | "
                        f"{n['iterations_per_batch']} | {_fmt(g)} | {_fmt(n)} | {_fmt(p)} | "
                        f"{_fmt(t)} | {g['median_us'] / n['median_us']:.2f}x | "
                        f"{p['median_us'] / n['median_us']:.2f}x | "
                        f"{t['median_us'] / n['median_us']:.2f}x |")
                lines.append("")
    if len(regimes) == 2:
        lines += ["## Hot vs rotating (rotating median / hot median, per implementation)", ""]
        for layer in LAYERS:
            lines += [f"### {layer}", "",
                      "| Op | Size | Generic | NEON | NumPy | PyTorch |", "|---|---:|---:|---:|---:|---:|"]
            for operation in operations:
                for size in sizes:
                    cells = []
                    for name in ("labneura-GENERIC", neon_name, "numpy", "pytorch"):
                        hot = idx.get(("hot", layer, operation, size, name))
                        rot = idx.get(("rotating", layer, operation, size, name))
                        cells.append(f"{rot['median_us'] / hot['median_us']:.2f}x"
                                     if hot and rot else "—")
                    lines.append(f"| {operation} | {size:,} | " + " | ".join(cells) + " |")
            lines.append("")
    lines += ["## Plots", ""]
    lines += [f"- `plots/fp32_{op.lower()}_latency.png`, `plots/fp32_{op.lower()}_ratio.png`"
              for op in operations]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(output_dir: Path, rows: List[Dict[str, Any]], sizes: Sequence[int],
                simd: str, regimes: Sequence[str], operations: Sequence[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = _index(rows)
    plots = output_dir / "plots"
    plots.mkdir()
    labels = {"labneura-GENERIC": "LabNeura Generic", f"labneura-{simd}": f"LabNeura {simd}",
              "numpy": "NumPy", "pytorch": "PyTorch CPU"}
    for operation in operations:
        for kind in ("latency", "ratio"):
            fig, axes = plt.subplots(len(regimes), 2, figsize=(12, 4.2 * len(regimes)),
                                     squeeze=False, sharex=True)
            for r, regime in enumerate(regimes):
                for c, layer in enumerate(LAYERS):
                    axis = axes[r][c]
                    for name, label in labels.items():
                        if kind == "ratio" and name == f"labneura-{simd}":
                            continue
                        xs, ys = [], []
                        for size in sizes:
                            row = idx.get((regime, layer, operation, size, name))
                            neon = idx.get((regime, layer, operation, size, f"labneura-{simd}"))
                            if not row or not neon:
                                continue
                            xs.append(size)
                            ys.append(row["median_us"] if kind == "latency"
                                      else row["median_us"] / neon["median_us"])
                        axis.plot(xs, ys, marker="o", label=label)
                    axis.set_xscale("log")
                    if kind == "latency":
                        axis.set_yscale("log")
                        axis.set_ylabel("Median latency per call (µs)")
                    else:
                        axis.axhline(1.0, color="black", linewidth=0.8)
                        axis.set_ylabel(f"time / {simd} time (>1: {simd} faster)")
                    axis.set_title(f"{regime} — {layer}")
                    axis.set_xlabel("Elements")
                    axis.grid(True, which="both", alpha=0.25)
            handles, names = axes[0][0].get_legend_handles_labels()
            fig.legend(handles, names, loc="upper center", ncol=4)
            fig.suptitle(f"FP32 {operation}: Python API {kind}")
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            fig.savefig(plots / f"fp32_{operation.lower()}_{kind}.png", dpi=130)
            plt.close(fig)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sizes", type=str, help="comma-separated; default is the required grid")
    parser.add_argument("--operations", type=str, default=",".join(OPERATIONS))
    parser.add_argument("--regimes", type=str, default=",".join(REGIMES))
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--target-batch-ms", type=float, default=10.0,
                        help="aim for batches of about this duration (bounded below)")
    parser.add_argument("--min-iterations", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=4096)
    parser.add_argument("--working-set-mb", type=int, default=64,
                        help="operand working set targeted by the rotating regime")
    parser.add_argument("--max-sets", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--cmake-build-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.smoke:
        args.batches = min(args.batches, 5)
        args.target_batch_ms = min(args.target_batch_ms, 2.0)
    if args.batches < 1 or args.min_iterations < 1:
        raise SystemExit("--batches and --min-iterations must be positive")
    sizes = ([int(v) for v in args.sizes.split(",")] if args.sizes
             else ([10, 1024, 65536] if args.smoke else list(REQUIRED_SIZES)))
    operations = [v.strip().upper() for v in args.operations.split(",")]
    regimes = [v.strip().lower() for v in args.regimes.split(",")]
    if any(v not in OPERATIONS for v in operations) or any(v not in REGIMES for v in regimes):
        raise SystemExit("unknown --operations or --regimes value")

    import torch
    from threadpoolctl import threadpool_info, threadpool_limits

    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if torch.get_num_threads() != 1:
            raise SystemExit("PyTorch is not single-threaded")
        with backend_override(None):
            simd = labneura.detect_backend()
        if simd == "GENERIC":
            raise SystemExit("No SIMD backend selected on this CPU")
        metadata = collect_metadata(torch, threadpool_info(), args.cmake_build_dir)
        module = Path(labneura.__file__)
        metadata.update({
            "macos_version": platform.mac_ver()[0] or "n/a",
            "git_dirty_files": len(subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True,
                cwd=ROOT).stdout.splitlines()),
            "labneura_module": str(module),
            "labneura_module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            "labneura_module_mtime": datetime.fromtimestamp(
                module.stat().st_mtime, timezone.utc).isoformat(),
            "simd_backend": simd, "seed": args.seed, "batches": args.batches,
            "working_set_mb": args.working_set_mb, "max_sets": args.max_sets,
            "target_batch_ms": args.target_batch_ms,
            "min_iterations": args.min_iterations, "max_iterations": args.max_iterations,
        })
        output_dir = args.output_dir
        if output_dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_dir = ROOT / "benchmarking" / "results" / (
                f"regimes-{stamp}-{metadata['git_commit']}")
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)

        rows: List[Dict[str, Any]] = []
        for size in sizes:
            for operation in operations:
                for layer in LAYERS:
                    for regime in regimes:
                        rows += run_combination(regime, layer, operation, size, args, simd, torch)
                        last = rows[-4:]
                        print(f"{regime:8s} {layer:10s} {operation} {size:>8,}  " + "  ".join(
                            f"{r['implementation'].replace('labneura-', 'lab-')}="
                            f"{r['median_us']:.3f}" for r in last), flush=True)

    payload = {"metadata": metadata,
               "methodology": {"sizes": sizes, "operations": operations, "regimes": regimes,
                               "layers": list(LAYERS), "dtype": "FP32",
                               "correctness": "bit-identical to NumPy float32 before timing"},
               "results": rows}
    (output_dir / "benchmark.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = [k for k in rows[0] if k != "samples_us"]
    with (output_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_report(output_dir / "report.md", metadata, rows, sizes, simd, regimes, operations)
    write_plots(output_dir, rows, sizes, simd, regimes, operations)
    print(f"Selected SIMD backend: {simd}\nResults: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
