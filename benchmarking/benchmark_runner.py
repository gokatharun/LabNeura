"""Reproducible Python/API benchmarks for LabNeura, NumPy, and PyTorch."""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import labneura


REQUIRED_SIZES = [
    7, 8, 9, 10, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
    256, 512, 1024, 4096, 16384, 65536, 262144, 1000000,
]
OPERATIONS = ("add", "sub", "mul", "div")
LAYERS = ("allocating", "inplace")
DTYPES = ("FP32", "INT8", "FP16", "INT16")
GENERIC_DTYPES = {"FP32", "INT8"}


def _operation(lhs: Any, rhs: Any, operation: str, **kwargs: Any) -> Any:
    return {
        "add": np.add,
        "sub": np.subtract,
        "mul": np.multiply,
        "div": np.divide,
    }[operation](lhs, rhs, **kwargs)


def _integer_result(lhs: np.ndarray, rhs: np.ndarray, operation: str,
                    dtype: np.dtype) -> np.ndarray:
    info = np.iinfo(dtype)
    lhs_wide = lhs.astype(np.int64)
    rhs_wide = rhs.astype(np.int64)
    if operation == "add":
        result = lhs_wide + rhs_wide
    elif operation == "sub":
        result = lhs_wide - rhs_wide
    elif operation == "mul":
        result = lhs_wide * rhs_wide
    else:
        result = np.trunc(lhs_wide / rhs_wide).astype(np.int64)
    return np.clip(result, info.min, info.max).astype(dtype)


def reference_result(lhs: np.ndarray, rhs: np.ndarray, operation: str,
                     dtype_name: str) -> np.ndarray:
    if dtype_name in ("INT8", "INT16"):
        return _integer_result(lhs, rhs, operation, lhs.dtype)
    return _operation(lhs, rhs, operation)


def generate_inputs(dtype_name: str, size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic operands with safe, nonzero divisors."""
    rng = np.random.default_rng(seed)
    if dtype_name in ("FP32", "FP16"):
        lhs = rng.uniform(-5.0, 5.0, size).astype(np.float32)
        rhs = rng.uniform(0.25, 2.0, size).astype(np.float32)
        rhs[::2] *= -1.0
        if dtype_name == "FP16":
            lhs = lhs.astype(np.float16)
            rhs = rhs.astype(np.float16)
        return lhs, rhs

    low, high = (-128, 128) if dtype_name == "INT8" else (-32768, 32768)
    dtype = np.int8 if dtype_name == "INT8" else np.int16
    lhs = rng.integers(low, high, size, dtype=np.int32).astype(dtype)
    rhs = rng.integers(low, high, size, dtype=np.int32).astype(dtype)
    rhs[rhs == 0] = 1
    return lhs, rhs


@contextmanager
def backend_override(name: Optional[str]) -> Iterable[None]:
    """Temporarily set LABNEURA_BACKEND and restore its original state."""
    key = "LABNEURA_BACKEND"
    was_set = key in os.environ
    old_value = os.environ.get(key)
    if name is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = name
    try:
        yield
    finally:
        if was_set:
            os.environ[key] = old_value if old_value is not None else ""
        else:
            os.environ.pop(key, None)


def _lab_values(tensor: Any, dtype_name: str) -> np.ndarray:
    if dtype_name == "FP32":
        return np.asarray(tensor.data_fp32(), dtype=np.float32)
    if dtype_name == "INT8":
        return np.asarray(tensor.data_int8(), dtype=np.int8)
    if dtype_name == "INT16":
        return np.asarray(tensor.data_int16(), dtype=np.int16)
    return np.asarray(
        [labneura.dequantize_fp16(bits) for bits in tensor.data_fp16()],
        dtype=np.float16,
    )


def _tensor(values: np.ndarray, dtype_name: str) -> Any:
    dtype_mode = getattr(labneura.QuantizationMode, dtype_name)
    return labneura.Tensor(values.tolist(), dtype_mode)


def _make_torch_values(values: np.ndarray, torch: Any) -> Any:
    return torch.from_numpy(values.copy()).to(device="cpu")


def _torch_dtype(dtype_name: str, torch: Any) -> Any:
    return {
        "FP32": torch.float32,
        "FP16": torch.float16,
        "INT8": torch.int8,
        "INT16": torch.int16,
    }[dtype_name]


def _torch_values(tensor: Any, dtype_name: str) -> np.ndarray:
    values = tensor.detach().cpu().numpy()
    if dtype_name in ("INT8", "INT16"):
        return values
    return values.astype(np.float32)


def _check_result(actual: np.ndarray, expected: np.ndarray, dtype_name: str) -> None:
    if dtype_name in ("INT8", "INT16"):
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3,
                                   equal_nan=True)


def _numpy_call(lhs: np.ndarray, rhs: np.ndarray, operation: str,
                layer: str, dtype_name: str, scratch: Optional[np.ndarray],
                rhs_wide: Optional[np.ndarray],
                quotient_scratch: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if dtype_name not in ("INT8", "INT16"):
        if layer == "inplace":
            _operation(lhs, rhs, operation, out=lhs)
            return None
        return _operation(lhs, rhs, operation)

    assert scratch is not None and rhs_wide is not None
    info = np.iinfo(lhs.dtype)
    np.copyto(scratch, lhs, casting="unsafe")
    if operation == "div":
        assert quotient_scratch is not None
        np.divide(scratch, rhs_wide, out=quotient_scratch)
        np.trunc(quotient_scratch, out=quotient_scratch)
        np.copyto(scratch, quotient_scratch, casting="unsafe")
    else:
        _operation(scratch, rhs_wide, operation, out=scratch)
    np.clip(scratch, info.min, info.max, out=scratch)
    if layer == "inplace":
        np.copyto(lhs, scratch, casting="unsafe")
        return None
    return scratch.astype(lhs.dtype)


def _torch_call(lhs: Any, rhs: Any, operation: str, layer: str,
                dtype_name: str, torch: Any, scratch: Optional[Any],
                rhs_wide: Optional[Any]) -> Optional[Any]:
    if dtype_name not in ("INT8", "INT16"):
        if layer == "inplace":
            {
                "add": torch.add,
                "sub": torch.sub,
                "mul": torch.mul,
                "div": torch.div,
            }[operation](lhs, rhs, out=lhs)
            return None
        return {
            "add": torch.add,
            "sub": torch.sub,
            "mul": torch.mul,
            "div": torch.div,
        }[operation](lhs, rhs)

    assert scratch is not None and rhs_wide is not None
    info = np.iinfo(np.int8 if dtype_name == "INT8" else np.int16)
    scratch.copy_(lhs)
    if operation == "div":
        torch.div(scratch, rhs_wide, rounding_mode="trunc", out=scratch)
    else:
        {
            "add": torch.add,
            "sub": torch.sub,
            "mul": torch.mul,
        }[operation](scratch, rhs_wide, out=scratch)
    scratch.clamp_(info.min, info.max)
    if layer == "inplace":
        lhs.copy_(scratch)
        return None
    return scratch.to(dtype=_torch_dtype(dtype_name, torch))


def _statistics(samples_seconds: Sequence[float]) -> Dict[str, float]:
    samples_us = [sample * 1e6 for sample in samples_seconds]
    q1, q3 = np.percentile(samples_us, [25, 75])
    return {
        "median_us": float(statistics.median(samples_us)),
        "iqr_us": float(q3 - q1),
        "p95_us": float(np.percentile(samples_us, 95)),
        "min_us": float(min(samples_us)),
        "max_us": float(max(samples_us)),
    }


def _cpu_model() -> str:
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip() or platform.processor() or platform.machine()
    if sys.platform.startswith("linux"):
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                if line.lower().startswith(("model name", "hardware")):
                    return line.split(":", 1)[-1].strip()
    return platform.processor() or platform.machine()


def _compiler_metadata(build_directory: Optional[Path]) -> Dict[str, Any]:
    if build_directory is None:
        return {
            "compiler": None,
            "build_type": None,
            "configured_flags": None,
            "cmake_build_directory": None,
            "backend_options": {},
        }
    cache_path = build_directory / "CMakeCache.txt"
    if not cache_path.is_file():
        raise FileNotFoundError(f"No CMakeCache.txt found in {build_directory}")
    values: Dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("CMAKE_CXX_COMPILER:FILEPATH=", "CMAKE_BUILD_TYPE:STRING=",
                            "CMAKE_CXX_FLAGS:STRING=", "CMAKE_CXX_FLAGS_RELEASE:STRING=",
                            "LABNEURA_ENABLE_")):
            key, value = line.split("=", 1)
            values[key.split(":", 1)[0]] = value
    compiler = values.get("CMAKE_CXX_COMPILER")
    compiler_version: Optional[str] = None
    if compiler:
        result = subprocess.run(
            [compiler, "--version"], check=True, capture_output=True, text=True
        )
        compiler_version = result.stdout.splitlines()[0] if result.stdout else compiler
    return {
        "compiler": compiler_version,
        "build_type": values.get("CMAKE_BUILD_TYPE"),
        "configured_flags": " ".join(
            value for key, value in values.items() if key.startswith("CMAKE_CXX_FLAGS")
        ),
        "cmake_build_directory": str(build_directory.resolve()),
        "backend_options": {
            key: value for key, value in values.items()
            if key.startswith("LABNEURA_ENABLE_")
        },
    }


def collect_metadata(torch: Any, threadpool_info: Sequence[Dict[str, Any]],
                     cmake_build_directory: Optional[Path] = None) -> Dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_model": _cpu_model(),
        "architecture": platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pytorch": torch.__version__,
        "labneura": getattr(labneura, "__version__", "unknown"),
        "git_commit": commit,
        "compiler": _compiler_metadata(cmake_build_directory),
        "numpy_threadpools": list(threadpool_info),
        "pytorch_threads": torch.get_num_threads(),
        "pytorch_device": "cpu",
        "working_set": "repeated deterministic inputs; cache-resident",
    }


class BenchmarkRunner:
    """Run correctness-checked comparisons without timing input setup."""

    def __init__(self, torch: Any, sizes: Sequence[int] = REQUIRED_SIZES,
                 batches: int = 7, iterations: int = 50, seed: int = 1729,
                 threads: int = 1, dtypes: Sequence[str] = DTYPES,
                 memory_budget_mb: int = 32):
        if batches < 1 or iterations < 1:
            raise ValueError("batches and iterations must be positive")
        if threads < 1:
            raise ValueError("threads must be positive")
        self.torch = torch
        self.sizes = list(sizes)
        self.batches = batches
        self.iterations = iterations
        self.seed = seed
        self.threads = threads
        unknown = [name for name in dtypes if name not in DTYPES]
        if unknown or not dtypes:
            raise ValueError(f"dtypes must be a non-empty subset of {DTYPES}")
        if memory_budget_mb < 1:
            raise ValueError("memory_budget_mb must be positive")
        self.dtypes = list(dtypes)
        self.memory_budget_bytes = memory_budget_mb * 1024 * 1024
        self.simd_backend = self._automatic_simd_backend()

    @staticmethod
    def _automatic_simd_backend() -> str:
        with backend_override(None):
            backend = labneura.detect_backend()
        if backend == "GENERIC":
            raise RuntimeError(
                "No SIMD backend is selected on this CPU; refusing to label "
                "Generic measurements as SIMD"
            )
        return backend

    def _append_unsupported(self, rows: List[Dict[str, Any]], backend: str,
                            dtype_name: str, size: int, operation: str,
                            layer: str) -> None:
        rows.append({
            "comparison_backend": backend,
            "selected_backend": backend,
            "library": "labneura",
            "dtype": dtype_name,
            "operation": operation.upper(),
            "size": size,
            "layer": layer,
            "correctness": "UNSUPPORTED",
            "unsupported_reason": "Generic backend has no FP16 or INT16 kernels",
            "median_us": None,
            "iqr_us": None,
            "p95_us": None,
            "min_us": None,
            "max_us": None,
            "batches": 0,
            "iterations_per_batch": 0,
        })

    def _measure_construction(self, library: str, dtype_name: str, values: np.ndarray,
                              backend: str, iterations: int) -> Dict[str, Any]:
        mode = getattr(labneura.QuantizationMode, dtype_name)
        python_values = values.tolist()

        def make_call() -> Callable[[], Any]:
            if library == "labneura":
                return lambda: labneura.Tensor(python_values, mode)
            if library == "numpy":
                return lambda: np.array(values, copy=True)
            return lambda: self.torch.tensor(values, device="cpu")

        def make_inputs() -> None:
            return None

        def build(_: Any) -> Callable[[], Any]:
            return make_call()

        result = make_call()()
        if library == "labneura":
            actual = _lab_values(result, dtype_name)
        elif library == "numpy":
            actual = result
        else:
            actual = _torch_values(result, dtype_name)
        _check_result(np.asarray(actual), values, dtype_name)

        samples = self._timed_samples(build, make_inputs, self.batches, iterations)
        row = {
            "comparison_backend": backend,
            "selected_backend": backend if library == "labneura" else "CPU (library default)",
            "library": library,
            "dtype": dtype_name,
            "operation": "CONSTRUCT",
            "size": len(values),
            "layer": "construction",
            "correctness": "PASS",
            "unsupported_reason": None,
            "batches": self.batches,
            "iterations_per_batch": iterations,
        }
        row.update(_statistics(samples))
        return row

    def _timed_samples(self, operation: Callable[[], Any], make_inputs: Callable[[], Any],
                       batches: int, iterations: int) -> List[float]:
        for _ in range(max(2, iterations // 10)):
            operands = make_inputs()
            operation_with_inputs = operation(operands)
            operation_with_inputs()

        samples: List[float] = []
        for _ in range(batches):
            prepared = [operation(make_inputs()) for _ in range(iterations)]
            start = time.perf_counter()
            for call in prepared:
                call()
            samples.append((time.perf_counter() - start) / iterations)
        return samples

    def _measure_library(self, library: str, dtype_name: str, lhs: np.ndarray,
                         rhs: np.ndarray, operation: str, layer: str,
                         backend: str, expected: np.ndarray,
                         iterations: int) -> Dict[str, Any]:
        size = len(lhs)
        wide_dtype = np.int32 if dtype_name == "INT8" else np.int64
        if dtype_name in ("INT8", "INT16"):
            wide_rhs_np = rhs.astype(wide_dtype)
            np_scratch = np.empty(size, dtype=wide_dtype)
            quotient_scratch = np.empty(size, dtype=np.float64)
            torch_dtype = self.torch.int32 if dtype_name == "INT8" else self.torch.int64
            torch_rhs_wide = _make_torch_values(rhs, self.torch).to(torch_dtype)
            torch_scratch = self.torch.empty(size, dtype=torch_dtype, device="cpu")
        else:
            wide_rhs_np = None
            np_scratch = None
            quotient_scratch = None
            torch_rhs_wide = None
            torch_scratch = None

        def make_inputs() -> Tuple[Any, Any]:
            if library == "labneura":
                return _tensor(lhs, dtype_name), _tensor(rhs, dtype_name)
            if library == "numpy":
                return lhs.copy(), rhs.copy()
            return (
                _make_torch_values(lhs, self.torch),
                _make_torch_values(rhs, self.torch),
            )

        def make_call(operands: Tuple[Any, Any]) -> Callable[[], Any]:
            left, right = operands
            if library == "labneura":
                if layer == "inplace":
                    method = getattr(left, operation + "_inplace")
                    return lambda: method(right)
                method = getattr(left, operation)
                return lambda: method(right)
            if library == "numpy":
                return lambda: _numpy_call(
                    left, right, operation, layer, dtype_name, np_scratch,
                    wide_rhs_np, quotient_scratch,
                )
            return lambda: _torch_call(
                left, right, operation, layer, dtype_name, self.torch,
                torch_scratch, torch_rhs_wide,
            )

        # Validate the exact allocating or in-place semantics before timing.
        check_inputs = make_inputs()
        check_call = make_call(check_inputs)
        check_call()
        if library == "labneura":
            actual = _lab_values(check_inputs[0], dtype_name) if layer == "inplace" else \
                _lab_values(check_call(), dtype_name)
        elif library == "numpy":
            actual = check_inputs[0] if layer == "inplace" else check_call()
        else:
            actual = _torch_values(check_inputs[0] if layer == "inplace" else check_call(),
                                   dtype_name)
        _check_result(np.asarray(actual), expected, dtype_name)

        samples = self._timed_samples(make_call, make_inputs,
                                      self.batches, iterations)
        row = {
            "comparison_backend": backend,
            "selected_backend": backend if library == "labneura" else "CPU (library default)",
            "library": library,
            "dtype": dtype_name,
            "operation": operation.upper(),
            "size": size,
            "layer": layer,
            "correctness": "PASS",
            "unsupported_reason": None,
            "batches": self.batches,
            "iterations_per_batch": iterations,
        }
        row.update(_statistics(samples))
        return row

    def run(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        backend_modes = (("GENERIC", "GENERIC"), (self.simd_backend, self.simd_backend))
        libraries = ("labneura", "numpy", "pytorch")
        self.torch.set_num_threads(self.threads)

        for dtype_name in self.dtypes:
            for size in self.sizes:
                lhs, rhs = generate_inputs(
                    dtype_name, size, self.seed + size + DTYPES.index(dtype_name) * 1000003
                )
                max_iterations_by_memory = max(
                    3, self.memory_budget_bytes // (size * lhs.dtype.itemsize * 2)
                )
                iterations = min(self.iterations, max_iterations_by_memory)
                for backend_label, override in backend_modes:
                    with backend_override(override):
                        actual = labneura.detect_backend()
                        if actual != backend_label:
                            raise RuntimeError(
                                f"Requested {backend_label}, but LabNeura selected {actual}"
                            )
                        for library in libraries:
                            if (library == "labneura" and backend_label == "GENERIC"
                                    and dtype_name not in GENERIC_DTYPES):
                                self._append_unsupported(
                                    rows, backend_label, dtype_name, size,
                                    "CONSTRUCT", "construction",
                                )
                                continue
                            rows.append(self._measure_construction(
                                library, dtype_name, lhs, backend_label, iterations
                            ))
                for operation in OPERATIONS:
                    expected = reference_result(lhs, rhs, operation, dtype_name)
                    for backend_label, override in backend_modes:
                        with backend_override(override):
                            actual = labneura.detect_backend()
                            if actual != backend_label:
                                raise RuntimeError(
                                    f"Requested {backend_label}, but LabNeura selected {actual}"
                                )
                            for layer in LAYERS:
                                for library in libraries:
                                    if (library == "labneura" and backend_label == "GENERIC"
                                            and dtype_name not in GENERIC_DTYPES):
                                        self._append_unsupported(
                                            rows, backend_label, dtype_name, size,
                                            operation, layer,
                                        )
                                        continue
                                    try:
                                        rows.append(self._measure_library(
                                            library, dtype_name, lhs, rhs, operation,
                                            layer, backend_label, expected, iterations,
                                        ))
                                    except Exception as error:
                                        raise RuntimeError(
                                            "Benchmark case failed before results were "
                                            f"written: backend={backend_label}, "
                                            f"library={library}, dtype={dtype_name}, "
                                            f"operation={operation}, size={size}, "
                                            f"layer={layer}"
                                        ) from error
        self._add_speedups(rows)
        return rows

    @staticmethod
    def _add_speedups(rows: List[Dict[str, Any]]) -> None:
        indexed = {
            (row["comparison_backend"], row["dtype"], row["operation"],
             row["size"], row["layer"], row["library"]): row
            for row in rows
        }
        for row in rows:
            if row["correctness"] != "PASS":
                row["speedup_vs_numpy"] = None
                row["speedup_vs_pytorch"] = None
                continue
            key = (row["comparison_backend"], row["dtype"], row["operation"],
                   row["size"], row["layer"])
            numpy_row = indexed.get((*key, "numpy"))
            pytorch_row = indexed.get((*key, "pytorch"))
            row["speedup_vs_numpy"] = (
                numpy_row["median_us"] / row["median_us"]
                if numpy_row and row["median_us"] else None
            )
            row["speedup_vs_pytorch"] = (
                pytorch_row["median_us"] / row["median_us"]
                if pytorch_row and row["median_us"] else None
            )
