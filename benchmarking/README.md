# LabNeura CPU Backend Benchmarks

The GitHub Actions workflow runs native C++ benchmarks on Linux x86-64 and
ARM64. Each runner compares the Generic backend with the available AVX2 or NEON
backend for FP32 and INT8 ADD, SUB, MUL, and DIV. Results are published to
[`results/latest.md`](results/latest.md) after pushes to `main` and manual
workflow runs on `main`.

The benchmark checks every result against independent scalar expected values
before timing. Tensor allocation and initial input copies are excluded from the
timed interval. The reported latency includes the backend operation dispatch
and kernel. Each entry is the median or p95 across 31 samples after 5 warmups.
INT8 uses saturating arithmetic, and division inputs are nonzero.

GitHub-hosted runner hardware can change between runs. The report is intended
for same-architecture observations, not as a stable performance guarantee or a
cross-architecture ranking. It does not compare against NumPy or PyTorch.

## Run locally

Configure and build the native test and benchmark targets:

```sh
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/native --parallel
ctest --test-dir build/native --output-on-failure
```

Run the benchmark on an AVX2-capable x86-64 host or an ARM64 host with NEON:

```sh
build/native/labneura_native_backend_benchmark benchmark-local.csv
```

The benchmark exits with an error if no supported SIMD backend is available;
it does not report Generic-only measurements as a SIMD comparison.

## Python/API comparison with NumPy and PyTorch

The Python runner is separate from the native C++ benchmark above. It compares
the Generic backend and the SIMD backend actually selected by the current CPU
with NumPy and PyTorch CPU for FP32, saturating INT8, FP16, and saturating INT16
ADD, SUB, MUL, and DIV. Generic FP16/INT16 cases are marked unsupported rather
than reported as measurements.

Benchmark dependencies (PyTorch, Matplotlib, threadpoolctl) are deliberately
not part of the LabNeura package. GitHub Actions installs them from
`benchmarking/requirements.txt` in the `python-benchmarks` job; locally, install
LabNeura from this checkout and the extra requirements only when benchmarking:

```sh
python -m pip install .
python -m pip install -r benchmarking/requirements.txt
python benchmarking/run_benchmarks.py --smoke
python benchmarking/run_benchmarks.py
```

The full run includes 10 through 1,000,000 elements, deterministic shared
inputs, correctness checks before timing, independent timing batches, median,
IQR, and p95. It measures input-object construction separately and separates
allocating operations (including output allocation) from in-place operations.
Input generation and dtype conversion are outside operation timings. Results
are saved in a unique directory under
`benchmarking/results/` as CSV, JSON, Markdown, and plots. Existing native and
Python result files are never overwritten. Use `--output-dir` to select a new
directory, `--sizes` to select test sizes, or `--threads` to set the CPU thread
count. Existing output directories are rejected.

The Python measurements are end-to-end Python API timings, not isolated native
kernel timings. NumPy/PyTorch run on CPU with one thread by default. The current
methodology repeatedly uses the same deterministic working set, so large cases
are cache-resident and do not represent streaming DRAM bandwidth. The report
records the selected backend and build metadata. It does not claim an AVX2
runtime result unless the automatic backend on the host selected AVX2.

The older exploratory scripts (`benchmark_int16_fp16.py`,
`benchmark_results.json`, and `examples/benchmark_numpy.py`) are retained for
historical context and are not inputs to the new report.
