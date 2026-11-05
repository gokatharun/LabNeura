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

## Existing Python benchmark scripts

`benchmark_runner.py` remains an exploratory comparison with NumPy and
PyTorch. It is not used by CI-generated reports; its operation semantics and
timing boundaries differ from the native backend benchmark.
