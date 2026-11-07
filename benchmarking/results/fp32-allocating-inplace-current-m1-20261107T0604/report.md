# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-07T00:34:41.775252+00:00
- Commit: `a509bbb`
- CPU: Apple M1 (arm64)
- OS: macOS-27.0.1-arm64-arm-64bit
- Python / NumPy / PyTorch: 3.9.6 / 2.0.2 / 2.8.0
- LabNeura: 0.4.1
- Actual SIMD backend selected: **NEON**
- Correctness-checked results: 324 passed; 0 explicitly unsupported

## Method

Sizes: 1,024, 4,096, 16,384, 65,536, 262,144, 1,000,000 elements. Inputs are generated deterministically from the recorded seed, and correctness is checked before each timed case. The primary metric is the median of independent batch averages; IQR and p95 are also included in the CSV and JSON.

Allocating timings include the operation and output allocation. In-place timings include only the API operation; input objects are prepared outside each timed batch and each receives one operation, so repeated timing does not mutate and reuse the same input. Input generation and dtype conversion are outside timing. Input-object construction is measured separately. These operation timings are end-to-end Python API measurements, not isolated native kernel timings.

INT8 and INT16 use saturating add/subtract/multiply and truncating division in all libraries; division denominators are nonzero. FP16 operands and results use 16-bit floating-point values. Generic FP16/INT16 rows are explicitly unsupported because the Generic backend has no kernels for those dtypes. NumPy/PyTorch rows are measured once per comparison backend context for consistent tables.

The repeated-input working set is cache-resident. This report does not claim streaming/DRAM bandwidth performance. NumPy and PyTorch run on CPU with one thread. See `benchmark.json` for the full metadata and `benchmark.csv` for every case.

The native C++ benchmark is a separate layer and remains documented in the [benchmark README](../../README.md); native timings must not be compared directly with the Python/API results here.

## Sample comparison points

| Dtype | Op | Layer | Size | Backend | LabNeura (µs) | NumPy (µs) | PyTorch (µs) | Lab/NumPy | Lab/PyTorch |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| FP32 | ADD | allocating | 1,024 | GENERIC | 0.551 | 0.822 | 1.214 | 1.492x | 2.205x |
| FP32 | ADD | allocating | 1,024 | NEON | 0.521 | 0.801 | 1.171 | 1.539x | 2.249x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 229.714 | 215.240 | 193.385 | 0.937x | 0.842x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 219.901 | 190.182 | 238.341 | 0.865x | 1.084x |
| FP32 | ADD | inplace | 1,024 | GENERIC | 0.259 | 0.863 | 1.068 | 3.332x | 4.123x |
| FP32 | ADD | inplace | 1,024 | NEON | 0.279 | 0.904 | 0.988 | 3.235x | 3.538x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 214.990 | 218.378 | 221.794 | 1.016x | 1.032x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 218.690 | 215.987 | 219.461 | 0.988x | 1.004x |
| FP32 | SUB | allocating | 1,024 | GENERIC | 0.599 | 0.792 | 1.203 | 1.322x | 2.009x |
| FP32 | SUB | allocating | 1,024 | NEON | 0.568 | 0.843 | 1.202 | 1.483x | 2.115x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 222.969 | 207.836 | 219.513 | 0.932x | 0.985x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 209.862 | 199.526 | 229.328 | 0.951x | 1.093x |
| FP32 | SUB | inplace | 1,024 | GENERIC | 0.283 | 0.923 | 1.062 | 3.267x | 3.758x |
| FP32 | SUB | inplace | 1,024 | NEON | 0.294 | 0.889 | 1.038 | 3.027x | 3.537x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 219.031 | 224.063 | 212.583 | 1.023x | 0.971x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 211.896 | 214.315 | 213.099 | 1.011x | 1.006x |
| FP32 | MUL | allocating | 1,024 | GENERIC | 0.593 | 0.821 | 1.214 | 1.384x | 2.046x |
| FP32 | MUL | allocating | 1,024 | NEON | 0.548 | 0.858 | 1.201 | 1.565x | 2.190x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 217.469 | 189.453 | 182.784 | 0.871x | 0.841x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 227.026 | 205.661 | 187.557 | 0.906x | 0.826x |
| FP32 | MUL | inplace | 1,024 | GENERIC | 0.290 | 0.924 | 1.042 | 3.182x | 3.590x |
| FP32 | MUL | inplace | 1,024 | NEON | 0.281 | 0.923 | 1.041 | 3.280x | 3.701x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 218.581 | 225.781 | 214.456 | 1.033x | 0.981x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 212.487 | 226.109 | 229.841 | 1.064x | 1.082x |
| FP32 | DIV | allocating | 1,024 | GENERIC | 0.598 | 0.846 | 1.213 | 1.413x | 2.027x |
| FP32 | DIV | allocating | 1,024 | NEON | 0.529 | 0.849 | 1.215 | 1.604x | 2.295x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 234.344 | 276.385 | 235.943 | 1.179x | 1.007x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 222.120 | 212.042 | 189.521 | 0.955x | 0.853x |
| FP32 | DIV | inplace | 1,024 | GENERIC | 0.303 | 0.919 | 1.061 | 3.034x | 3.503x |
| FP32 | DIV | inplace | 1,024 | NEON | 0.297 | 0.917 | 1.040 | 3.090x | 3.504x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 232.474 | 210.948 | 219.768 | 0.907x | 0.945x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 218.763 | 216.599 | 213.362 | 0.990x | 0.975x |

## Visualization

- `fp32_add.png`
- `fp32_sub.png`
- `fp32_mul.png`
- `fp32_div.png`
