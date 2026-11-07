# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-06T15:50:28.051157+00:00
- Commit: `a509bbb`
- CPU: Apple M1 (arm64)
- OS: macOS-27.0.1-arm64-arm-64bit
- Python / NumPy / PyTorch: 3.9.6 / 2.0.2 / 2.8.0
- LabNeura: 0.4.1
- Actual SIMD backend selected: **NEON**
- Correctness-checked results: 648 passed; 0 explicitly unsupported

## Method

Sizes: 10, 32, 64, 128, 256, 512, 1,024, 4,096, 16,384, 65,536, 262,144, 1,000,000 elements. Inputs are generated deterministically from the recorded seed, and correctness is checked before each timed case. The primary metric is the median of independent batch averages; IQR and p95 are also included in the CSV and JSON.

Allocating timings include the operation and output allocation. In-place timings include only the API operation; input objects are prepared outside each timed batch and each receives one operation, so repeated timing does not mutate and reuse the same input. Input generation and dtype conversion are outside timing. Input-object construction is measured separately. These operation timings are end-to-end Python API measurements, not isolated native kernel timings.

INT8 and INT16 use saturating add/subtract/multiply and truncating division in all libraries; division denominators are nonzero. FP16 operands and results use 16-bit floating-point values. Generic FP16/INT16 rows are explicitly unsupported because the Generic backend has no kernels for those dtypes. NumPy/PyTorch rows are measured once per comparison backend context for consistent tables.

The repeated-input working set is cache-resident. This report does not claim streaming/DRAM bandwidth performance. NumPy and PyTorch run on CPU with one thread. See `benchmark.json` for the full metadata and `benchmark.csv` for every case.

The native C++ benchmark is a separate layer and remains documented in the [benchmark README](../../README.md); native timings must not be compared directly with the Python/API results here.

## Sample comparison points

| Dtype | Op | Layer | Size | Backend | LabNeura (µs) | NumPy (µs) | PyTorch (µs) | Lab/NumPy | Lab/PyTorch |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| FP32 | ADD | allocating | 10 | GENERIC | 0.361 | 0.646 | 1.048 | 1.791x | 2.904x |
| FP32 | ADD | allocating | 10 | NEON | 0.354 | 0.656 | 1.025 | 1.851x | 2.895x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 235.354 | 184.083 | 187.063 | 0.782x | 0.795x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 228.266 | 183.130 | 184.922 | 0.802x | 0.810x |
| FP32 | ADD | inplace | 10 | GENERIC | 0.132 | 0.723 | 0.907 | 5.495x | 6.897x |
| FP32 | ADD | inplace | 10 | NEON | 0.130 | 0.682 | 0.926 | 5.267x | 7.146x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 221.536 | 211.714 | 215.932 | 0.956x | 0.975x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 217.620 | 204.990 | 209.047 | 0.942x | 0.961x |
| FP32 | SUB | allocating | 10 | GENERIC | 0.362 | 0.604 | 1.094 | 1.667x | 3.022x |
| FP32 | SUB | allocating | 10 | NEON | 0.355 | 0.618 | 1.076 | 1.743x | 3.031x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 229.406 | 178.672 | 184.396 | 0.779x | 0.804x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 230.813 | 178.193 | 183.927 | 0.772x | 0.797x |
| FP32 | SUB | inplace | 10 | GENERIC | 0.128 | 0.676 | 0.943 | 5.301x | 7.393x |
| FP32 | SUB | inplace | 10 | NEON | 0.133 | 0.693 | 0.940 | 5.216x | 7.078x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 218.745 | 208.734 | 208.927 | 0.954x | 0.955x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 219.552 | 221.250 | 248.057 | 1.008x | 1.130x |
| FP32 | MUL | allocating | 10 | GENERIC | 0.368 | 0.617 | 1.094 | 1.675x | 2.970x |
| FP32 | MUL | allocating | 10 | NEON | 0.353 | 0.618 | 1.038 | 1.753x | 2.941x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 233.318 | 194.896 | 203.438 | 0.835x | 0.872x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 225.625 | 183.458 | 182.302 | 0.813x | 0.808x |
| FP32 | MUL | inplace | 10 | GENERIC | 0.133 | 0.679 | 0.937 | 5.088x | 7.020x |
| FP32 | MUL | inplace | 10 | NEON | 0.129 | 0.680 | 0.897 | 5.278x | 6.960x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 235.479 | 205.276 | 208.151 | 0.872x | 0.884x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 214.984 | 204.859 | 207.437 | 0.953x | 0.965x |
| FP32 | DIV | allocating | 10 | GENERIC | 0.363 | 0.604 | 1.100 | 1.661x | 3.029x |
| FP32 | DIV | allocating | 10 | NEON | 0.354 | 0.644 | 1.053 | 1.818x | 2.972x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 233.573 | 178.443 | 181.792 | 0.764x | 0.778x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 227.651 | 190.766 | 194.010 | 0.838x | 0.852x |
| FP32 | DIV | inplace | 10 | GENERIC | 0.135 | 0.702 | 0.933 | 5.183x | 6.889x |
| FP32 | DIV | inplace | 10 | NEON | 0.130 | 0.676 | 0.895 | 5.221x | 6.905x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 216.739 | 260.625 | 220.703 | 1.202x | 1.018x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 227.474 | 220.141 | 223.458 | 0.968x | 0.982x |

## Visualization

- `fp32_add.png`
- `fp32_sub.png`
- `fp32_mul.png`
- `fp32_div.png`
