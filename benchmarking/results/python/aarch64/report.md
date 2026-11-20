# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-20T07:38:53.139396+00:00
- Commit: `70c975a`
- CPU: aarch64 (aarch64)
- OS: Linux-7.0.0-1017-azure-aarch64-with-glibc2.39
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.4.9
- Actual SIMD backend selected: **NEON**
- Correctness-checked results: 4752 passed; 432 explicitly unsupported

## Method

Sizes: 7, 8, 9, 10, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256, 512, 1,024, 4,096, 16,384, 65,536, 262,144, 1,000,000 elements. Inputs are generated deterministically from the recorded seed, and correctness is checked before each timed case. The primary metric is the median of independent batch averages; IQR and p95 are also included in the CSV and JSON.

Allocating timings include the operation and output allocation. In-place timings include only the API operation; input objects are prepared outside each timed batch and each receives one operation, so repeated timing does not mutate and reuse the same input. Input generation and dtype conversion are outside timing. Input-object construction is measured separately. These operation timings are end-to-end Python API measurements, not isolated native kernel timings.

INT8 and INT16 use saturating add/subtract/multiply and truncating division in all libraries; division denominators are nonzero. FP16 operands and results use 16-bit floating-point values. Generic FP16/INT16 rows are explicitly unsupported because the Generic backend has no kernels for those dtypes. NumPy/PyTorch rows are measured once per comparison backend context for consistent tables.

The repeated-input working set is cache-resident. This report does not claim streaming/DRAM bandwidth performance. NumPy and PyTorch run on CPU with one thread. See `benchmark.json` for the full metadata and `benchmark.csv` for every case.

The native C++ benchmark is a separate layer and remains documented in the [benchmark README](../../README.md); native timings must not be compared directly with the Python/API results here.

## Sample comparison points

| Dtype | Op | Layer | Size | Backend | LabNeura (µs) | NumPy (µs) | PyTorch (µs) | Lab/NumPy | Lab/PyTorch |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| FP32 | ADD | allocating | 7 | GENERIC | 0.605 | 0.790 | 1.383 | 1.305x | 2.285x |
| FP32 | ADD | allocating | 7 | NEON | 0.604 | 0.793 | 1.373 | 1.313x | 2.275x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 466.595 | 497.697 | 322.510 | 1.067x | 0.691x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 397.979 | 553.961 | 266.034 | 1.392x | 0.668x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.230 | 0.925 | 1.240 | 4.027x | 5.396x |
| FP32 | ADD | inplace | 7 | NEON | 0.261 | 0.922 | 1.231 | 3.537x | 4.723x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 352.807 | 281.447 | 251.356 | 0.798x | 0.712x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 338.006 | 237.675 | 243.982 | 0.703x | 0.722x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.598 | 0.783 | 1.376 | 1.310x | 2.300x |
| FP32 | SUB | allocating | 7 | NEON | 0.582 | 0.787 | 1.383 | 1.352x | 2.378x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 462.116 | 483.053 | 270.864 | 1.045x | 0.586x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 400.331 | 545.295 | 299.919 | 1.362x | 0.749x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.244 | 0.923 | 1.305 | 3.791x | 5.357x |
| FP32 | SUB | inplace | 7 | NEON | 0.262 | 0.933 | 1.297 | 3.560x | 4.951x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 344.431 | 235.825 | 237.124 | 0.685x | 0.688x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 358.443 | 235.461 | 242.387 | 0.657x | 0.676x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.602 | 0.789 | 1.364 | 1.310x | 2.265x |
| FP32 | MUL | allocating | 7 | NEON | 0.591 | 0.787 | 1.376 | 1.332x | 2.328x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 462.052 | 508.448 | 280.973 | 1.100x | 0.608x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 378.898 | 493.242 | 267.300 | 1.302x | 0.705x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.240 | 0.933 | 1.229 | 3.886x | 5.118x |
| FP32 | MUL | inplace | 7 | NEON | 0.273 | 0.941 | 1.229 | 3.449x | 4.501x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 336.275 | 219.489 | 216.865 | 0.653x | 0.645x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 436.474 | 226.939 | 232.959 | 0.520x | 0.534x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.593 | 0.773 | 1.374 | 1.303x | 2.316x |
| FP32 | DIV | allocating | 7 | NEON | 0.590 | 0.775 | 1.370 | 1.312x | 2.321x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 681.167 | 730.304 | 483.927 | 1.072x | 0.710x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 475.002 | 733.978 | 484.433 | 1.545x | 1.020x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.240 | 0.911 | 1.433 | 3.802x | 5.984x |
| FP32 | DIV | inplace | 7 | NEON | 0.274 | 0.908 | 1.526 | 3.317x | 5.579x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 465.530 | 468.698 | 472.336 | 1.007x | 1.015x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 465.552 | 462.170 | 470.789 | 0.993x | 1.011x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.607 | 6.483 | 8.963 | 10.679x | 14.764x |
| INT8 | ADD | allocating | 7 | NEON | 0.667 | 6.324 | 8.945 | 9.476x | 13.403x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1014.389 | 549.522 | 586.763 | 0.542x | 0.578x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 117.631 | 547.458 | 591.309 | 4.654x | 5.027x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.238 | 6.417 | 6.818 | 26.934x | 28.617x |
| INT8 | ADD | inplace | 7 | NEON | 0.271 | 6.367 | 7.194 | 23.490x | 26.541x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 956.356 | 553.049 | 594.569 | 0.578x | 0.622x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 88.695 | 530.679 | 581.803 | 5.983x | 6.560x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.604 | 6.302 | 10.273 | 10.431x | 17.003x |
| INT8 | SUB | allocating | 7 | NEON | 0.669 | 6.342 | 9.683 | 9.476x | 14.467x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1015.732 | 550.971 | 593.165 | 0.542x | 0.584x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 112.171 | 550.629 | 592.265 | 4.909x | 5.280x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.248 | 6.360 | 7.907 | 25.677x | 31.920x |
| INT8 | SUB | inplace | 7 | NEON | 0.270 | 6.333 | 7.493 | 23.476x | 27.773x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 955.689 | 533.081 | 582.924 | 0.558x | 0.610x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 87.425 | 529.324 | 584.077 | 6.055x | 6.681x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.612 | 6.362 | 8.135 | 10.395x | 13.292x |
| INT8 | MUL | allocating | 7 | NEON | 0.673 | 6.317 | 8.230 | 9.386x | 12.229x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 969.282 | 546.665 | 588.338 | 0.564x | 0.607x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 120.926 | 551.717 | 580.250 | 4.562x | 4.798x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.243 | 6.343 | 6.597 | 26.114x | 27.161x |
| INT8 | MUL | inplace | 7 | NEON | 0.287 | 6.345 | 6.608 | 22.142x | 23.057x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 911.328 | 528.095 | 566.173 | 0.579x | 0.621x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 87.442 | 533.011 | 566.552 | 6.096x | 6.479x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.616 | 7.145 | 9.246 | 11.593x | 15.002x |
| INT8 | DIV | allocating | 7 | NEON | 0.665 | 7.158 | 9.137 | 10.769x | 13.747x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1677.075 | 2735.607 | 2062.031 | 1.631x | 1.230x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 570.479 | 2786.766 | 2054.504 | 4.885x | 3.601x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.241 | 7.086 | 9.533 | 29.429x | 39.588x |
| INT8 | DIV | inplace | 7 | NEON | 0.273 | 7.137 | 8.655 | 26.117x | 31.671x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1618.257 | 2738.054 | 2048.620 | 1.692x | 1.266x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 508.386 | 2739.175 | 2045.290 | 5.388x | 4.023x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.804 | 1.358 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.676 | 0.802 | 1.419 | 1.187x | 2.100x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1967.926 | 134.733 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1004.595 | 1954.394 | 128.512 | 1.945x | 0.128x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.955 | 1.238 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.273 | 0.938 | 1.277 | 3.437x | 4.678x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1886.917 | 115.694 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 898.608 | 1880.653 | 113.133 | 2.093x | 0.126x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.810 | 1.372 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.667 | 0.809 | 1.393 | 1.213x | 2.089x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1973.929 | 134.472 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1008.362 | 1976.304 | 133.042 | 1.960x | 0.132x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.941 | 1.329 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.279 | 0.942 | 1.354 | 3.372x | 4.846x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1897.633 | 115.870 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 895.817 | 1898.497 | 112.382 | 2.119x | 0.125x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.807 | 1.368 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.672 | 0.805 | 1.360 | 1.198x | 2.024x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1966.943 | 132.046 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1004.229 | 1973.519 | 139.070 | 1.965x | 0.138x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.937 | 1.216 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.275 | 0.940 | 1.213 | 3.422x | 4.418x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1895.131 | 115.344 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 895.164 | 1901.503 | 115.479 | 2.124x | 0.129x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.791 | 1.365 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.665 | 0.792 | 1.383 | 1.191x | 2.079x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2183.039 | 356.337 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1301.496 | 2183.556 | 355.004 | 1.678x | 0.273x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.921 | 1.568 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.265 | 0.924 | 1.513 | 3.488x | 5.708x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2100.142 | 348.087 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1190.754 | 2101.559 | 348.758 | 1.765x | 0.293x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.258 | 8.358 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.671 | 6.299 | 9.186 | 9.382x | 13.682x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1333.355 | 1489.100 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 246.113 | 1339.175 | 1505.649 | 5.441x | 6.118x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.204 | 6.580 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.276 | 6.287 | 6.505 | 22.778x | 23.570x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1254.410 | 1481.236 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 178.700 | 1243.254 | 1423.155 | 6.957x | 7.964x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.224 | 9.366 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.675 | 6.220 | 8.824 | 9.211x | 13.068x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1344.817 | 1433.221 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 248.776 | 1351.383 | 1473.052 | 5.432x | 5.921x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.219 | 7.567 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.282 | 6.274 | 7.654 | 22.230x | 27.119x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1269.015 | 1463.740 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 186.426 | 1248.174 | 1455.198 | 6.695x | 7.806x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.291 | 8.465 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.657 | 6.233 | 8.012 | 9.480x | 12.186x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1399.060 | 1412.968 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 249.266 | 1411.357 | 1350.798 | 5.662x | 5.419x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.222 | 6.735 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.265 | 6.233 | 6.813 | 23.498x | 25.682x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1390.158 | 1398.298 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 177.196 | 1366.780 | 1363.061 | 7.713x | 7.692x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.873 | 8.694 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.648 | 6.877 | 10.473 | 10.613x | 16.162x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3765.122 | 2663.618 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 614.908 | 3884.038 | 2690.823 | 6.316x | 4.376x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.925 | 7.417 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.286 | 6.961 | 9.681 | 24.318x | 33.820x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3740.027 | 2653.819 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 513.866 | 3716.487 | 2651.803 | 7.232x | 5.160x |

## Visualization

- `fp32_add.png`
- `fp32_sub.png`
- `fp32_mul.png`
- `fp32_div.png`
- `int8_add.png`
- `int8_sub.png`
- `int8_mul.png`
- `int8_div.png`
- `fp16_add.png`
- `fp16_sub.png`
- `fp16_mul.png`
- `fp16_div.png`
- `int16_add.png`
- `int16_sub.png`
- `int16_mul.png`
- `int16_div.png`
