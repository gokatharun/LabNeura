# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-20T07:12:54.424417+00:00
- Commit: `727e44d`
- CPU: AMD EPYC 7763 64-Core Processor (x86_64)
- OS: Linux-7.0.0-1019-azure-x86_64-with-glibc2.43
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.4.9
- Actual SIMD backend selected: **AVX2**
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.578 | 1.064 | 1.881 | 1.841x | 3.255x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.534 | 1.063 | 1.899 | 1.990x | 3.556x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 493.676 | 741.122 | 755.920 | 1.501x | 1.531x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 411.793 | 772.291 | 733.761 | 1.875x | 1.782x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.205 | 1.211 | 1.677 | 5.907x | 8.181x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.213 | 1.185 | 1.674 | 5.571x | 7.864x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 282.978 | 275.151 | 296.468 | 0.972x | 1.048x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 339.090 | 332.337 | 343.589 | 0.980x | 1.013x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.557 | 1.055 | 1.833 | 1.894x | 3.292x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.533 | 1.038 | 1.827 | 1.949x | 3.430x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 508.574 | 765.568 | 725.376 | 1.505x | 1.426x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 401.627 | 748.218 | 735.965 | 1.863x | 1.832x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.205 | 1.193 | 1.686 | 5.819x | 8.227x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.209 | 1.184 | 1.650 | 5.654x | 7.883x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 286.535 | 284.914 | 285.548 | 0.994x | 0.997x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 342.439 | 281.873 | 305.197 | 0.823x | 0.891x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.558 | 1.029 | 1.818 | 1.843x | 3.255x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.542 | 1.045 | 1.826 | 1.928x | 3.370x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 539.519 | 734.761 | 712.223 | 1.362x | 1.320x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 414.155 | 760.711 | 718.312 | 1.837x | 1.734x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.211 | 1.165 | 1.623 | 5.515x | 7.685x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.205 | 1.165 | 1.623 | 5.670x | 7.904x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 290.818 | 288.901 | 282.943 | 0.993x | 0.973x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 320.768 | 294.870 | 290.091 | 0.919x | 0.904x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.572 | 1.015 | 1.812 | 1.775x | 3.168x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.542 | 1.033 | 1.796 | 1.905x | 3.313x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 491.056 | 729.216 | 738.660 | 1.485x | 1.504x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 399.026 | 733.143 | 752.546 | 1.837x | 1.886x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.194 | 1.178 | 1.647 | 6.081x | 8.498x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.201 | 1.153 | 1.637 | 5.731x | 8.136x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 296.536 | 305.522 | 311.799 | 1.030x | 1.051x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 345.813 | 282.940 | 301.916 | 0.818x | 0.873x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.573 | 8.119 | 8.681 | 14.157x | 15.138x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.586 | 8.443 | 8.568 | 14.414x | 14.629x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 2909.132 | 497.504 | 426.637 | 0.171x | 0.147x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 113.957 | 539.184 | 460.211 | 4.731x | 4.038x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.225 | 8.299 | 6.660 | 36.814x | 29.545x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.220 | 8.563 | 6.617 | 38.854x | 30.024x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 2848.225 | 462.315 | 399.882 | 0.162x | 0.140x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 74.813 | 476.927 | 418.126 | 6.375x | 5.589x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.583 | 8.084 | 8.295 | 13.865x | 14.227x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.591 | 8.045 | 8.362 | 13.610x | 14.147x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 2911.180 | 492.861 | 447.982 | 0.169x | 0.154x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 107.094 | 478.395 | 442.840 | 4.467x | 4.135x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.213 | 8.423 | 6.646 | 39.544x | 31.201x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.209 | 8.172 | 6.713 | 39.025x | 32.057x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 2850.345 | 470.784 | 393.972 | 0.165x | 0.138x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 72.734 | 522.756 | 481.731 | 7.187x | 6.623x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.569 | 8.233 | 8.602 | 14.463x | 15.111x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.595 | 8.094 | 8.548 | 13.610x | 14.373x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4891.392 | 503.916 | 450.694 | 0.103x | 0.092x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 151.153 | 490.061 | 490.416 | 3.242x | 3.245x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.209 | 8.194 | 6.695 | 39.170x | 32.004x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.221 | 8.177 | 6.627 | 36.965x | 29.955x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4837.514 | 458.371 | 404.844 | 0.095x | 0.084x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 103.229 | 468.736 | 394.694 | 4.541x | 3.823x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.563 | 9.075 | 8.378 | 16.112x | 14.873x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.597 | 9.055 | 8.383 | 15.174x | 14.049x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1933.524 | 1462.744 | 2211.552 | 0.757x | 1.144x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 769.997 | 1523.180 | 2204.119 | 1.978x | 2.863x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.211 | 9.114 | 6.582 | 43.155x | 31.166x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.210 | 9.450 | 6.579 | 44.956x | 31.297x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1875.423 | 1489.235 | 2209.234 | 0.794x | 1.178x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 714.442 | 1443.348 | 2193.726 | 2.020x | 3.071x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 1.084 | 1.857 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.588 | 1.084 | 1.847 | 1.845x | 3.141x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 6668.055 | 349.850 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 271.591 | 6755.254 | 350.217 | 24.873x | 1.289x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 1.221 | 1.669 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.208 | 1.208 | 1.667 | 5.815x | 8.024x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 6615.235 | 240.750 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 161.402 | 6648.945 | 254.041 | 41.195x | 1.574x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 1.086 | 1.824 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.591 | 1.080 | 1.814 | 1.828x | 3.070x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 6976.319 | 357.382 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 270.478 | 6974.245 | 350.483 | 25.785x | 1.296x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 1.212 | 1.658 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.212 | 1.207 | 1.664 | 5.694x | 7.850x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 6891.925 | 243.926 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 175.464 | 6920.429 | 245.083 | 39.441x | 1.397x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 1.077 | 1.807 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.594 | 1.089 | 1.800 | 1.832x | 3.028x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 6665.869 | 324.145 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 260.122 | 6670.656 | 322.495 | 25.644x | 1.240x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 1.205 | 1.615 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.221 | 1.213 | 1.610 | 5.500x | 7.296x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 6836.541 | 212.265 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 168.336 | 6644.210 | 217.216 | 39.470x | 1.290x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 1.068 | 1.807 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.587 | 1.062 | 1.793 | 1.809x | 3.053x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 6858.909 | 594.293 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 283.295 | 6653.938 | 320.595 | 23.488x | 1.132x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 1.195 | 1.628 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.226 | 1.204 | 1.630 | 5.336x | 7.224x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 6639.463 | 211.830 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 171.236 | 6568.624 | 217.236 | 38.360x | 1.269x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 7.953 | 8.262 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 1.083 | 10.259 | 8.593 | 9.473x | 7.934x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1148.287 | 1460.658 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 239.024 | 1198.135 | 1419.160 | 5.013x | 5.937x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 9.130 | 11.131 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.219 | 8.149 | 6.668 | 37.176x | 30.422x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1092.212 | 1368.624 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 144.998 | 1094.885 | 1400.326 | 7.551x | 9.658x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 8.133 | 8.464 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.601 | 7.942 | 8.353 | 13.221x | 13.905x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1157.129 | 1699.808 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 246.022 | 1185.959 | 1440.314 | 4.821x | 5.854x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 8.165 | 6.797 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.217 | 8.321 | 6.738 | 38.414x | 31.108x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1010.274 | 1348.052 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 153.492 | 1091.429 | 1295.690 | 7.111x | 8.441x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 7.944 | 8.324 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.605 | 7.996 | 8.358 | 13.228x | 13.826x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1150.000 | 1445.368 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 318.858 | 1283.521 | 1444.412 | 4.025x | 4.530x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 8.112 | 6.642 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.224 | 8.212 | 6.642 | 36.692x | 29.680x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1054.010 | 1347.618 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 198.834 | 1117.896 | 1393.615 | 5.622x | 7.009x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 8.911 | 8.686 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.606 | 9.107 | 8.667 | 15.025x | 14.299x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3063.860 | 2983.094 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 772.424 | 2851.137 | 2932.254 | 3.691x | 3.796x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 9.142 | 6.693 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.226 | 9.111 | 6.572 | 40.348x | 29.103x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 2807.478 | 3009.161 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 643.183 | 2841.670 | 2866.100 | 4.418x | 4.456x |

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
