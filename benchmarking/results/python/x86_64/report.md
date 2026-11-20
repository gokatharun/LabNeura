# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-20T06:46:12.228618+00:00
- Commit: `f78502b`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.564 | 1.038 | 1.876 | 1.840x | 3.327x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.535 | 1.036 | 1.839 | 1.935x | 3.435x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 499.960 | 771.151 | 741.728 | 1.542x | 1.484x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 337.507 | 793.177 | 723.021 | 2.350x | 2.142x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.204 | 1.177 | 1.686 | 5.770x | 8.266x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.213 | 1.185 | 1.686 | 5.560x | 7.907x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 277.938 | 293.249 | 302.740 | 1.055x | 1.089x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 331.952 | 260.704 | 276.688 | 0.785x | 0.834x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.558 | 1.034 | 1.834 | 1.854x | 3.287x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.529 | 1.033 | 1.866 | 1.951x | 3.525x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 436.704 | 701.809 | 723.024 | 1.607x | 1.656x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 386.015 | 718.715 | 740.972 | 1.862x | 1.920x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.207 | 1.191 | 1.681 | 5.748x | 8.111x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.203 | 1.176 | 1.678 | 5.790x | 8.256x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 273.708 | 278.540 | 268.661 | 1.018x | 0.982x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 312.418 | 269.658 | 267.231 | 0.863x | 0.855x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.556 | 1.034 | 1.797 | 1.859x | 3.232x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.540 | 1.041 | 1.804 | 1.928x | 3.339x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 439.417 | 784.145 | 731.394 | 1.785x | 1.664x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 404.865 | 880.181 | 764.358 | 2.174x | 1.888x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.197 | 1.179 | 1.624 | 5.987x | 8.244x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.203 | 1.181 | 1.627 | 5.815x | 8.006x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 271.386 | 282.535 | 303.176 | 1.041x | 1.117x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 292.969 | 278.747 | 274.943 | 0.951x | 0.938x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.564 | 1.031 | 1.788 | 1.829x | 3.171x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.539 | 1.018 | 1.778 | 1.889x | 3.300x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 477.265 | 717.054 | 415.623 | 1.502x | 0.871x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 384.530 | 698.989 | 379.382 | 1.818x | 0.987x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.190 | 1.174 | 1.640 | 6.174x | 8.625x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.198 | 1.167 | 1.627 | 5.889x | 8.211x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 274.752 | 260.210 | 274.272 | 0.947x | 0.998x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 313.189 | 245.037 | 254.342 | 0.782x | 0.812x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.588 | 8.027 | 8.515 | 13.659x | 14.490x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.588 | 8.066 | 8.481 | 13.705x | 14.412x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 2902.817 | 507.375 | 434.482 | 0.175x | 0.150x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 106.461 | 500.001 | 436.486 | 4.697x | 4.100x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.210 | 8.110 | 6.733 | 38.691x | 32.124x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.212 | 8.034 | 6.695 | 37.900x | 31.584x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 2853.957 | 528.561 | 395.667 | 0.185x | 0.139x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 69.587 | 554.545 | 429.872 | 7.969x | 6.178x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.563 | 8.069 | 8.635 | 14.342x | 15.347x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.591 | 8.060 | 8.398 | 13.640x | 14.212x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 2901.128 | 556.434 | 454.022 | 0.192x | 0.156x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 102.090 | 546.633 | 534.422 | 5.354x | 5.235x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.208 | 8.314 | 6.709 | 39.938x | 32.225x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.207 | 8.108 | 6.674 | 39.172x | 32.245x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 2849.705 | 520.722 | 455.193 | 0.183x | 0.160x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 70.675 | 499.595 | 440.371 | 7.069x | 6.231x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.564 | 8.044 | 8.472 | 14.272x | 15.032x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.601 | 8.116 | 8.218 | 13.506x | 13.675x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4896.646 | 528.682 | 425.909 | 0.108x | 0.087x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 146.811 | 551.691 | 407.688 | 3.758x | 2.777x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.208 | 8.092 | 6.500 | 38.947x | 31.285x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.214 | 7.999 | 6.463 | 37.415x | 30.229x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4840.625 | 529.215 | 442.533 | 0.109x | 0.091x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 100.633 | 487.479 | 412.761 | 4.844x | 4.102x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.578 | 9.112 | 8.541 | 15.757x | 14.770x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.597 | 9.041 | 8.432 | 15.156x | 14.136x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1926.342 | 1469.631 | 2213.259 | 0.763x | 1.149x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 766.791 | 1474.748 | 2217.239 | 1.923x | 2.892x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.213 | 9.124 | 6.494 | 42.837x | 30.488x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.209 | 9.126 | 6.554 | 43.706x | 31.389x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1874.459 | 1440.017 | 2187.917 | 0.768x | 1.167x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 714.439 | 1452.434 | 2182.578 | 2.033x | 3.055x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 1.083 | 1.878 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.588 | 1.077 | 1.947 | 1.832x | 3.310x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 6701.048 | 352.368 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 256.811 | 6653.393 | 349.412 | 25.908x | 1.361x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 1.225 | 1.720 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.206 | 1.229 | 1.727 | 5.961x | 8.375x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 6585.217 | 240.833 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 152.458 | 6574.282 | 246.789 | 43.122x | 1.619x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 1.087 | 1.897 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.589 | 1.075 | 1.874 | 1.824x | 3.180x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 7003.060 | 352.291 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 256.224 | 6966.488 | 357.209 | 27.189x | 1.394x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 1.236 | 1.739 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.208 | 1.213 | 1.736 | 5.839x | 8.356x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 6953.227 | 245.148 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 152.302 | 6881.755 | 241.986 | 45.185x | 1.589x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 1.099 | 1.810 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.593 | 1.103 | 1.834 | 1.862x | 3.095x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 6687.124 | 313.927 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 247.569 | 6659.526 | 335.213 | 26.900x | 1.354x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 1.239 | 1.643 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.211 | 1.230 | 1.619 | 5.821x | 7.658x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 6574.976 | 206.976 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 155.271 | 6588.890 | 211.507 | 42.435x | 1.362x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 1.078 | 1.796 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.585 | 1.070 | 1.798 | 1.830x | 3.076x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 6677.779 | 309.897 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 280.303 | 6675.284 | 319.033 | 23.815x | 1.138x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 1.208 | 1.638 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.209 | 1.217 | 1.666 | 5.829x | 7.980x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 6646.856 | 206.079 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 165.092 | 6636.278 | 206.538 | 40.198x | 1.251x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 8.171 | 8.590 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.600 | 7.995 | 8.389 | 13.336x | 13.994x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1140.595 | 1415.517 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 230.469 | 1211.950 | 1415.778 | 5.259x | 6.143x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 7.922 | 6.646 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.214 | 8.226 | 6.693 | 38.369x | 31.218x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1064.294 | 1326.936 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 134.462 | 1183.072 | 1259.678 | 8.799x | 9.368x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 7.970 | 8.350 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.602 | 8.118 | 8.708 | 13.487x | 14.467x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1054.194 | 1368.871 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 226.527 | 1148.273 | 1355.378 | 5.069x | 5.983x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 8.095 | 6.592 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.211 | 8.182 | 6.623 | 38.779x | 31.390x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 996.356 | 1298.374 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 135.112 | 1040.518 | 1292.115 | 7.701x | 9.563x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 7.926 | 8.198 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.606 | 8.108 | 8.385 | 13.386x | 13.843x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1149.012 | 1445.905 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 299.930 | 1145.632 | 1518.808 | 3.820x | 5.064x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 7.963 | 6.557 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.214 | 8.029 | 6.585 | 37.520x | 30.775x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1042.610 | 1350.600 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 190.606 | 1203.560 | 1353.928 | 6.314x | 7.103x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 8.963 | 8.254 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.598 | 8.972 | 8.416 | 15.001x | 14.071x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 2811.338 | 2929.082 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 762.887 | 2705.258 | 2922.407 | 3.546x | 3.831x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 9.200 | 6.511 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.217 | 9.153 | 6.504 | 42.104x | 29.918x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 2825.927 | 2927.613 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 642.945 | 2748.993 | 2902.338 | 4.276x | 4.514x |

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
