# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-07T07:16:16.341374+00:00
- Commit: `f94f6a8`
- CPU: AMD EPYC 9V45 96-Core Processor (x86_64)
- OS: Linux-7.0.0-1017-azure-x86_64-with-glibc2.39
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.4.1
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.413 | 0.643 | 1.154 | 1.559x | 2.797x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.335 | 0.607 | 1.138 | 1.811x | 3.396x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 297.802 | 557.621 | 576.577 | 1.872x | 1.936x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 234.789 | 571.219 | 554.522 | 2.433x | 2.362x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.111 | 0.726 | 1.046 | 6.531x | 9.408x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.137 | 0.737 | 1.048 | 5.396x | 7.675x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 215.180 | 200.888 | 200.234 | 0.934x | 0.931x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 218.379 | 200.695 | 212.448 | 0.919x | 0.973x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.422 | 0.649 | 1.150 | 1.535x | 2.721x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.329 | 0.609 | 1.123 | 1.849x | 3.409x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 311.199 | 564.542 | 571.880 | 1.814x | 1.838x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 244.036 | 531.900 | 555.393 | 2.180x | 2.276x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.112 | 0.707 | 1.032 | 6.291x | 9.184x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.133 | 0.703 | 1.025 | 5.277x | 7.695x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 211.489 | 200.229 | 204.619 | 0.947x | 0.968x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 208.690 | 201.812 | 198.457 | 0.967x | 0.951x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.337 | 0.597 | 1.084 | 1.773x | 3.220x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.333 | 0.597 | 1.087 | 1.795x | 3.267x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 300.691 | 561.505 | 574.705 | 1.867x | 1.911x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 245.270 | 558.851 | 560.651 | 2.279x | 2.286x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.112 | 0.703 | 0.998 | 6.255x | 8.885x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.132 | 0.701 | 0.997 | 5.321x | 7.562x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 209.306 | 208.855 | 201.714 | 0.998x | 0.964x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 224.486 | 215.946 | 202.113 | 0.962x | 0.900x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.335 | 0.594 | 1.090 | 1.774x | 3.258x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.329 | 0.592 | 1.094 | 1.795x | 3.320x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 330.606 | 568.821 | 566.480 | 1.721x | 1.713x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 278.240 | 569.509 | 524.754 | 2.047x | 1.886x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.112 | 0.685 | 1.005 | 6.093x | 8.942x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.133 | 0.689 | 1.000 | 5.171x | 7.504x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 208.192 | 192.473 | 192.298 | 0.924x | 0.924x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 226.317 | 177.853 | 184.343 | 0.786x | 0.815x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.323 | 4.094 | 4.088 | 12.680x | 12.660x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.360 | 4.075 | 4.057 | 11.326x | 11.276x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 2687.208 | 300.482 | 264.969 | 0.112x | 0.099x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 64.559 | 298.490 | 260.375 | 4.623x | 4.033x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.112 | 4.172 | 3.484 | 37.132x | 31.008x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.131 | 4.170 | 3.504 | 31.930x | 26.828x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 2658.682 | 282.491 | 239.347 | 0.106x | 0.090x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 49.900 | 286.182 | 239.383 | 5.735x | 4.797x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.323 | 4.160 | 4.131 | 12.867x | 12.778x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.361 | 4.141 | 4.055 | 11.461x | 11.222x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 2640.829 | 306.335 | 270.363 | 0.116x | 0.102x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 64.137 | 307.750 | 265.257 | 4.798x | 4.136x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.113 | 4.294 | 3.549 | 38.009x | 31.414x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.130 | 4.217 | 3.511 | 32.336x | 26.921x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 2593.975 | 281.858 | 242.244 | 0.109x | 0.093x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 50.595 | 286.297 | 247.095 | 5.659x | 4.884x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.324 | 4.134 | 4.078 | 12.756x | 12.584x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.366 | 4.160 | 4.146 | 11.360x | 11.324x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4250.465 | 320.876 | 268.983 | 0.075x | 0.063x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 76.221 | 299.983 | 277.720 | 3.936x | 3.644x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.111 | 4.239 | 3.518 | 38.201x | 31.702x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.134 | 4.243 | 3.471 | 31.759x | 25.979x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4188.812 | 287.080 | 250.038 | 0.069x | 0.060x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 53.129 | 277.997 | 244.857 | 5.232x | 4.609x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.323 | 4.645 | 3.988 | 14.360x | 12.328x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.366 | 4.645 | 4.025 | 12.692x | 11.000x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1447.384 | 962.508 | 1598.630 | 0.665x | 1.104x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 468.161 | 999.969 | 1623.028 | 2.136x | 3.467x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.113 | 4.700 | 3.420 | 41.682x | 30.333x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.134 | 4.763 | 3.470 | 35.648x | 25.973x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1413.556 | 1032.080 | 1625.950 | 0.730x | 1.150x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 437.288 | 951.269 | 1602.885 | 2.175x | 3.666x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.616 | 1.104 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.391 | 0.617 | 1.126 | 1.577x | 2.881x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 3681.049 | 220.464 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 147.716 | 3633.398 | 219.553 | 24.597x | 1.486x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.712 | 1.005 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.135 | 0.722 | 1.022 | 5.362x | 7.595x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 3605.905 | 110.220 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 99.680 | 3563.975 | 107.903 | 35.754x | 1.082x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.623 | 1.104 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.372 | 0.614 | 1.108 | 1.649x | 2.976x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 3622.224 | 213.955 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 147.156 | 3610.309 | 207.676 | 24.534x | 1.411x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.714 | 1.015 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.132 | 0.708 | 1.008 | 5.375x | 7.647x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 3547.117 | 109.518 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 100.484 | 3587.590 | 108.254 | 35.703x | 1.077x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.618 | 1.064 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.382 | 0.618 | 1.068 | 1.619x | 2.799x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 3582.691 | 210.970 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 144.267 | 3646.829 | 210.378 | 25.278x | 1.458x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.719 | 0.979 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.132 | 0.711 | 0.974 | 5.380x | 7.365x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 3565.754 | 104.047 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 100.367 | 3489.040 | 106.086 | 34.763x | 1.057x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.604 | 1.084 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.378 | 0.605 | 1.079 | 1.600x | 2.854x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 3702.478 | 218.522 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 157.788 | 3633.353 | 216.293 | 23.027x | 1.371x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.700 | 0.984 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.132 | 0.697 | 0.984 | 5.298x | 7.475x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 3602.801 | 119.768 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 100.055 | 3625.939 | 119.995 | 36.240x | 1.199x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 4.052 | 4.067 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.365 | 4.043 | 4.087 | 11.091x | 11.210x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 748.277 | 733.087 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 135.228 | 762.542 | 719.286 | 5.639x | 5.319x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 4.128 | 3.459 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.125 | 4.166 | 3.476 | 33.382x | 27.853x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 701.611 | 695.138 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 99.452 | 732.864 | 711.647 | 7.369x | 7.156x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 4.068 | 4.087 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.354 | 4.043 | 4.041 | 11.423x | 11.417x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 819.703 | 801.778 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 135.248 | 756.099 | 712.210 | 5.590x | 5.266x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 4.160 | 3.478 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.127 | 4.136 | 3.492 | 32.672x | 27.585x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 685.772 | 679.708 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 101.167 | 697.904 | 696.404 | 6.899x | 6.884x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 4.057 | 4.025 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.369 | 4.125 | 4.074 | 11.186x | 11.048x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 773.861 | 758.092 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 157.629 | 779.846 | 765.921 | 4.947x | 4.859x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 4.144 | 3.478 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.129 | 4.229 | 3.528 | 32.730x | 27.307x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 727.401 | 723.780 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 102.274 | 725.617 | 730.693 | 7.095x | 7.144x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 4.610 | 4.036 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.376 | 4.655 | 4.044 | 12.389x | 10.762x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 2018.208 | 2111.880 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 486.366 | 1996.403 | 2137.734 | 4.105x | 4.395x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 4.729 | 3.483 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.130 | 4.723 | 3.481 | 36.216x | 26.694x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 1875.703 | 2065.299 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 418.503 | 1822.202 | 2025.254 | 4.354x | 4.839x |

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
