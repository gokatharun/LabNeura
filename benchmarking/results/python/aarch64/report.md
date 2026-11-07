# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-07T05:43:22.520228+00:00
- Commit: `3d8d879`
- CPU: aarch64 (aarch64)
- OS: Linux-7.0.0-1017-azure-aarch64-with-glibc2.39
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.4.1
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.580 | 0.831 | 1.405 | 1.434x | 2.422x |
| FP32 | ADD | allocating | 7 | NEON | 0.592 | 0.835 | 1.413 | 1.409x | 2.387x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 482.855 | 503.887 | 278.224 | 1.044x | 0.576x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 410.576 | 490.686 | 293.309 | 1.195x | 0.714x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.220 | 0.982 | 1.301 | 4.471x | 5.922x |
| FP32 | ADD | inplace | 7 | NEON | 0.261 | 0.980 | 1.267 | 3.759x | 4.857x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 359.845 | 251.160 | 240.742 | 0.698x | 0.669x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 364.557 | 280.082 | 267.149 | 0.768x | 0.733x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.575 | 0.825 | 1.454 | 1.436x | 2.529x |
| FP32 | SUB | allocating | 7 | NEON | 0.592 | 0.823 | 1.465 | 1.389x | 2.473x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 478.620 | 477.566 | 260.877 | 0.998x | 0.545x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 409.548 | 530.207 | 311.062 | 1.295x | 0.760x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.225 | 0.970 | 1.373 | 4.315x | 6.106x |
| FP32 | SUB | inplace | 7 | NEON | 0.253 | 0.967 | 1.333 | 3.823x | 5.267x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 359.111 | 240.602 | 237.538 | 0.670x | 0.661x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 355.727 | 233.227 | 227.127 | 0.656x | 0.638x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.586 | 0.856 | 1.417 | 1.460x | 2.417x |
| FP32 | MUL | allocating | 7 | NEON | 0.606 | 0.863 | 1.401 | 1.424x | 2.313x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 469.335 | 502.587 | 273.548 | 1.071x | 0.583x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 439.895 | 498.447 | 342.878 | 1.133x | 0.779x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.223 | 0.994 | 1.243 | 4.451x | 5.563x |
| FP32 | MUL | inplace | 7 | NEON | 0.255 | 0.998 | 1.263 | 3.907x | 4.946x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 353.521 | 215.181 | 215.725 | 0.609x | 0.610x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 361.397 | 224.075 | 266.899 | 0.620x | 0.739x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.579 | 0.819 | 1.414 | 1.416x | 2.443x |
| FP32 | DIV | allocating | 7 | NEON | 0.591 | 0.819 | 1.418 | 1.385x | 2.398x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 681.221 | 702.193 | 483.736 | 1.031x | 0.710x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 481.240 | 748.437 | 488.986 | 1.555x | 1.016x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.223 | 0.960 | 1.577 | 4.301x | 7.064x |
| FP32 | DIV | inplace | 7 | NEON | 0.255 | 0.957 | 1.573 | 3.759x | 6.177x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 462.959 | 466.217 | 473.769 | 1.007x | 1.023x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 471.459 | 460.779 | 467.221 | 0.977x | 0.991x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.594 | 6.507 | 8.544 | 10.956x | 14.386x |
| INT8 | ADD | allocating | 7 | NEON | 0.675 | 6.433 | 8.183 | 9.537x | 12.132x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1017.660 | 584.256 | 617.881 | 0.574x | 0.607x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 120.047 | 558.904 | 613.309 | 4.656x | 5.109x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.227 | 6.442 | 7.142 | 28.334x | 31.415x |
| INT8 | ADD | inplace | 7 | NEON | 0.264 | 6.472 | 6.677 | 24.474x | 25.248x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 957.400 | 547.035 | 602.255 | 0.571x | 0.629x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 96.676 | 546.168 | 596.205 | 5.649x | 6.167x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.602 | 6.521 | 9.108 | 10.836x | 15.137x |
| INT8 | SUB | allocating | 7 | NEON | 0.664 | 6.473 | 9.539 | 9.754x | 14.374x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1019.610 | 580.674 | 629.006 | 0.570x | 0.617x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 119.171 | 574.145 | 634.074 | 4.818x | 5.321x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.241 | 6.477 | 7.392 | 26.844x | 30.636x |
| INT8 | SUB | inplace | 7 | NEON | 0.266 | 6.588 | 7.745 | 24.805x | 29.159x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 956.495 | 558.294 | 615.272 | 0.584x | 0.643x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 93.654 | 564.372 | 611.129 | 6.026x | 6.525x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.594 | 6.480 | 7.716 | 10.905x | 12.985x |
| INT8 | MUL | allocating | 7 | NEON | 0.667 | 6.446 | 8.233 | 9.657x | 12.334x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 972.734 | 571.526 | 604.607 | 0.588x | 0.622x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 123.790 | 584.050 | 620.753 | 4.718x | 5.015x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.231 | 6.499 | 6.648 | 28.110x | 28.756x |
| INT8 | MUL | inplace | 7 | NEON | 0.262 | 6.471 | 6.861 | 24.706x | 26.195x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 911.070 | 542.200 | 582.599 | 0.595x | 0.639x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 94.724 | 560.425 | 601.434 | 5.916x | 6.349x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.592 | 7.090 | 10.156 | 11.984x | 17.165x |
| INT8 | DIV | allocating | 7 | NEON | 0.661 | 7.076 | 10.013 | 10.704x | 15.146x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1682.031 | 2858.188 | 2064.084 | 1.699x | 1.227x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 562.088 | 2827.897 | 2077.906 | 5.031x | 3.697x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.233 | 7.031 | 8.400 | 30.226x | 36.108x |
| INT8 | DIV | inplace | 7 | NEON | 0.268 | 7.141 | 8.780 | 26.616x | 32.723x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1618.763 | 2853.191 | 2059.227 | 1.763x | 1.272x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 516.265 | 2855.606 | 2071.804 | 5.531x | 4.013x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.848 | 1.393 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.655 | 0.847 | 1.426 | 1.291x | 2.175x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1948.510 | 176.578 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1015.983 | 1958.661 | 164.399 | 1.928x | 0.162x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.989 | 1.267 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.256 | 0.991 | 1.261 | 3.865x | 4.917x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1888.814 | 144.578 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 898.116 | 1887.502 | 143.863 | 2.102x | 0.160x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.846 | 1.401 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.655 | 0.847 | 1.477 | 1.294x | 2.256x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1975.305 | 169.180 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1019.811 | 1984.170 | 179.382 | 1.946x | 0.176x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.984 | 1.325 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.255 | 1.002 | 1.323 | 3.933x | 5.192x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1900.021 | 128.833 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 898.816 | 1907.852 | 156.428 | 2.123x | 0.174x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.843 | 1.378 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.676 | 0.845 | 1.408 | 1.251x | 2.084x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1987.144 | 171.057 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1017.502 | 1987.564 | 174.563 | 1.953x | 0.172x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 1.002 | 1.234 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.257 | 0.986 | 1.229 | 3.832x | 4.773x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1897.393 | 144.129 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 898.716 | 1902.382 | 143.544 | 2.117x | 0.160x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.835 | 1.398 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.663 | 0.828 | 1.400 | 1.249x | 2.111x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2190.743 | 367.836 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1313.563 | 2182.137 | 363.470 | 1.661x | 0.277x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.983 | 1.567 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.261 | 0.983 | 1.532 | 3.759x | 5.860x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2107.977 | 355.599 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1194.904 | 2104.871 | 354.914 | 1.762x | 0.297x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.334 | 8.539 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.671 | 6.402 | 8.672 | 9.541x | 12.923x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1338.693 | 1469.713 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 247.578 | 1333.114 | 1504.630 | 5.385x | 6.077x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.385 | 6.899 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.259 | 6.357 | 6.479 | 24.571x | 25.046x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1290.565 | 1464.943 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 186.988 | 1294.731 | 1451.067 | 6.924x | 7.760x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.341 | 8.901 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.672 | 6.336 | 8.836 | 9.430x | 13.150x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1345.988 | 1495.171 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 345.177 | 1365.187 | 1424.015 | 3.955x | 4.125x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.360 | 7.333 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.259 | 6.350 | 7.252 | 24.482x | 27.961x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1324.505 | 1476.114 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 188.980 | 1285.966 | 1430.895 | 6.805x | 7.572x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.286 | 7.929 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.671 | 6.346 | 8.244 | 9.460x | 12.289x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1432.724 | 1384.109 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 273.091 | 1448.290 | 1419.769 | 5.303x | 5.199x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.310 | 5.992 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.261 | 6.367 | 6.613 | 24.370x | 25.310x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1377.564 | 1384.083 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 188.673 | 1407.287 | 1433.179 | 7.459x | 7.596x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.967 | 9.383 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.667 | 6.948 | 8.769 | 10.412x | 13.141x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3886.158 | 2696.681 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 618.366 | 3791.949 | 2705.093 | 6.132x | 4.375x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.950 | 7.747 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.259 | 6.981 | 7.962 | 26.986x | 30.777x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3907.674 | 2675.854 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 496.022 | 3784.944 | 2674.486 | 7.631x | 5.392x |

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
