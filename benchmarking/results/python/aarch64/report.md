# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-07T03:11:30.384703+00:00
- Commit: `88dce7a`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.585 | 0.784 | 1.372 | 1.340x | 2.346x |
| FP32 | ADD | allocating | 7 | NEON | 0.583 | 0.796 | 1.447 | 1.364x | 2.481x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 519.183 | 650.830 | 421.487 | 1.254x | 0.812x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 427.427 | 697.625 | 484.949 | 1.632x | 1.135x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.217 | 0.919 | 1.212 | 4.235x | 5.587x |
| FP32 | ADD | inplace | 7 | NEON | 0.215 | 0.916 | 1.206 | 4.251x | 5.598x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 383.536 | 354.951 | 371.911 | 0.925x | 0.970x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 379.706 | 384.528 | 348.973 | 1.013x | 0.919x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.591 | 0.777 | 1.418 | 1.315x | 2.402x |
| FP32 | SUB | allocating | 7 | NEON | 0.587 | 0.782 | 1.355 | 1.331x | 2.308x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 565.449 | 572.119 | 355.798 | 1.012x | 0.629x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 421.473 | 678.355 | 441.526 | 1.609x | 1.048x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.220 | 0.926 | 1.541 | 4.202x | 6.987x |
| FP32 | SUB | inplace | 7 | NEON | 0.211 | 0.925 | 1.527 | 4.384x | 7.239x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 377.894 | 404.315 | 439.935 | 1.070x | 1.164x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 365.839 | 358.891 | 367.961 | 0.981x | 1.006x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.669 | 0.815 | 1.378 | 1.218x | 2.060x |
| FP32 | MUL | allocating | 7 | NEON | 0.587 | 0.816 | 1.367 | 1.391x | 2.330x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 498.331 | 701.549 | 401.364 | 1.408x | 0.805x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 414.444 | 728.888 | 424.712 | 1.759x | 1.025x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.222 | 0.959 | 1.177 | 4.329x | 5.313x |
| FP32 | MUL | inplace | 7 | NEON | 0.213 | 0.964 | 1.186 | 4.525x | 5.570x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 364.260 | 331.045 | 355.028 | 0.909x | 0.975x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 380.367 | 316.167 | 321.697 | 0.831x | 0.846x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.586 | 0.775 | 1.344 | 1.321x | 2.293x |
| FP32 | DIV | allocating | 7 | NEON | 0.594 | 0.769 | 1.339 | 1.294x | 2.253x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 697.724 | 735.861 | 495.550 | 1.055x | 0.710x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 474.158 | 771.637 | 506.030 | 1.627x | 1.067x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.217 | 0.921 | 1.521 | 4.249x | 7.020x |
| FP32 | DIV | inplace | 7 | NEON | 0.210 | 0.917 | 1.493 | 4.372x | 7.120x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 470.459 | 513.506 | 508.150 | 1.092x | 1.080x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 497.209 | 491.745 | 515.462 | 0.989x | 1.037x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.589 | 6.480 | 8.984 | 11.002x | 15.252x |
| INT8 | ADD | allocating | 7 | NEON | 0.613 | 6.368 | 9.471 | 10.394x | 15.459x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1022.912 | 577.574 | 618.622 | 0.565x | 0.605x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 116.843 | 593.822 | 647.358 | 5.082x | 5.540x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.229 | 6.332 | 6.951 | 27.654x | 30.357x |
| INT8 | ADD | inplace | 7 | NEON | 0.223 | 6.335 | 6.947 | 28.419x | 31.168x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 957.142 | 587.964 | 627.194 | 0.614x | 0.655x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 101.951 | 575.875 | 632.949 | 5.649x | 6.208x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.585 | 6.320 | 8.750 | 10.808x | 14.962x |
| INT8 | SUB | allocating | 7 | NEON | 0.613 | 6.364 | 9.372 | 10.375x | 15.277x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1023.250 | 564.627 | 617.926 | 0.552x | 0.604x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 120.228 | 608.831 | 656.598 | 5.064x | 5.461x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.224 | 6.408 | 6.947 | 28.630x | 31.037x |
| INT8 | SUB | inplace | 7 | NEON | 0.218 | 6.318 | 7.989 | 28.991x | 36.655x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 956.803 | 602.866 | 662.268 | 0.630x | 0.692x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 92.501 | 576.658 | 650.287 | 6.234x | 7.030x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.578 | 6.372 | 8.348 | 11.020x | 14.436x |
| INT8 | MUL | allocating | 7 | NEON | 0.586 | 6.320 | 7.713 | 10.779x | 13.156x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 971.963 | 614.674 | 631.386 | 0.632x | 0.650x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 132.688 | 727.748 | 636.498 | 5.485x | 4.797x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.218 | 6.305 | 6.096 | 28.869x | 27.914x |
| INT8 | MUL | inplace | 7 | NEON | 0.222 | 6.355 | 6.149 | 28.638x | 27.710x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 911.807 | 575.821 | 593.208 | 0.632x | 0.651x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 101.194 | 582.733 | 617.071 | 5.759x | 6.098x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.576 | 7.040 | 9.004 | 12.231x | 15.644x |
| INT8 | DIV | allocating | 7 | NEON | 0.595 | 7.124 | 11.123 | 11.968x | 18.687x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1684.575 | 3218.140 | 2110.387 | 1.910x | 1.253x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 568.993 | 3075.599 | 2094.360 | 5.405x | 3.681x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.230 | 6.971 | 9.150 | 30.317x | 39.794x |
| INT8 | DIV | inplace | 7 | NEON | 0.219 | 7.000 | 9.171 | 31.906x | 41.804x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1620.327 | 3169.183 | 2130.685 | 1.956x | 1.315x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 512.869 | 3115.114 | 2091.289 | 6.074x | 4.078x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.806 | 1.343 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.599 | 0.801 | 1.352 | 1.338x | 2.259x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1972.707 | 181.494 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1019.507 | 1965.321 | 188.470 | 1.928x | 0.185x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.945 | 1.229 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.222 | 0.953 | 1.203 | 4.295x | 5.427x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1892.953 | 164.556 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 902.309 | 1899.229 | 189.502 | 2.105x | 0.210x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.801 | 1.350 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.606 | 0.804 | 1.359 | 1.327x | 2.243x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1991.643 | 187.006 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1015.601 | 1997.981 | 206.619 | 1.967x | 0.203x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.946 | 1.535 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.215 | 0.951 | 1.491 | 4.418x | 6.929x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1909.135 | 160.582 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 901.916 | 1904.370 | 157.217 | 2.111x | 0.174x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.798 | 1.346 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.603 | 0.797 | 1.365 | 1.323x | 2.265x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1995.270 | 193.563 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1020.323 | 1983.196 | 172.018 | 1.944x | 0.169x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.948 | 1.198 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.219 | 0.943 | 1.204 | 4.307x | 5.502x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1910.977 | 174.272 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 900.367 | 1907.397 | 174.033 | 2.118x | 0.193x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.789 | 1.339 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.600 | 0.791 | 1.371 | 1.317x | 2.283x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2195.286 | 362.828 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1310.966 | 2196.940 | 362.226 | 1.676x | 0.276x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.942 | 1.513 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.220 | 0.937 | 1.481 | 4.263x | 6.743x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2115.970 | 353.828 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1192.960 | 2109.162 | 352.003 | 1.768x | 0.295x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.259 | 9.338 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.597 | 6.213 | 8.274 | 10.408x | 13.859x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1623.695 | 1643.857 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 255.788 | 1602.266 | 1703.825 | 6.264x | 6.661x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.226 | 7.250 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.219 | 6.231 | 6.796 | 28.489x | 31.073x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1432.425 | 1511.880 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 196.109 | 1379.199 | 1518.891 | 7.033x | 7.745x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.161 | 9.898 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.584 | 6.236 | 9.167 | 10.686x | 15.710x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1460.614 | 1502.748 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 241.831 | 1671.136 | 1531.046 | 6.910x | 6.331x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.185 | 7.808 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.218 | 6.211 | 7.497 | 28.461x | 34.352x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1438.294 | 1569.954 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 189.344 | 1461.569 | 1596.215 | 7.719x | 8.430x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.191 | 7.740 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.612 | 6.213 | 7.667 | 10.159x | 12.538x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1561.396 | 1614.236 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 260.668 | 1521.454 | 1541.777 | 5.837x | 5.915x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.229 | 6.854 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.215 | 6.225 | 6.195 | 28.948x | 28.808x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1492.123 | 1506.602 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 207.637 | 1509.206 | 1513.239 | 7.268x | 7.288x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.839 | 8.586 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.599 | 6.903 | 8.535 | 11.533x | 14.259x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 4215.455 | 2767.993 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 614.062 | 3890.343 | 2720.022 | 6.335x | 4.430x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.842 | 7.309 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.221 | 6.898 | 7.965 | 31.239x | 36.074x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 4203.107 | 2839.127 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 511.218 | 4191.658 | 2744.542 | 8.199x | 5.369x |

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
