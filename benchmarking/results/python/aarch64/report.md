# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-07T07:16:44.183325+00:00
- Commit: `f94f6a8`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.682 | 0.787 | 1.360 | 1.155x | 1.995x |
| FP32 | ADD | allocating | 7 | NEON | 0.579 | 0.790 | 1.365 | 1.364x | 2.357x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 496.731 | 603.855 | 300.020 | 1.216x | 0.604x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 418.207 | 569.070 | 308.909 | 1.361x | 0.739x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.226 | 0.931 | 1.273 | 4.129x | 5.643x |
| FP32 | ADD | inplace | 7 | NEON | 0.258 | 0.929 | 1.295 | 3.600x | 5.016x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 355.356 | 279.838 | 266.199 | 0.787x | 0.749x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 366.606 | 256.172 | 259.927 | 0.699x | 0.709x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.586 | 0.786 | 1.412 | 1.342x | 2.411x |
| FP32 | SUB | allocating | 7 | NEON | 0.577 | 0.775 | 1.473 | 1.342x | 2.551x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 496.506 | 612.570 | 336.206 | 1.234x | 0.677x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 446.289 | 572.946 | 304.495 | 1.284x | 0.682x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.223 | 0.925 | 1.275 | 4.149x | 5.722x |
| FP32 | SUB | inplace | 7 | NEON | 0.257 | 0.920 | 1.298 | 3.584x | 5.054x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 360.276 | 467.292 | 502.898 | 1.297x | 1.396x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 350.392 | 231.135 | 257.418 | 0.660x | 0.735x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.592 | 0.790 | 1.386 | 1.334x | 2.341x |
| FP32 | MUL | allocating | 7 | NEON | 0.588 | 0.789 | 1.384 | 1.341x | 2.354x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 486.601 | 642.702 | 296.220 | 1.321x | 0.609x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 410.271 | 562.969 | 319.608 | 1.372x | 0.779x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.220 | 0.926 | 1.199 | 4.211x | 5.454x |
| FP32 | MUL | inplace | 7 | NEON | 0.255 | 0.914 | 1.205 | 3.589x | 4.735x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 361.704 | 269.929 | 297.225 | 0.746x | 0.822x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 357.404 | 259.890 | 272.889 | 0.727x | 0.764x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.612 | 0.768 | 1.372 | 1.255x | 2.241x |
| FP32 | DIV | allocating | 7 | NEON | 0.593 | 0.769 | 1.357 | 1.296x | 2.288x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 696.900 | 708.999 | 487.099 | 1.017x | 0.699x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 476.508 | 754.514 | 496.729 | 1.583x | 1.042x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.222 | 0.908 | 1.513 | 4.093x | 6.821x |
| FP32 | DIV | inplace | 7 | NEON | 0.256 | 0.909 | 1.549 | 3.554x | 6.060x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 469.462 | 474.068 | 478.146 | 1.010x | 1.018x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 472.462 | 466.746 | 481.917 | 0.988x | 1.020x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.609 | 6.360 | 8.610 | 10.439x | 14.132x |
| INT8 | ADD | allocating | 7 | NEON | 0.645 | 6.297 | 8.313 | 9.769x | 12.896x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1018.488 | 556.018 | 603.468 | 0.546x | 0.593x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 116.993 | 556.537 | 623.288 | 4.757x | 5.328x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.238 | 6.394 | 6.716 | 26.897x | 28.250x |
| INT8 | ADD | inplace | 7 | NEON | 0.264 | 6.354 | 6.654 | 24.042x | 25.174x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 956.007 | 539.208 | 581.319 | 0.564x | 0.608x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 91.666 | 543.949 | 589.147 | 5.934x | 6.427x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.600 | 6.278 | 9.188 | 10.464x | 15.314x |
| INT8 | SUB | allocating | 7 | NEON | 0.662 | 6.297 | 9.707 | 9.512x | 14.662x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1017.390 | 556.455 | 608.729 | 0.547x | 0.598x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 123.855 | 557.183 | 610.860 | 4.499x | 4.932x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.240 | 6.353 | 7.430 | 26.421x | 30.898x |
| INT8 | SUB | inplace | 7 | NEON | 0.262 | 6.466 | 7.186 | 24.657x | 27.401x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 958.381 | 549.529 | 599.178 | 0.573x | 0.625x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 96.843 | 584.221 | 621.646 | 6.033x | 6.419x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.607 | 6.346 | 8.002 | 10.462x | 13.193x |
| INT8 | MUL | allocating | 7 | NEON | 0.669 | 6.337 | 8.048 | 9.471x | 12.028x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 973.180 | 571.772 | 614.260 | 0.588x | 0.631x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 124.850 | 556.812 | 611.734 | 4.460x | 4.900x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.230 | 6.327 | 6.462 | 27.521x | 28.109x |
| INT8 | MUL | inplace | 7 | NEON | 0.264 | 6.366 | 6.767 | 24.131x | 25.648x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 912.464 | 544.960 | 605.032 | 0.597x | 0.663x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 93.728 | 566.279 | 596.196 | 6.042x | 6.361x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.611 | 7.005 | 10.939 | 11.467x | 17.907x |
| INT8 | DIV | allocating | 7 | NEON | 0.655 | 6.975 | 8.995 | 10.654x | 13.740x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1680.276 | 2795.938 | 2079.551 | 1.664x | 1.238x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 574.530 | 2889.422 | 2077.555 | 5.029x | 3.616x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.231 | 7.051 | 8.254 | 30.502x | 35.704x |
| INT8 | DIV | inplace | 7 | NEON | 0.265 | 7.048 | 8.824 | 26.603x | 33.306x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1615.563 | 2927.550 | 2067.322 | 1.812x | 1.280x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 515.452 | 2859.643 | 2065.657 | 5.548x | 4.007x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.804 | 1.384 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.670 | 0.792 | 1.386 | 1.182x | 2.069x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1963.365 | 166.771 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1016.391 | 1967.502 | 195.262 | 1.936x | 0.192x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.928 | 1.275 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.267 | 0.928 | 1.258 | 3.468x | 4.705x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1888.435 | 134.795 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 898.534 | 1889.584 | 140.425 | 2.103x | 0.156x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.806 | 1.403 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.650 | 0.806 | 1.434 | 1.240x | 2.206x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1987.298 | 172.082 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1016.113 | 1982.754 | 167.285 | 1.951x | 0.165x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.935 | 1.297 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.261 | 0.927 | 1.278 | 3.554x | 4.897x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1906.088 | 134.199 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 898.435 | 1911.212 | 150.650 | 2.127x | 0.168x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.803 | 1.366 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.656 | 0.802 | 1.385 | 1.221x | 2.109x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1980.916 | 212.595 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1015.213 | 2001.591 | 152.076 | 1.972x | 0.150x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.935 | 1.213 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.259 | 0.932 | 1.208 | 3.599x | 4.667x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1931.537 | 243.397 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 899.791 | 1900.999 | 146.992 | 2.113x | 0.163x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.795 | 1.374 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.670 | 0.791 | 1.388 | 1.181x | 2.073x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2186.154 | 362.566 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1312.668 | 2183.757 | 362.915 | 1.664x | 0.276x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.919 | 1.583 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.256 | 0.925 | 1.543 | 3.617x | 6.031x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2106.081 | 356.510 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1194.542 | 2107.049 | 353.923 | 1.764x | 0.296x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.259 | 8.427 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.667 | 6.281 | 8.519 | 9.421x | 12.779x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1353.260 | 1443.372 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 248.955 | 1331.702 | 1438.874 | 5.349x | 5.780x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.290 | 6.735 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.259 | 6.309 | 6.260 | 24.401x | 24.212x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1251.130 | 1449.605 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 224.170 | 1259.302 | 1407.230 | 5.618x | 6.278x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.266 | 10.945 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.659 | 6.393 | 9.704 | 9.701x | 14.725x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1341.549 | 1454.467 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 245.223 | 1304.259 | 1395.074 | 5.319x | 5.689x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.299 | 7.384 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.260 | 6.296 | 7.473 | 24.171x | 28.691x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1280.571 | 1516.142 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 184.784 | 1253.671 | 1377.519 | 6.785x | 7.455x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.265 | 8.014 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.654 | 6.239 | 7.924 | 9.543x | 12.122x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1396.154 | 1384.087 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 269.085 | 1396.752 | 1407.479 | 5.191x | 5.231x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.253 | 6.600 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.258 | 6.260 | 6.266 | 24.288x | 24.312x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1331.214 | 1414.504 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 182.337 | 1340.486 | 1352.891 | 7.352x | 7.420x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 7.026 | 8.893 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.656 | 6.915 | 9.122 | 10.544x | 13.910x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3679.450 | 2701.547 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 621.928 | 3834.402 | 2664.287 | 6.165x | 4.284x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.945 | 7.215 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.258 | 6.984 | 8.262 | 27.097x | 32.055x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3719.053 | 2647.600 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 498.971 | 3882.758 | 2667.335 | 7.782x | 5.346x |

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
