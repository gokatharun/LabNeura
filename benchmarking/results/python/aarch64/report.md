# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-20T12:04:18.698725+00:00
- Commit: `2c5ef6c`
- CPU: aarch64 (aarch64)
- OS: Linux-7.0.0-1017-azure-aarch64-with-glibc2.39
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.5.1
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.594 | 0.789 | 1.411 | 1.327x | 2.374x |
| FP32 | ADD | allocating | 7 | NEON | 0.598 | 0.791 | 1.394 | 1.322x | 2.330x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 480.630 | 563.335 | 309.924 | 1.172x | 0.645x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 423.963 | 614.045 | 372.561 | 1.448x | 0.879x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.227 | 0.920 | 1.282 | 4.057x | 5.656x |
| FP32 | ADD | inplace | 7 | NEON | 0.255 | 0.921 | 1.248 | 3.612x | 4.897x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 351.012 | 362.487 | 341.254 | 1.033x | 0.972x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 420.762 | 311.457 | 336.988 | 0.740x | 0.801x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.595 | 0.782 | 1.395 | 1.314x | 2.343x |
| FP32 | SUB | allocating | 7 | NEON | 0.593 | 0.781 | 1.400 | 1.318x | 2.360x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 475.733 | 625.994 | 398.772 | 1.316x | 0.838x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 392.523 | 603.986 | 385.257 | 1.539x | 0.981x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.231 | 0.925 | 1.302 | 4.012x | 5.649x |
| FP32 | SUB | inplace | 7 | NEON | 0.254 | 0.925 | 1.313 | 3.638x | 5.162x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 360.239 | 309.963 | 335.923 | 0.860x | 0.933x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 366.676 | 327.561 | 352.442 | 0.893x | 0.961x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.606 | 0.788 | 1.366 | 1.301x | 2.256x |
| FP32 | MUL | allocating | 7 | NEON | 0.588 | 0.796 | 1.377 | 1.354x | 2.344x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 491.470 | 639.279 | 394.769 | 1.301x | 0.803x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 399.694 | 596.505 | 418.766 | 1.492x | 1.048x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.235 | 0.943 | 1.214 | 4.007x | 5.158x |
| FP32 | MUL | inplace | 7 | NEON | 0.260 | 0.936 | 1.231 | 3.597x | 4.730x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 363.774 | 316.531 | 345.512 | 0.870x | 0.950x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 351.328 | 325.413 | 283.660 | 0.926x | 0.807x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.603 | 0.776 | 1.375 | 1.288x | 2.282x |
| FP32 | DIV | allocating | 7 | NEON | 0.601 | 0.777 | 1.397 | 1.294x | 2.326x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 673.793 | 739.253 | 484.427 | 1.097x | 0.719x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 499.963 | 756.706 | 492.571 | 1.514x | 0.985x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.224 | 0.924 | 1.540 | 4.124x | 6.870x |
| FP32 | DIV | inplace | 7 | NEON | 0.261 | 0.922 | 1.505 | 3.527x | 5.759x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 464.266 | 468.174 | 483.151 | 1.008x | 1.041x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 491.222 | 476.464 | 499.868 | 0.970x | 1.018x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.636 | 6.379 | 8.927 | 10.026x | 14.032x |
| INT8 | ADD | allocating | 7 | NEON | 0.666 | 6.314 | 9.053 | 9.484x | 13.597x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1015.025 | 553.778 | 644.228 | 0.546x | 0.635x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 126.826 | 589.367 | 647.502 | 4.647x | 5.105x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.245 | 6.401 | 7.561 | 26.080x | 30.806x |
| INT8 | ADD | inplace | 7 | NEON | 0.274 | 6.369 | 6.545 | 23.267x | 23.906x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 958.112 | 580.603 | 645.420 | 0.606x | 0.674x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 88.007 | 553.123 | 600.222 | 6.285x | 6.820x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.610 | 6.354 | 10.182 | 10.414x | 16.690x |
| INT8 | SUB | allocating | 7 | NEON | 0.664 | 6.346 | 9.032 | 9.554x | 13.599x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1015.664 | 591.211 | 633.483 | 0.582x | 0.624x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 120.816 | 613.399 | 672.240 | 5.077x | 5.564x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.237 | 6.404 | 6.873 | 27.023x | 29.004x |
| INT8 | SUB | inplace | 7 | NEON | 0.267 | 6.345 | 7.838 | 23.748x | 29.333x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 957.365 | 562.644 | 637.510 | 0.588x | 0.666x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 98.355 | 609.967 | 650.553 | 6.202x | 6.614x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.614 | 6.353 | 8.186 | 10.351x | 13.337x |
| INT8 | MUL | allocating | 7 | NEON | 0.663 | 6.381 | 8.411 | 9.623x | 12.685x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 969.396 | 554.278 | 596.331 | 0.572x | 0.615x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 129.297 | 583.720 | 626.019 | 4.515x | 4.842x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.231 | 6.391 | 6.235 | 27.683x | 27.005x |
| INT8 | MUL | inplace | 7 | NEON | 0.273 | 6.401 | 6.782 | 23.466x | 24.860x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 911.014 | 564.750 | 615.160 | 0.620x | 0.675x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 95.328 | 544.262 | 592.947 | 5.709x | 6.220x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.619 | 7.190 | 9.286 | 11.623x | 15.011x |
| INT8 | DIV | allocating | 7 | NEON | 0.671 | 7.207 | 9.626 | 10.735x | 14.338x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1686.971 | 3142.726 | 2098.379 | 1.863x | 1.244x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 575.019 | 3106.397 | 2115.646 | 5.402x | 3.679x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.235 | 7.194 | 7.754 | 30.606x | 32.991x |
| INT8 | DIV | inplace | 7 | NEON | 0.272 | 7.175 | 9.550 | 26.410x | 35.153x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1619.664 | 2897.768 | 2100.163 | 1.789x | 1.297x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 517.842 | 3235.699 | 2112.780 | 6.248x | 4.080x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.807 | 1.416 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.670 | 0.803 | 1.410 | 1.199x | 2.103x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1969.747 | 177.503 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1022.822 | 2054.763 | 226.465 | 2.009x | 0.221x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.952 | 1.249 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.277 | 0.949 | 1.229 | 3.419x | 4.431x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1886.592 | 176.679 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 900.927 | 1882.184 | 176.849 | 2.089x | 0.196x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.801 | 1.398 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.654 | 0.803 | 1.403 | 1.228x | 2.143x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1986.077 | 226.481 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1016.681 | 2017.590 | 207.046 | 1.984x | 0.204x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.947 | 1.271 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.261 | 0.952 | 1.378 | 3.642x | 5.272x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1910.915 | 198.147 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 898.563 | 1899.656 | 162.246 | 2.114x | 0.181x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.812 | 1.376 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.674 | 0.807 | 1.363 | 1.197x | 2.022x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1983.456 | 175.785 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1011.051 | 1988.452 | 171.821 | 1.967x | 0.170x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.948 | 1.210 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.269 | 0.950 | 1.224 | 3.533x | 4.551x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1899.855 | 144.960 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 897.528 | 1900.250 | 143.915 | 2.117x | 0.160x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.795 | 1.399 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.660 | 0.792 | 1.383 | 1.201x | 2.096x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2194.023 | 362.347 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1314.000 | 2183.074 | 363.700 | 1.661x | 0.277x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.935 | 1.529 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.269 | 0.941 | 1.471 | 3.494x | 5.464x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2105.503 | 353.470 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1190.860 | 2105.515 | 355.239 | 1.768x | 0.298x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.300 | 8.629 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.670 | 6.349 | 8.198 | 9.482x | 12.243x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1551.447 | 1571.788 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 242.189 | 1336.158 | 1450.428 | 5.517x | 5.989x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.311 | 7.410 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.273 | 6.348 | 7.314 | 23.281x | 26.826x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1537.417 | 1554.474 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 177.374 | 1332.845 | 1419.552 | 7.514x | 8.003x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.257 | 8.866 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.660 | 6.341 | 9.133 | 9.605x | 13.835x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1291.500 | 1468.789 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 233.409 | 1358.806 | 1498.903 | 5.822x | 6.422x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.258 | 7.115 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.269 | 6.331 | 7.901 | 23.568x | 29.410x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1302.013 | 1517.637 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 180.316 | 1571.559 | 1478.646 | 8.716x | 8.200x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.284 | 8.652 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.669 | 6.273 | 8.098 | 9.377x | 12.106x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1468.875 | 1436.421 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 257.933 | 1510.781 | 1398.670 | 5.857x | 5.423x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.336 | 6.523 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.267 | 6.353 | 6.800 | 23.758x | 25.433x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1463.157 | 1464.144 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 194.894 | 1409.421 | 1413.584 | 7.232x | 7.253x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 7.084 | 10.724 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.672 | 6.958 | 8.993 | 10.351x | 13.379x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3888.206 | 2701.172 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 612.894 | 3860.720 | 2707.801 | 6.299x | 4.418x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.947 | 9.560 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.266 | 6.978 | 9.446 | 26.194x | 35.456x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3837.814 | 2714.956 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 503.964 | 4012.782 | 2693.777 | 7.962x | 5.345x |

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
