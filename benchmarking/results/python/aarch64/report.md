# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-20T16:58:52.732277+00:00
- Commit: `12feb95`
- CPU: aarch64 (aarch64)
- OS: Linux-7.0.0-1017-azure-aarch64-with-glibc2.39
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.5.2
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.595 | 0.795 | 1.379 | 1.337x | 2.318x |
| FP32 | ADD | allocating | 7 | NEON | 0.610 | 0.789 | 1.393 | 1.293x | 2.284x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 466.950 | 539.427 | 298.549 | 1.155x | 0.639x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 386.660 | 524.869 | 363.106 | 1.357x | 0.939x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.226 | 0.938 | 1.260 | 4.150x | 5.573x |
| FP32 | ADD | inplace | 7 | NEON | 0.263 | 0.933 | 1.230 | 3.547x | 4.677x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 339.279 | 234.857 | 275.500 | 0.692x | 0.812x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 412.117 | 253.566 | 255.659 | 0.615x | 0.620x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.599 | 0.786 | 1.398 | 1.313x | 2.336x |
| FP32 | SUB | allocating | 7 | NEON | 0.598 | 0.783 | 1.411 | 1.310x | 2.359x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 458.336 | 521.637 | 287.634 | 1.138x | 0.628x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 394.144 | 521.471 | 287.432 | 1.323x | 0.729x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.229 | 0.923 | 1.343 | 4.039x | 5.876x |
| FP32 | SUB | inplace | 7 | NEON | 0.263 | 0.915 | 1.330 | 3.477x | 5.054x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 336.129 | 228.133 | 244.534 | 0.679x | 0.728x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 344.411 | 242.528 | 259.817 | 0.704x | 0.754x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.604 | 0.792 | 1.357 | 1.311x | 2.247x |
| FP32 | MUL | allocating | 7 | NEON | 0.599 | 0.801 | 1.365 | 1.339x | 2.281x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 456.498 | 537.122 | 295.835 | 1.177x | 0.648x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 377.029 | 514.681 | 297.255 | 1.365x | 0.788x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.227 | 0.928 | 1.205 | 4.085x | 5.303x |
| FP32 | MUL | inplace | 7 | NEON | 0.255 | 0.926 | 1.213 | 3.634x | 4.761x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 344.391 | 235.373 | 260.033 | 0.683x | 0.755x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 352.090 | 238.343 | 264.625 | 0.677x | 0.752x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.614 | 0.786 | 1.365 | 1.280x | 2.222x |
| FP32 | DIV | allocating | 7 | NEON | 0.592 | 0.777 | 1.373 | 1.314x | 2.319x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 672.664 | 714.460 | 492.045 | 1.062x | 0.731x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 474.696 | 713.736 | 497.884 | 1.504x | 1.049x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.231 | 0.913 | 1.650 | 3.957x | 7.152x |
| FP32 | DIV | inplace | 7 | NEON | 0.252 | 0.915 | 1.581 | 3.629x | 6.274x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 465.254 | 468.146 | 470.226 | 1.006x | 1.011x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 468.458 | 466.304 | 479.789 | 0.995x | 1.024x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.620 | 6.427 | 8.135 | 10.362x | 13.114x |
| INT8 | ADD | allocating | 7 | NEON | 0.666 | 6.336 | 8.620 | 9.520x | 12.951x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1015.090 | 547.397 | 617.542 | 0.539x | 0.608x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 111.674 | 545.607 | 605.887 | 4.886x | 5.426x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.234 | 6.408 | 6.871 | 27.398x | 29.374x |
| INT8 | ADD | inplace | 7 | NEON | 0.268 | 6.328 | 6.652 | 23.585x | 24.795x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 956.510 | 533.457 | 601.583 | 0.558x | 0.629x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 88.023 | 545.144 | 607.130 | 6.193x | 6.897x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.620 | 6.324 | 8.964 | 10.201x | 14.459x |
| INT8 | SUB | allocating | 7 | NEON | 0.678 | 6.321 | 10.317 | 9.328x | 15.226x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1014.848 | 548.720 | 629.640 | 0.541x | 0.620x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 112.085 | 553.379 | 611.668 | 4.937x | 5.457x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.233 | 6.351 | 8.427 | 27.284x | 36.201x |
| INT8 | SUB | inplace | 7 | NEON | 0.267 | 6.503 | 7.792 | 24.352x | 29.180x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 958.315 | 646.348 | 711.746 | 0.674x | 0.743x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 87.160 | 538.152 | 597.173 | 6.174x | 6.851x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.607 | 6.296 | 7.956 | 10.364x | 13.097x |
| INT8 | MUL | allocating | 7 | NEON | 0.686 | 6.318 | 8.653 | 9.212x | 12.616x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 967.675 | 573.331 | 593.352 | 0.592x | 0.613x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 142.255 | 590.921 | 594.574 | 4.154x | 4.180x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.236 | 6.392 | 6.758 | 27.030x | 28.576x |
| INT8 | MUL | inplace | 7 | NEON | 0.263 | 6.362 | 6.540 | 24.146x | 24.821x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 911.158 | 549.752 | 590.889 | 0.603x | 0.649x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 91.656 | 548.851 | 590.897 | 5.988x | 6.447x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.631 | 7.009 | 10.005 | 11.099x | 15.843x |
| INT8 | DIV | allocating | 7 | NEON | 0.687 | 6.999 | 10.376 | 10.187x | 15.104x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1680.785 | 2864.167 | 2063.313 | 1.704x | 1.228x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 570.397 | 2884.511 | 2073.940 | 5.057x | 3.636x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.241 | 7.095 | 8.612 | 29.448x | 35.743x |
| INT8 | DIV | inplace | 7 | NEON | 0.262 | 7.063 | 9.440 | 27.003x | 36.089x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1618.786 | 2856.591 | 2069.243 | 1.765x | 1.278x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 511.595 | 2905.237 | 2067.747 | 5.679x | 4.042x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.801 | 1.416 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.668 | 0.808 | 1.397 | 1.209x | 2.090x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1964.131 | 190.229 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1012.050 | 1976.235 | 182.284 | 1.953x | 0.180x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.934 | 1.222 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.279 | 0.947 | 1.229 | 3.398x | 4.406x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1893.657 | 163.315 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 901.026 | 1889.751 | 171.787 | 2.097x | 0.191x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.808 | 1.395 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.671 | 0.813 | 1.443 | 1.211x | 2.150x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1990.366 | 194.172 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1013.461 | 1987.750 | 202.407 | 1.961x | 0.200x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.943 | 1.320 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.261 | 0.934 | 1.306 | 3.585x | 5.012x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1896.686 | 160.056 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 901.859 | 1899.072 | 151.258 | 2.106x | 0.168x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.801 | 1.361 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.667 | 0.811 | 1.352 | 1.215x | 2.027x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1986.207 | 179.325 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1014.014 | 1982.466 | 207.492 | 1.955x | 0.205x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.944 | 1.204 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.266 | 0.949 | 1.211 | 3.570x | 4.559x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1912.241 | 164.286 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 901.907 | 1902.346 | 173.697 | 2.109x | 0.193x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.792 | 1.414 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.688 | 0.793 | 1.371 | 1.153x | 1.994x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2199.605 | 371.242 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1307.205 | 2214.723 | 366.940 | 1.694x | 0.281x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.929 | 1.527 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.269 | 0.934 | 1.556 | 3.473x | 5.785x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2149.019 | 356.529 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1328.069 | 2102.932 | 356.037 | 1.583x | 0.268x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.231 | 8.251 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.666 | 6.247 | 8.562 | 9.378x | 12.852x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1354.238 | 1499.996 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 251.947 | 1343.886 | 1504.967 | 5.334x | 5.973x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.240 | 6.730 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.267 | 6.216 | 6.910 | 23.308x | 25.908x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1363.302 | 1482.289 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 179.562 | 1315.362 | 1442.770 | 7.325x | 8.035x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.215 | 9.240 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.670 | 6.199 | 8.663 | 9.254x | 12.932x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1327.435 | 1518.261 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 230.258 | 1367.951 | 1519.123 | 5.941x | 6.597x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.232 | 7.208 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.262 | 6.256 | 8.539 | 23.873x | 32.584x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1310.828 | 1469.140 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 178.467 | 1301.100 | 1513.686 | 7.290x | 8.482x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.211 | 7.854 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.663 | 6.355 | 7.750 | 9.579x | 11.681x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1469.211 | 1537.452 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 276.408 | 1452.343 | 1432.997 | 5.254x | 5.184x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.230 | 6.236 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.262 | 6.268 | 6.811 | 23.888x | 25.958x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1393.248 | 1425.864 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 176.396 | 1400.134 | 1411.738 | 7.937x | 8.003x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.938 | 8.979 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.690 | 6.905 | 9.357 | 10.002x | 13.554x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3937.250 | 2698.150 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 611.313 | 3970.557 | 2704.892 | 6.495x | 4.425x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.923 | 7.595 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.265 | 6.949 | 8.940 | 26.229x | 33.745x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3931.762 | 2675.454 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 517.757 | 3759.166 | 2702.050 | 7.260x | 5.219x |

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
