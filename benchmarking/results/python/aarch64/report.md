# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-07T16:23:23.656673+00:00
- Commit: `895b763`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.607 | 0.797 | 1.364 | 1.314x | 2.248x |
| FP32 | ADD | allocating | 7 | NEON | 0.584 | 0.795 | 1.381 | 1.361x | 2.365x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 476.612 | 546.364 | 327.099 | 1.146x | 0.686x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 389.635 | 534.952 | 297.446 | 1.373x | 0.763x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.227 | 0.945 | 1.228 | 4.169x | 5.416x |
| FP32 | ADD | inplace | 7 | NEON | 0.263 | 0.940 | 1.228 | 3.580x | 4.676x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 345.954 | 286.403 | 274.489 | 0.828x | 0.793x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 388.003 | 243.998 | 239.024 | 0.629x | 0.616x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.603 | 0.790 | 1.386 | 1.311x | 2.299x |
| FP32 | SUB | allocating | 7 | NEON | 0.600 | 0.790 | 1.393 | 1.317x | 2.323x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 479.418 | 584.635 | 325.337 | 1.219x | 0.679x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 440.923 | 534.270 | 296.622 | 1.212x | 0.673x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.231 | 0.932 | 1.303 | 4.039x | 5.646x |
| FP32 | SUB | inplace | 7 | NEON | 0.257 | 0.922 | 1.261 | 3.586x | 4.905x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 350.126 | 233.494 | 256.878 | 0.667x | 0.734x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 346.232 | 258.547 | 299.836 | 0.747x | 0.866x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.605 | 0.800 | 1.357 | 1.322x | 2.241x |
| FP32 | MUL | allocating | 7 | NEON | 0.589 | 0.801 | 1.361 | 1.360x | 2.312x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 473.752 | 578.129 | 297.870 | 1.220x | 0.629x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 412.001 | 576.907 | 291.000 | 1.400x | 0.706x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.238 | 0.934 | 1.204 | 3.933x | 5.068x |
| FP32 | MUL | inplace | 7 | NEON | 0.267 | 0.937 | 1.199 | 3.513x | 4.497x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 350.174 | 259.777 | 245.680 | 0.742x | 0.702x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 340.447 | 245.834 | 260.042 | 0.722x | 0.764x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.615 | 0.786 | 1.359 | 1.277x | 2.209x |
| FP32 | DIV | allocating | 7 | NEON | 0.595 | 0.787 | 1.394 | 1.322x | 2.342x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 682.379 | 715.959 | 486.668 | 1.049x | 0.713x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 481.354 | 734.642 | 496.096 | 1.526x | 1.031x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.234 | 0.925 | 1.616 | 3.950x | 6.904x |
| FP32 | DIV | inplace | 7 | NEON | 0.261 | 0.921 | 1.583 | 3.527x | 6.067x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 465.162 | 474.094 | 485.616 | 1.019x | 1.044x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 474.654 | 484.806 | 479.040 | 1.021x | 1.009x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.612 | 6.407 | 8.943 | 10.471x | 14.616x |
| INT8 | ADD | allocating | 7 | NEON | 0.661 | 6.400 | 8.937 | 9.675x | 13.511x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1017.243 | 554.152 | 599.539 | 0.545x | 0.589x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 115.377 | 557.191 | 618.492 | 4.829x | 5.361x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.236 | 6.349 | 7.436 | 26.884x | 31.486x |
| INT8 | ADD | inplace | 7 | NEON | 0.266 | 6.350 | 6.821 | 23.862x | 25.631x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 957.552 | 543.192 | 602.061 | 0.567x | 0.629x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 94.260 | 553.879 | 598.720 | 5.876x | 6.352x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.604 | 6.355 | 9.986 | 10.530x | 16.546x |
| INT8 | SUB | allocating | 7 | NEON | 0.686 | 6.344 | 10.312 | 9.245x | 15.026x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1016.440 | 555.148 | 599.773 | 0.546x | 0.590x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 116.532 | 564.875 | 616.041 | 4.847x | 5.286x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.237 | 6.365 | 7.322 | 26.826x | 30.856x |
| INT8 | SUB | inplace | 7 | NEON | 0.280 | 6.407 | 7.090 | 22.894x | 25.336x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 957.519 | 533.785 | 589.558 | 0.557x | 0.616x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 95.443 | 611.528 | 614.285 | 6.407x | 6.436x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.594 | 6.370 | 7.792 | 10.725x | 13.120x |
| INT8 | MUL | allocating | 7 | NEON | 0.661 | 6.395 | 8.762 | 9.671x | 13.250x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 970.886 | 555.703 | 601.015 | 0.572x | 0.619x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 129.751 | 564.893 | 595.759 | 4.354x | 4.592x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.238 | 6.364 | 6.411 | 26.747x | 26.946x |
| INT8 | MUL | inplace | 7 | NEON | 0.276 | 6.463 | 6.887 | 23.458x | 24.997x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 912.107 | 558.173 | 598.984 | 0.612x | 0.657x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 94.349 | 557.523 | 590.250 | 5.909x | 6.256x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.622 | 7.014 | 9.628 | 11.284x | 15.488x |
| INT8 | DIV | allocating | 7 | NEON | 0.671 | 7.039 | 9.188 | 10.493x | 13.695x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1677.802 | 2978.717 | 2080.533 | 1.775x | 1.240x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 573.499 | 2782.361 | 2071.630 | 4.852x | 3.612x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.236 | 7.062 | 7.848 | 29.882x | 33.205x |
| INT8 | DIV | inplace | 7 | NEON | 0.271 | 7.026 | 7.650 | 25.938x | 28.241x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1615.707 | 2848.076 | 2065.205 | 1.763x | 1.278x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 512.807 | 2799.128 | 2056.978 | 5.458x | 4.011x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.814 | 1.362 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.665 | 0.808 | 1.385 | 1.214x | 2.081x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1958.522 | 142.056 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1011.100 | 1968.945 | 141.308 | 1.947x | 0.140x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.949 | 1.210 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.269 | 0.943 | 1.227 | 3.505x | 4.561x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1892.674 | 127.254 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 897.358 | 1891.119 | 128.790 | 2.107x | 0.144x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.810 | 1.383 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.665 | 0.818 | 1.403 | 1.231x | 2.110x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1985.719 | 147.988 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1010.793 | 1986.992 | 194.174 | 1.966x | 0.192x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.946 | 1.314 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.267 | 0.944 | 1.320 | 3.532x | 4.939x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1902.340 | 146.426 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 898.326 | 1903.735 | 134.268 | 2.119x | 0.149x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.805 | 1.359 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.662 | 0.807 | 1.379 | 1.220x | 2.084x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1989.137 | 182.942 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1010.744 | 1985.157 | 160.420 | 1.964x | 0.159x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.946 | 1.199 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.269 | 0.941 | 1.194 | 3.497x | 4.437x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1899.762 | 142.265 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 899.576 | 1897.775 | 122.774 | 2.110x | 0.136x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.805 | 1.362 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.675 | 0.804 | 1.388 | 1.190x | 2.055x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2192.238 | 360.358 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1306.005 | 2180.031 | 358.444 | 1.669x | 0.274x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.939 | 1.531 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.264 | 0.939 | 1.582 | 3.556x | 5.993x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2102.491 | 353.086 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1192.386 | 2102.160 | 355.633 | 1.763x | 0.298x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.276 | 8.366 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.666 | 6.321 | 8.252 | 9.485x | 12.382x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1340.590 | 1414.867 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 242.286 | 1332.359 | 1457.958 | 5.499x | 6.018x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.275 | 6.469 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.264 | 6.398 | 6.296 | 24.231x | 23.845x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1249.604 | 1422.650 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 180.945 | 1268.467 | 1396.134 | 7.010x | 7.716x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.229 | 9.766 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.670 | 6.281 | 9.131 | 9.371x | 13.623x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1294.220 | 1410.742 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 236.916 | 1317.890 | 1400.393 | 5.563x | 5.911x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.272 | 7.613 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.264 | 6.281 | 7.479 | 23.818x | 28.364x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1244.146 | 1408.965 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 182.403 | 1282.923 | 1409.406 | 7.033x | 7.727x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.252 | 8.151 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.664 | 6.262 | 8.434 | 9.436x | 12.710x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1407.479 | 1365.656 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 250.802 | 1461.684 | 1378.008 | 5.828x | 5.494x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.286 | 6.469 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.261 | 6.277 | 6.648 | 24.022x | 25.442x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1356.308 | 1381.972 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 175.908 | 1349.732 | 1368.301 | 7.673x | 7.779x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.923 | 8.916 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.676 | 7.026 | 10.269 | 10.388x | 15.183x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3739.174 | 2693.919 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 611.645 | 3770.673 | 2689.412 | 6.165x | 4.397x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.942 | 9.274 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.269 | 7.022 | 8.471 | 26.107x | 31.495x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3848.836 | 2649.861 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 496.901 | 3663.412 | 2649.527 | 7.373x | 5.332x |

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
