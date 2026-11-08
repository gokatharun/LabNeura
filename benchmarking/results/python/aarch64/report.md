# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-08T05:13:39.009318+00:00
- Commit: `665991b`
- CPU: aarch64 (aarch64)
- OS: Linux-7.0.0-1017-azure-aarch64-with-glibc2.39
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.4.2
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.670 | 0.788 | 1.363 | 1.177x | 2.036x |
| FP32 | ADD | allocating | 7 | NEON | 0.572 | 0.786 | 1.379 | 1.373x | 2.408x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 456.925 | 496.260 | 255.060 | 1.086x | 0.558x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 384.260 | 476.736 | 243.754 | 1.241x | 0.634x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.224 | 0.918 | 1.275 | 4.107x | 5.705x |
| FP32 | ADD | inplace | 7 | NEON | 0.255 | 0.916 | 1.259 | 3.597x | 4.942x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 338.783 | 217.483 | 220.719 | 0.642x | 0.652x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 332.488 | 208.786 | 217.995 | 0.628x | 0.656x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.583 | 0.787 | 1.407 | 1.350x | 2.414x |
| FP32 | SUB | allocating | 7 | NEON | 0.582 | 0.786 | 1.409 | 1.351x | 2.423x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 463.306 | 496.045 | 245.719 | 1.071x | 0.530x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 377.017 | 532.544 | 267.118 | 1.413x | 0.709x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.224 | 0.917 | 1.283 | 4.085x | 5.715x |
| FP32 | SUB | inplace | 7 | NEON | 0.249 | 0.919 | 1.264 | 3.692x | 5.077x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 336.580 | 215.582 | 224.223 | 0.641x | 0.666x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 342.132 | 250.343 | 253.958 | 0.732x | 0.742x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.593 | 0.793 | 1.385 | 1.337x | 2.336x |
| FP32 | MUL | allocating | 7 | NEON | 0.569 | 0.797 | 1.381 | 1.401x | 2.428x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 461.788 | 506.865 | 254.625 | 1.098x | 0.551x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 383.569 | 532.060 | 255.424 | 1.387x | 0.666x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.227 | 0.930 | 1.203 | 4.101x | 5.303x |
| FP32 | MUL | inplace | 7 | NEON | 0.256 | 0.932 | 1.227 | 3.633x | 4.782x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 328.295 | 217.562 | 216.356 | 0.663x | 0.659x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 324.670 | 218.830 | 233.303 | 0.674x | 0.719x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.587 | 0.778 | 1.425 | 1.324x | 2.426x |
| FP32 | DIV | allocating | 7 | NEON | 0.575 | 0.775 | 1.429 | 1.348x | 2.485x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 675.904 | 701.597 | 474.326 | 1.038x | 0.702x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 472.824 | 714.775 | 479.378 | 1.512x | 1.014x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.227 | 0.911 | 1.599 | 4.006x | 7.033x |
| FP32 | DIV | inplace | 7 | NEON | 0.260 | 0.911 | 1.545 | 3.508x | 5.955x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 457.373 | 461.401 | 473.214 | 1.009x | 1.035x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 467.006 | 462.759 | 471.194 | 0.991x | 1.009x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.588 | 6.346 | 8.934 | 10.796x | 15.198x |
| INT8 | ADD | allocating | 7 | NEON | 0.636 | 6.353 | 8.488 | 9.991x | 13.349x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1014.777 | 540.423 | 585.708 | 0.533x | 0.577x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 112.333 | 541.633 | 589.439 | 4.822x | 5.247x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.225 | 6.378 | 6.503 | 28.331x | 28.886x |
| INT8 | ADD | inplace | 7 | NEON | 0.258 | 6.358 | 6.783 | 24.637x | 26.283x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 956.592 | 532.466 | 576.672 | 0.557x | 0.603x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 87.909 | 524.356 | 576.248 | 5.965x | 6.555x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.585 | 6.349 | 10.299 | 10.850x | 17.601x |
| INT8 | SUB | allocating | 7 | NEON | 0.632 | 6.360 | 9.544 | 10.056x | 15.090x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1016.217 | 538.383 | 584.855 | 0.530x | 0.576x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 113.479 | 541.962 | 590.955 | 4.776x | 5.208x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.228 | 6.336 | 8.245 | 27.831x | 36.211x |
| INT8 | SUB | inplace | 7 | NEON | 0.256 | 6.446 | 7.547 | 25.179x | 29.479x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 956.863 | 523.711 | 578.467 | 0.547x | 0.605x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 93.565 | 532.012 | 598.447 | 5.686x | 6.396x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.591 | 6.361 | 8.355 | 10.756x | 14.129x |
| INT8 | MUL | allocating | 7 | NEON | 0.634 | 6.347 | 7.941 | 10.006x | 12.520x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 968.286 | 541.622 | 575.506 | 0.559x | 0.594x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 119.970 | 541.338 | 582.276 | 4.512x | 4.854x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.226 | 6.333 | 6.405 | 28.014x | 28.333x |
| INT8 | MUL | inplace | 7 | NEON | 0.262 | 6.352 | 6.303 | 24.251x | 24.061x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 910.175 | 526.718 | 575.732 | 0.579x | 0.633x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 88.612 | 528.640 | 574.300 | 5.966x | 6.481x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.591 | 7.075 | 8.924 | 11.977x | 15.106x |
| INT8 | DIV | allocating | 7 | NEON | 0.642 | 7.030 | 9.217 | 10.954x | 14.362x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1677.658 | 2725.708 | 2055.135 | 1.625x | 1.225x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 575.421 | 2726.579 | 2054.669 | 4.738x | 3.571x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.230 | 7.070 | 8.058 | 30.793x | 35.095x |
| INT8 | DIV | inplace | 7 | NEON | 0.260 | 7.036 | 8.804 | 27.060x | 33.858x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1618.467 | 2719.616 | 2040.309 | 1.680x | 1.261x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 518.177 | 2729.184 | 2045.047 | 5.267x | 3.947x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.807 | 1.400 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.646 | 0.800 | 1.380 | 1.239x | 2.138x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1958.728 | 127.819 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1005.393 | 1960.475 | 131.694 | 1.950x | 0.131x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.941 | 1.260 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.263 | 0.941 | 1.263 | 3.573x | 4.800x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1887.984 | 112.323 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 895.346 | 1881.263 | 105.247 | 2.101x | 0.118x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.813 | 1.469 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.641 | 0.809 | 1.394 | 1.261x | 2.174x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1967.217 | 125.064 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1006.287 | 1978.731 | 129.007 | 1.966x | 0.128x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.934 | 1.314 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.264 | 0.936 | 1.353 | 3.546x | 5.125x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1899.140 | 112.480 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 894.662 | 1903.551 | 120.461 | 2.128x | 0.135x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.802 | 1.394 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.660 | 0.801 | 1.376 | 1.214x | 2.085x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1973.944 | 123.790 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1006.438 | 1972.487 | 126.587 | 1.960x | 0.126x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.934 | 1.221 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.261 | 0.932 | 1.234 | 3.578x | 4.739x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1900.537 | 109.518 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 894.528 | 1899.470 | 104.572 | 2.123x | 0.117x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.792 | 1.417 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.661 | 0.792 | 1.396 | 1.198x | 2.111x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2176.483 | 356.277 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1299.087 | 2256.596 | 356.335 | 1.737x | 0.274x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.917 | 1.502 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.254 | 0.928 | 1.554 | 3.646x | 6.110x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2101.523 | 349.805 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1190.580 | 2101.780 | 348.792 | 1.765x | 0.293x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.328 | 8.415 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.652 | 6.274 | 8.257 | 9.618x | 12.657x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1237.300 | 1316.987 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 229.302 | 1252.983 | 1320.966 | 5.464x | 5.761x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.296 | 6.741 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.257 | 6.310 | 7.508 | 24.524x | 29.183x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1170.415 | 1348.053 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 172.586 | 1172.606 | 1324.049 | 6.794x | 7.672x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.228 | 8.701 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.638 | 6.250 | 9.148 | 9.794x | 14.336x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1257.020 | 1343.061 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 231.578 | 1245.643 | 1355.861 | 5.379x | 5.855x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.277 | 7.531 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.263 | 6.297 | 7.865 | 23.968x | 29.935x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1195.003 | 1334.138 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 174.125 | 1177.712 | 1329.738 | 6.764x | 7.637x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.276 | 8.191 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.636 | 6.276 | 7.935 | 9.875x | 12.485x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1331.269 | 1312.870 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 249.150 | 1372.928 | 1297.007 | 5.510x | 5.206x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.289 | 6.284 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.259 | 6.280 | 6.193 | 24.272x | 23.935x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1301.595 | 1281.191 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 175.327 | 1300.702 | 1279.675 | 7.419x | 7.299x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.856 | 9.438 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.642 | 6.938 | 9.086 | 10.808x | 14.155x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3588.818 | 2627.204 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 612.364 | 3567.139 | 2627.425 | 5.825x | 4.291x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.882 | 7.747 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.259 | 6.913 | 8.870 | 26.735x | 34.301x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3572.839 | 2629.813 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 498.408 | 3558.098 | 2620.794 | 7.139x | 5.258x |

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
