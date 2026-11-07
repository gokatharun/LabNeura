# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-07T15:40:01.377759+00:00
- Commit: `47369a6`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.588 | 0.793 | 1.377 | 1.349x | 2.342x |
| FP32 | ADD | allocating | 7 | NEON | 0.574 | 0.795 | 1.382 | 1.384x | 2.408x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 567.315 | 676.408 | 408.212 | 1.192x | 0.720x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 428.971 | 739.088 | 422.826 | 1.723x | 0.986x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.220 | 0.931 | 1.243 | 4.228x | 5.644x |
| FP32 | ADD | inplace | 7 | NEON | 0.261 | 0.933 | 1.241 | 3.570x | 4.745x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 372.411 | 314.518 | 316.652 | 0.845x | 0.850x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 384.462 | 312.671 | 328.842 | 0.813x | 0.855x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.581 | 0.792 | 1.437 | 1.364x | 2.473x |
| FP32 | SUB | allocating | 7 | NEON | 0.566 | 0.794 | 1.410 | 1.403x | 2.491x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 502.633 | 603.681 | 415.402 | 1.201x | 0.826x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 453.856 | 656.781 | 409.264 | 1.447x | 0.902x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.229 | 0.937 | 1.361 | 4.097x | 5.953x |
| FP32 | SUB | inplace | 7 | NEON | 0.251 | 0.934 | 1.335 | 3.722x | 5.318x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 368.059 | 289.174 | 338.715 | 0.786x | 0.920x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 392.447 | 331.116 | 359.392 | 0.844x | 0.916x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.594 | 0.821 | 1.370 | 1.382x | 2.306x |
| FP32 | MUL | allocating | 7 | NEON | 0.583 | 0.822 | 1.375 | 1.410x | 2.360x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 510.566 | 682.818 | 413.564 | 1.337x | 0.810x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 433.831 | 687.831 | 420.556 | 1.585x | 0.969x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.227 | 0.965 | 1.219 | 4.247x | 5.368x |
| FP32 | MUL | inplace | 7 | NEON | 0.257 | 0.959 | 1.204 | 3.736x | 4.690x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 374.029 | 308.923 | 315.368 | 0.826x | 0.843x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 403.107 | 347.647 | 358.246 | 0.862x | 0.889x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.583 | 0.788 | 1.382 | 1.352x | 2.371x |
| FP32 | DIV | allocating | 7 | NEON | 0.578 | 0.785 | 1.356 | 1.358x | 2.348x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 722.015 | 740.636 | 510.630 | 1.026x | 0.707x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 509.056 | 799.808 | 509.202 | 1.571x | 1.000x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.228 | 0.922 | 1.569 | 4.039x | 6.871x |
| FP32 | DIV | inplace | 7 | NEON | 0.262 | 0.925 | 1.534 | 3.534x | 5.861x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 475.240 | 480.882 | 493.289 | 1.012x | 1.038x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 485.094 | 478.436 | 499.991 | 0.986x | 1.031x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.604 | 6.434 | 9.380 | 10.650x | 15.526x |
| INT8 | ADD | allocating | 7 | NEON | 0.667 | 6.436 | 8.659 | 9.644x | 12.976x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1023.293 | 596.981 | 645.588 | 0.583x | 0.631x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 134.565 | 583.699 | 623.502 | 4.338x | 4.633x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.247 | 6.399 | 6.731 | 25.907x | 27.251x |
| INT8 | ADD | inplace | 7 | NEON | 0.276 | 6.370 | 6.968 | 23.094x | 25.262x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 957.820 | 594.384 | 645.268 | 0.621x | 0.674x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 100.239 | 649.808 | 624.159 | 6.483x | 6.227x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.598 | 6.371 | 10.436 | 10.647x | 17.441x |
| INT8 | SUB | allocating | 7 | NEON | 0.650 | 6.405 | 10.137 | 9.849x | 15.587x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1021.628 | 578.784 | 644.374 | 0.567x | 0.631x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 124.287 | 626.603 | 654.012 | 5.042x | 5.262x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.239 | 6.493 | 7.506 | 27.219x | 31.468x |
| INT8 | SUB | inplace | 7 | NEON | 0.264 | 6.530 | 7.312 | 24.782x | 27.751x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 957.922 | 592.535 | 634.485 | 0.619x | 0.662x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 95.740 | 581.704 | 628.596 | 6.076x | 6.566x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.595 | 6.399 | 8.879 | 10.761x | 14.931x |
| INT8 | MUL | allocating | 7 | NEON | 0.655 | 6.394 | 8.475 | 9.767x | 12.945x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 976.097 | 607.613 | 633.931 | 0.622x | 0.649x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 132.848 | 610.152 | 657.060 | 4.593x | 4.946x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.240 | 6.408 | 5.964 | 26.757x | 24.904x |
| INT8 | MUL | inplace | 7 | NEON | 0.272 | 6.379 | 6.851 | 23.466x | 25.203x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 912.573 | 566.754 | 621.455 | 0.621x | 0.681x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 106.093 | 567.949 | 605.909 | 5.353x | 5.711x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.592 | 7.112 | 9.879 | 12.008x | 16.679x |
| INT8 | DIV | allocating | 7 | NEON | 0.650 | 7.124 | 8.667 | 10.964x | 13.340x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1684.659 | 3042.233 | 2112.211 | 1.806x | 1.254x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 579.881 | 2908.601 | 2087.518 | 5.016x | 3.600x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.235 | 7.033 | 9.405 | 29.863x | 39.937x |
| INT8 | DIV | inplace | 7 | NEON | 0.278 | 7.090 | 7.302 | 25.469x | 26.229x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1616.265 | 2851.822 | 2074.903 | 1.764x | 1.284x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 513.507 | 3082.510 | 2088.259 | 6.003x | 4.067x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.820 | 1.394 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.653 | 0.808 | 1.395 | 1.237x | 2.135x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1956.169 | 146.848 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1017.245 | 1968.660 | 162.919 | 1.935x | 0.160x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.952 | 1.223 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.263 | 0.949 | 1.252 | 3.605x | 4.758x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1884.505 | 148.941 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 900.492 | 1889.771 | 136.263 | 2.099x | 0.151x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.821 | 1.424 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.636 | 0.824 | 1.395 | 1.297x | 2.195x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1990.914 | 161.755 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1012.583 | 1986.938 | 145.635 | 1.962x | 0.144x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.961 | 1.298 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.271 | 0.957 | 1.308 | 3.530x | 4.827x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1908.430 | 134.863 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 900.474 | 1903.824 | 135.143 | 2.114x | 0.150x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.810 | 1.391 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.630 | 0.809 | 1.376 | 1.284x | 2.184x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1975.470 | 176.467 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1018.049 | 1976.990 | 157.790 | 1.942x | 0.155x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.949 | 1.194 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.264 | 0.952 | 1.219 | 3.607x | 4.619x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1900.540 | 146.223 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 896.907 | 1901.433 | 135.338 | 2.120x | 0.151x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.822 | 1.410 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.638 | 0.816 | 1.380 | 1.279x | 2.162x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2191.590 | 360.903 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1306.760 | 2177.395 | 355.878 | 1.666x | 0.272x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.961 | 1.521 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.264 | 0.967 | 1.544 | 3.671x | 5.858x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2112.633 | 349.459 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1203.160 | 2099.610 | 349.039 | 1.745x | 0.290x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.390 | 8.393 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.654 | 6.305 | 8.175 | 9.642x | 12.502x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1360.641 | 1390.879 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 253.749 | 1340.304 | 1472.566 | 5.282x | 5.803x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.325 | 7.402 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.272 | 6.307 | 6.827 | 23.175x | 25.086x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1239.825 | 1404.337 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 185.195 | 1233.181 | 1327.488 | 6.659x | 7.168x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.284 | 9.968 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.643 | 6.428 | 9.393 | 10.004x | 14.619x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1271.604 | 1391.110 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 248.264 | 1411.018 | 1480.952 | 5.684x | 5.965x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.293 | 7.451 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.270 | 6.296 | 7.181 | 23.356x | 26.637x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1229.940 | 1365.766 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 185.958 | 1295.248 | 1475.014 | 6.965x | 7.932x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.298 | 8.025 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.639 | 6.280 | 8.155 | 9.826x | 12.758x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1540.597 | 1450.538 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 261.699 | 1446.013 | 1465.298 | 5.525x | 5.599x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.280 | 6.107 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.265 | 6.311 | 6.346 | 23.820x | 23.954x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1360.927 | 1430.689 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 180.797 | 1383.677 | 1390.016 | 7.653x | 7.688x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 7.022 | 8.454 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.645 | 7.029 | 9.560 | 10.904x | 14.831x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3580.325 | 2628.289 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 617.327 | 3739.787 | 2688.062 | 6.058x | 4.354x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.995 | 8.030 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.280 | 6.978 | 7.756 | 24.952x | 27.735x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3484.890 | 2600.381 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 498.358 | 3954.272 | 2652.165 | 7.935x | 5.322x |

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
