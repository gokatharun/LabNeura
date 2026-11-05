# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-05T20:24:06.788163+00:00
- Commit: `b91df83`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.470 | 0.789 | 1.439 | 1.677x | 3.060x |
| FP32 | ADD | allocating | 7 | NEON | 0.486 | 0.782 | 1.374 | 1.607x | 2.827x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 456.847 | 199.347 | 220.651 | 0.436x | 0.483x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 469.674 | 209.301 | 241.198 | 0.446x | 0.514x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.217 | 0.903 | 1.247 | 4.162x | 5.747x |
| FP32 | ADD | inplace | 7 | NEON | 0.220 | 0.904 | 1.231 | 4.107x | 5.593x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 339.787 | 178.516 | 221.470 | 0.525x | 0.652x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 358.394 | 189.488 | 254.822 | 0.529x | 0.711x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.472 | 0.787 | 1.410 | 1.669x | 2.989x |
| FP32 | SUB | allocating | 7 | NEON | 0.482 | 0.777 | 1.387 | 1.611x | 2.877x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 459.077 | 200.615 | 236.872 | 0.437x | 0.516x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 459.763 | 466.726 | 304.928 | 1.015x | 0.663x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.210 | 0.922 | 1.258 | 4.397x | 5.998x |
| FP32 | SUB | inplace | 7 | NEON | 0.216 | 0.911 | 1.266 | 4.216x | 5.862x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 338.311 | 180.768 | 233.020 | 0.534x | 0.689x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 341.185 | 194.416 | 225.434 | 0.570x | 0.661x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.472 | 0.800 | 1.363 | 1.694x | 2.886x |
| FP32 | MUL | allocating | 7 | NEON | 0.483 | 0.788 | 1.355 | 1.630x | 2.804x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 450.159 | 197.653 | 261.577 | 0.439x | 0.581x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 466.559 | 202.835 | 291.233 | 0.435x | 0.624x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.210 | 0.931 | 1.201 | 4.423x | 5.706x |
| FP32 | MUL | inplace | 7 | NEON | 0.206 | 0.933 | 1.215 | 4.521x | 5.885x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 335.263 | 177.552 | 228.265 | 0.530x | 0.681x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 334.405 | 189.160 | 225.317 | 0.566x | 0.674x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.474 | 0.783 | 1.356 | 1.652x | 2.859x |
| FP32 | DIV | allocating | 7 | NEON | 0.494 | 0.778 | 1.377 | 1.575x | 2.787x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 679.527 | 453.411 | 496.524 | 0.667x | 0.731x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 604.524 | 694.921 | 504.679 | 1.150x | 0.835x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.207 | 0.917 | 1.628 | 4.426x | 7.854x |
| FP32 | DIV | inplace | 7 | NEON | 0.210 | 0.924 | 1.716 | 4.405x | 8.180x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 471.864 | 452.887 | 463.800 | 0.960x | 0.983x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 427.044 | 457.409 | 476.766 | 1.071x | 1.116x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.473 | 6.362 | 9.674 | 13.442x | 20.439x |
| INT8 | ADD | allocating | 7 | NEON | 0.488 | 6.286 | 9.541 | 12.888x | 19.563x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1013.306 | 519.355 | 581.217 | 0.513x | 0.574x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 109.360 | 539.562 | 586.981 | 4.934x | 5.367x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.215 | 6.321 | 7.404 | 29.441x | 34.482x |
| INT8 | ADD | inplace | 7 | NEON | 0.220 | 6.303 | 6.911 | 28.669x | 31.435x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 956.831 | 516.100 | 569.362 | 0.539x | 0.595x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 88.996 | 527.436 | 577.979 | 5.926x | 6.494x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.482 | 6.299 | 9.229 | 13.062x | 19.138x |
| INT8 | SUB | allocating | 7 | NEON | 0.485 | 6.281 | 9.301 | 12.938x | 19.158x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1013.680 | 518.984 | 591.376 | 0.512x | 0.583x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 110.868 | 543.564 | 604.019 | 4.903x | 5.448x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.220 | 6.330 | 7.258 | 28.834x | 33.062x |
| INT8 | SUB | inplace | 7 | NEON | 0.217 | 6.300 | 8.055 | 29.082x | 37.179x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 957.266 | 517.711 | 580.105 | 0.541x | 0.606x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 87.984 | 520.664 | 583.236 | 5.918x | 6.629x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.480 | 6.283 | 9.423 | 13.077x | 19.612x |
| INT8 | MUL | allocating | 7 | NEON | 0.495 | 6.300 | 8.079 | 12.718x | 16.308x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 968.814 | 517.413 | 583.353 | 0.534x | 0.602x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 119.548 | 547.039 | 577.493 | 4.576x | 4.831x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.215 | 6.316 | 6.989 | 29.410x | 32.544x |
| INT8 | MUL | inplace | 7 | NEON | 0.218 | 6.304 | 6.418 | 28.930x | 29.450x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 911.217 | 518.491 | 566.973 | 0.569x | 0.622x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 88.273 | 525.795 | 573.255 | 5.956x | 6.494x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.479 | 6.987 | 9.599 | 14.576x | 20.024x |
| INT8 | DIV | allocating | 7 | NEON | 0.477 | 6.925 | 9.066 | 14.505x | 18.987x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1672.203 | 2688.500 | 2058.956 | 1.608x | 1.231x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 571.635 | 2783.018 | 2057.997 | 4.869x | 3.600x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.215 | 7.052 | 7.849 | 32.843x | 36.556x |
| INT8 | DIV | inplace | 7 | NEON | 0.224 | 6.995 | 8.003 | 31.250x | 35.755x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1614.530 | 2703.574 | 2044.921 | 1.675x | 1.267x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 513.501 | 2739.350 | 2051.452 | 5.335x | 3.995x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.812 | 1.445 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.489 | 0.813 | 1.385 | 1.663x | 2.834x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1951.682 | 138.760 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1004.953 | 1889.301 | 148.880 | 1.880x | 0.148x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.942 | 1.238 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.215 | 0.926 | 1.233 | 4.301x | 5.723x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1880.799 | 117.505 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 897.986 | 1876.443 | 123.759 | 2.090x | 0.138x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.809 | 1.387 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.475 | 0.808 | 1.426 | 1.699x | 2.999x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1975.828 | 135.155 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1007.232 | 1898.633 | 147.955 | 1.885x | 0.147x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.934 | 1.410 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.223 | 0.940 | 1.383 | 4.217x | 6.205x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1896.127 | 115.525 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 897.193 | 1898.502 | 117.221 | 2.116x | 0.131x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.803 | 1.399 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.477 | 0.815 | 1.409 | 1.706x | 2.951x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1972.766 | 137.935 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1006.045 | 1983.782 | 141.283 | 1.972x | 0.140x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.934 | 1.219 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.217 | 0.929 | 1.222 | 4.290x | 5.643x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1896.061 | 112.080 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 896.491 | 1896.455 | 112.251 | 2.115x | 0.125x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.794 | 1.413 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.489 | 0.794 | 1.388 | 1.625x | 2.840x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2176.071 | 358.937 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1303.712 | 2102.918 | 360.094 | 1.613x | 0.276x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.921 | 1.705 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.216 | 0.925 | 1.557 | 4.280x | 7.208x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2099.383 | 348.646 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1192.786 | 2099.782 | 348.872 | 1.760x | 0.292x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.230 | 8.551 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.491 | 6.269 | 8.233 | 12.767x | 16.766x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1273.202 | 1448.514 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 239.915 | 1291.173 | 1443.800 | 5.382x | 6.018x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.247 | 7.295 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.216 | 6.320 | 7.085 | 29.192x | 32.729x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1230.653 | 1412.897 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 174.787 | 1220.634 | 1422.393 | 6.984x | 8.138x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.193 | 8.668 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.500 | 6.250 | 9.851 | 12.496x | 19.695x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1270.984 | 1434.963 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 232.047 | 1271.408 | 1432.725 | 5.479x | 6.174x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.250 | 7.410 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.221 | 6.214 | 7.493 | 28.160x | 33.957x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1235.608 | 1433.689 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 176.301 | 1218.301 | 1417.754 | 6.910x | 8.042x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.185 | 7.430 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.493 | 6.184 | 8.921 | 12.536x | 18.085x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1363.223 | 1418.939 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 251.952 | 1406.903 | 1373.965 | 5.584x | 5.453x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.237 | 6.963 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.214 | 6.205 | 6.428 | 28.959x | 30.002x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1337.001 | 1374.726 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 178.016 | 1330.550 | 1377.546 | 7.474x | 7.738x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.904 | 9.470 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.487 | 6.905 | 9.082 | 14.182x | 18.652x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3679.372 | 2677.272 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 612.200 | 3702.840 | 2680.821 | 6.048x | 4.379x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.950 | 7.440 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.214 | 6.966 | 8.486 | 32.611x | 39.724x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3712.869 | 2661.919 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 502.463 | 3801.663 | 2670.390 | 7.566x | 5.315x |

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
