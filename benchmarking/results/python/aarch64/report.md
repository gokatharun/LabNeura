# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-20T19:21:48.291698+00:00
- Commit: `32726d8`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.711 | 0.796 | 1.428 | 1.120x | 2.009x |
| FP32 | ADD | allocating | 7 | NEON | 0.615 | 0.788 | 1.395 | 1.281x | 2.268x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 472.677 | 520.069 | 291.073 | 1.100x | 0.616x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 398.483 | 494.495 | 294.736 | 1.241x | 0.740x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.228 | 0.922 | 1.229 | 4.045x | 5.394x |
| FP32 | ADD | inplace | 7 | NEON | 0.258 | 0.919 | 1.228 | 3.566x | 4.765x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 344.863 | 243.960 | 227.427 | 0.707x | 0.659x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 360.614 | 234.569 | 243.230 | 0.650x | 0.674x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.609 | 0.780 | 1.429 | 1.281x | 2.347x |
| FP32 | SUB | allocating | 7 | NEON | 0.599 | 0.777 | 1.416 | 1.297x | 2.363x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 476.245 | 495.951 | 277.386 | 1.041x | 0.582x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 510.468 | 519.771 | 366.904 | 1.018x | 0.719x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.230 | 0.924 | 1.285 | 4.018x | 5.585x |
| FP32 | SUB | inplace | 7 | NEON | 0.260 | 0.920 | 1.246 | 3.536x | 4.785x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 348.971 | 244.738 | 290.379 | 0.701x | 0.832x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 366.765 | 287.547 | 267.216 | 0.784x | 0.729x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.623 | 0.794 | 1.382 | 1.275x | 2.219x |
| FP32 | MUL | allocating | 7 | NEON | 0.600 | 0.792 | 1.378 | 1.319x | 2.296x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 473.568 | 536.037 | 304.036 | 1.132x | 0.642x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 404.743 | 577.112 | 363.982 | 1.426x | 0.899x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.230 | 0.933 | 1.202 | 4.056x | 5.226x |
| FP32 | MUL | inplace | 7 | NEON | 0.255 | 0.929 | 1.201 | 3.637x | 4.702x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 347.795 | 231.173 | 278.201 | 0.665x | 0.800x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 351.429 | 252.499 | 273.288 | 0.718x | 0.778x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.618 | 0.776 | 1.404 | 1.255x | 2.272x |
| FP32 | DIV | allocating | 7 | NEON | 0.601 | 0.781 | 1.396 | 1.300x | 2.324x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 686.116 | 748.832 | 498.749 | 1.091x | 0.727x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 478.514 | 745.076 | 488.028 | 1.557x | 1.020x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.226 | 0.913 | 1.568 | 4.045x | 6.946x |
| FP32 | DIV | inplace | 7 | NEON | 0.258 | 0.914 | 1.521 | 3.537x | 5.887x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 469.159 | 470.087 | 481.320 | 1.002x | 1.026x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 464.831 | 474.390 | 481.542 | 1.021x | 1.036x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.631 | 6.354 | 8.341 | 10.065x | 13.212x |
| INT8 | ADD | allocating | 7 | NEON | 0.672 | 6.254 | 9.097 | 9.310x | 13.542x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1016.576 | 565.481 | 852.145 | 0.556x | 0.838x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 117.843 | 569.907 | 627.741 | 4.836x | 5.327x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.237 | 6.366 | 6.810 | 26.829x | 28.700x |
| INT8 | ADD | inplace | 7 | NEON | 0.278 | 6.344 | 6.871 | 22.803x | 24.695x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 957.423 | 548.553 | 606.413 | 0.573x | 0.633x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 90.734 | 539.916 | 606.842 | 5.951x | 6.688x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.619 | 6.321 | 8.608 | 10.211x | 13.907x |
| INT8 | SUB | allocating | 7 | NEON | 0.668 | 6.435 | 8.896 | 9.634x | 13.319x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1015.851 | 574.844 | 614.194 | 0.566x | 0.605x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 115.171 | 557.088 | 621.268 | 4.837x | 5.394x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.233 | 6.425 | 7.547 | 27.563x | 32.374x |
| INT8 | SUB | inplace | 7 | NEON | 0.270 | 6.320 | 8.114 | 23.401x | 30.044x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 956.151 | 551.310 | 615.033 | 0.577x | 0.643x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 90.673 | 551.200 | 612.424 | 6.079x | 6.754x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.625 | 6.272 | 8.732 | 10.035x | 13.970x |
| INT8 | MUL | allocating | 7 | NEON | 0.687 | 6.298 | 8.433 | 9.169x | 12.278x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 972.934 | 566.060 | 598.434 | 0.582x | 0.615x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 126.581 | 580.182 | 608.292 | 4.583x | 4.806x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.233 | 6.321 | 6.081 | 27.136x | 26.106x |
| INT8 | MUL | inplace | 7 | NEON | 0.269 | 6.326 | 6.282 | 23.551x | 23.386x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 911.579 | 548.479 | 601.143 | 0.602x | 0.659x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 92.000 | 574.642 | 607.289 | 6.246x | 6.601x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.613 | 7.014 | 8.709 | 11.449x | 14.216x |
| INT8 | DIV | allocating | 7 | NEON | 0.682 | 7.004 | 8.846 | 10.271x | 12.973x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1677.207 | 2918.615 | 2091.493 | 1.740x | 1.247x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 572.134 | 2893.407 | 2075.848 | 5.057x | 3.628x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.239 | 6.987 | 7.343 | 29.192x | 30.679x |
| INT8 | DIV | inplace | 7 | NEON | 0.274 | 7.096 | 7.048 | 25.924x | 25.747x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1614.741 | 2955.389 | 2079.822 | 1.830x | 1.288x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 501.247 | 2892.494 | 2064.575 | 5.771x | 4.119x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.810 | 1.418 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.660 | 0.812 | 1.386 | 1.229x | 2.098x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1964.047 | 175.773 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1012.387 | 1968.644 | 197.613 | 1.945x | 0.195x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.945 | 1.252 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.268 | 0.944 | 1.239 | 3.517x | 4.617x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1896.572 | 143.468 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 900.816 | 1891.885 | 159.256 | 2.100x | 0.177x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.802 | 1.399 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.660 | 0.803 | 1.400 | 1.217x | 2.122x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1983.886 | 187.893 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1010.962 | 1992.902 | 168.451 | 1.971x | 0.167x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.943 | 1.273 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.276 | 0.936 | 1.252 | 3.389x | 4.535x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1910.023 | 151.886 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 901.362 | 1906.658 | 147.121 | 2.115x | 0.163x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.811 | 1.358 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.668 | 0.802 | 1.367 | 1.200x | 2.045x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1979.616 | 166.719 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1013.146 | 1988.172 | 162.922 | 1.962x | 0.161x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.942 | 1.203 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.270 | 0.940 | 1.206 | 3.488x | 4.474x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1902.352 | 153.494 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 900.700 | 1905.601 | 140.073 | 2.116x | 0.156x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.793 | 1.406 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.675 | 0.791 | 1.388 | 1.170x | 2.055x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2196.043 | 364.175 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1310.068 | 2193.591 | 365.176 | 1.674x | 0.279x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.925 | 1.549 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.272 | 0.932 | 1.546 | 3.422x | 5.675x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2110.029 | 352.414 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1195.231 | 2106.943 | 353.734 | 1.763x | 0.296x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.268 | 8.859 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.670 | 6.248 | 7.865 | 9.325x | 11.738x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1314.048 | 1437.982 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 240.949 | 1409.849 | 1472.067 | 5.851x | 6.109x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.337 | 7.266 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.266 | 6.258 | 7.546 | 23.520x | 28.361x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1276.065 | 1429.892 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 194.458 | 1282.218 | 1409.156 | 6.594x | 7.247x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.240 | 9.321 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.666 | 6.222 | 9.333 | 9.348x | 14.022x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1357.551 | 1441.745 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 237.880 | 1332.462 | 1439.988 | 5.601x | 6.053x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.286 | 7.917 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.265 | 6.313 | 8.183 | 23.813x | 30.869x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1258.815 | 1399.414 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 182.794 | 1279.950 | 1482.596 | 7.002x | 8.111x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.217 | 8.065 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.663 | 6.216 | 7.794 | 9.382x | 11.765x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1438.613 | 1402.682 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 277.106 | 1477.538 | 1406.007 | 5.332x | 5.074x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.236 | 6.821 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.268 | 6.277 | 6.295 | 23.408x | 23.477x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1409.087 | 1429.097 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 188.381 | 1403.003 | 1389.252 | 7.448x | 7.375x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.948 | 8.566 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.665 | 6.908 | 8.858 | 10.395x | 13.329x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3778.702 | 2676.254 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 610.846 | 3999.214 | 2685.301 | 6.547x | 4.396x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.977 | 8.413 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.269 | 6.910 | 7.468 | 25.647x | 27.718x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3712.770 | 2668.657 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 498.090 | 3889.986 | 2679.066 | 7.810x | 5.379x |

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
