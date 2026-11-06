# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-05T21:36:42.527759+00:00
- Commit: `659f17b`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.474 | 0.799 | 1.394 | 1.685x | 2.940x |
| FP32 | ADD | allocating | 7 | NEON | 0.501 | 0.795 | 1.405 | 1.586x | 2.805x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 450.145 | 194.951 | 216.308 | 0.433x | 0.481x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 473.484 | 206.365 | 269.239 | 0.436x | 0.569x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.209 | 0.928 | 1.239 | 4.444x | 5.932x |
| FP32 | ADD | inplace | 7 | NEON | 0.217 | 0.930 | 1.278 | 4.288x | 5.890x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 333.673 | 174.184 | 213.147 | 0.522x | 0.639x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 346.748 | 177.284 | 285.082 | 0.511x | 0.822x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.490 | 0.799 | 1.385 | 1.630x | 2.826x |
| FP32 | SUB | allocating | 7 | NEON | 0.496 | 0.785 | 1.384 | 1.583x | 2.792x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 454.633 | 202.349 | 230.686 | 0.445x | 0.507x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 469.258 | 204.261 | 304.253 | 0.435x | 0.648x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.210 | 0.915 | 1.252 | 4.349x | 5.951x |
| FP32 | SUB | inplace | 7 | NEON | 0.214 | 0.922 | 1.310 | 4.312x | 6.128x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 331.854 | 177.022 | 263.151 | 0.533x | 0.793x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 339.162 | 180.214 | 299.552 | 0.531x | 0.883x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.484 | 0.822 | 1.381 | 1.700x | 2.857x |
| FP32 | MUL | allocating | 7 | NEON | 0.490 | 0.823 | 1.387 | 1.681x | 2.832x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 449.343 | 204.661 | 235.214 | 0.455x | 0.523x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 461.426 | 200.675 | 361.046 | 0.435x | 0.782x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.208 | 0.948 | 1.220 | 4.562x | 5.873x |
| FP32 | MUL | inplace | 7 | NEON | 0.214 | 0.955 | 1.211 | 4.467x | 5.667x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 329.925 | 178.694 | 201.971 | 0.542x | 0.612x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 349.224 | 179.600 | 259.035 | 0.514x | 0.742x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.481 | 0.785 | 1.416 | 1.634x | 2.948x |
| FP32 | DIV | allocating | 7 | NEON | 0.480 | 0.781 | 1.374 | 1.627x | 2.861x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 681.461 | 455.115 | 473.946 | 0.668x | 0.695x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 593.550 | 456.819 | 473.696 | 0.770x | 0.798x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.215 | 0.910 | 1.585 | 4.233x | 7.375x |
| FP32 | DIV | inplace | 7 | NEON | 0.209 | 0.910 | 1.563 | 4.350x | 7.473x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 461.745 | 452.501 | 460.631 | 0.980x | 0.998x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 389.215 | 453.363 | 467.134 | 1.165x | 1.200x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.483 | 6.357 | 9.729 | 13.174x | 20.161x |
| INT8 | ADD | allocating | 7 | NEON | 0.479 | 6.307 | 8.751 | 13.161x | 18.262x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1014.429 | 519.624 | 578.620 | 0.512x | 0.570x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 116.717 | 551.561 | 598.357 | 4.726x | 5.127x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.222 | 6.366 | 7.563 | 28.687x | 34.080x |
| INT8 | ADD | inplace | 7 | NEON | 0.218 | 6.308 | 6.938 | 28.965x | 31.859x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 956.982 | 515.071 | 570.024 | 0.538x | 0.596x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 91.367 | 519.543 | 579.489 | 5.686x | 6.342x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.480 | 6.313 | 9.662 | 13.159x | 20.142x |
| INT8 | SUB | allocating | 7 | NEON | 0.489 | 6.330 | 9.038 | 12.937x | 18.470x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1012.866 | 515.441 | 581.018 | 0.509x | 0.574x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 112.991 | 536.697 | 606.934 | 4.750x | 5.372x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.213 | 6.396 | 8.189 | 29.986x | 38.393x |
| INT8 | SUB | inplace | 7 | NEON | 0.214 | 6.330 | 7.684 | 29.635x | 35.976x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 956.133 | 516.375 | 581.602 | 0.540x | 0.608x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 88.727 | 522.410 | 609.517 | 5.888x | 6.870x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.489 | 6.282 | 7.873 | 12.848x | 16.101x |
| INT8 | MUL | allocating | 7 | NEON | 0.493 | 6.302 | 7.860 | 12.771x | 15.928x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 966.850 | 526.729 | 577.911 | 0.545x | 0.598x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 117.821 | 538.656 | 591.507 | 4.572x | 5.020x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.212 | 6.316 | 6.437 | 29.771x | 30.340x |
| INT8 | MUL | inplace | 7 | NEON | 0.214 | 6.327 | 7.536 | 29.598x | 35.252x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 910.161 | 518.671 | 563.585 | 0.570x | 0.619x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 91.423 | 522.749 | 566.449 | 5.718x | 6.196x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.481 | 7.017 | 10.620 | 14.604x | 22.102x |
| INT8 | DIV | allocating | 7 | NEON | 0.479 | 7.006 | 9.910 | 14.620x | 20.680x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1671.086 | 2659.237 | 2045.485 | 1.591x | 1.224x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 557.117 | 2728.119 | 2058.943 | 4.897x | 3.696x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.214 | 6.991 | 8.311 | 32.655x | 38.820x |
| INT8 | DIV | inplace | 7 | NEON | 0.217 | 6.989 | 9.808 | 32.263x | 45.272x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1614.903 | 2678.351 | 2033.118 | 1.659x | 1.259x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 505.615 | 2743.295 | 2061.768 | 5.426x | 4.078x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.828 | 1.368 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.489 | 0.820 | 1.368 | 1.677x | 2.798x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1960.292 | 129.059 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1002.807 | 1951.972 | 129.636 | 1.947x | 0.129x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.945 | 1.232 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.219 | 0.949 | 1.249 | 4.342x | 5.713x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1881.874 | 122.718 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 896.704 | 1885.524 | 110.820 | 2.103x | 0.124x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.820 | 1.377 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.489 | 0.818 | 1.405 | 1.674x | 2.873x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1974.032 | 124.669 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1002.239 | 1980.603 | 132.354 | 1.976x | 0.132x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.941 | 1.289 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.215 | 0.956 | 1.268 | 4.439x | 5.888x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1899.486 | 115.786 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 894.634 | 1897.034 | 121.249 | 2.120x | 0.136x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.822 | 1.402 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.495 | 0.823 | 1.405 | 1.664x | 2.841x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1969.688 | 131.476 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1001.169 | 1977.058 | 129.863 | 1.975x | 0.130x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.948 | 1.212 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.226 | 0.951 | 1.237 | 4.215x | 5.482x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1901.659 | 119.777 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 893.610 | 1896.399 | 116.818 | 2.122x | 0.131x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.820 | 1.417 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.500 | 0.814 | 1.392 | 1.629x | 2.787x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2180.128 | 356.840 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1299.458 | 2168.880 | 357.413 | 1.669x | 0.275x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.952 | 1.637 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.212 | 0.942 | 1.567 | 4.435x | 7.381x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2101.059 | 347.871 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1192.886 | 2097.132 | 348.376 | 1.758x | 0.292x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.234 | 9.072 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.502 | 6.267 | 8.128 | 12.489x | 16.198x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1253.076 | 1429.400 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 230.897 | 1296.951 | 1368.717 | 5.617x | 5.928x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.286 | 6.521 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.216 | 6.330 | 7.445 | 29.345x | 34.516x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1220.424 | 1389.573 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 177.063 | 1204.635 | 1394.703 | 6.803x | 7.877x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.208 | 9.008 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.495 | 6.260 | 9.780 | 12.653x | 19.769x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1247.274 | 1365.617 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 230.939 | 1288.709 | 1409.107 | 5.580x | 6.102x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.219 | 7.105 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.221 | 6.258 | 7.476 | 28.322x | 33.836x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1202.331 | 1368.640 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 173.611 | 1215.238 | 1343.115 | 7.000x | 7.736x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.226 | 8.038 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.497 | 6.220 | 7.955 | 12.508x | 15.997x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1357.619 | 1341.857 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 239.357 | 1383.120 | 1447.732 | 5.778x | 6.048x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.271 | 6.326 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.213 | 6.214 | 7.659 | 29.177x | 35.960x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1296.677 | 1320.313 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 180.226 | 1303.260 | 1375.663 | 7.231x | 7.633x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.957 | 10.827 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.491 | 6.899 | 10.963 | 14.062x | 22.348x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3464.161 | 2641.964 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 605.657 | 3530.341 | 2631.057 | 5.829x | 4.344x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 7.010 | 9.816 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.217 | 6.976 | 8.754 | 32.084x | 40.260x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3536.775 | 2636.740 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 494.781 | 3519.408 | 2646.022 | 7.113x | 5.348x |

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
