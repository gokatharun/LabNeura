# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-20T07:12:42.429979+00:00
- Commit: `727e44d`
- CPU: aarch64 (aarch64)
- OS: Linux-7.0.0-1017-azure-aarch64-with-glibc2.39
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.4.9
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.599 | 0.787 | 1.396 | 1.313x | 2.330x |
| FP32 | ADD | allocating | 7 | NEON | 0.571 | 0.791 | 1.381 | 1.385x | 2.420x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 499.232 | 573.181 | 349.256 | 1.148x | 0.700x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 410.173 | 590.261 | 313.781 | 1.439x | 0.765x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.219 | 0.924 | 1.270 | 4.226x | 5.808x |
| FP32 | ADD | inplace | 7 | NEON | 0.252 | 0.914 | 1.277 | 3.630x | 5.073x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 362.529 | 253.559 | 261.073 | 0.699x | 0.720x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 363.407 | 309.629 | 274.632 | 0.852x | 0.756x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.582 | 0.775 | 1.385 | 1.331x | 2.379x |
| FP32 | SUB | allocating | 7 | NEON | 0.561 | 0.787 | 1.405 | 1.403x | 2.505x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 498.675 | 532.401 | 346.798 | 1.068x | 0.695x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 471.635 | 641.127 | 381.623 | 1.359x | 0.809x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.221 | 0.913 | 1.347 | 4.133x | 6.095x |
| FP32 | SUB | inplace | 7 | NEON | 0.253 | 0.924 | 1.363 | 3.649x | 5.380x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 366.567 | 304.651 | 279.422 | 0.831x | 0.762x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 388.058 | 275.404 | 295.751 | 0.710x | 0.762x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.584 | 0.816 | 1.391 | 1.399x | 2.383x |
| FP32 | MUL | allocating | 7 | NEON | 0.561 | 0.819 | 1.353 | 1.459x | 2.410x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 499.234 | 592.756 | 346.456 | 1.187x | 0.694x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 415.985 | 609.306 | 329.082 | 1.465x | 0.791x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.222 | 0.954 | 1.197 | 4.302x | 5.397x |
| FP32 | MUL | inplace | 7 | NEON | 0.253 | 0.963 | 1.219 | 3.800x | 4.812x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 361.289 | 250.589 | 279.282 | 0.694x | 0.773x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 382.818 | 300.967 | 314.289 | 0.786x | 0.821x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.571 | 0.779 | 1.409 | 1.365x | 2.468x |
| FP32 | DIV | allocating | 7 | NEON | 0.568 | 0.783 | 1.368 | 1.380x | 2.411x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 693.703 | 761.171 | 492.069 | 1.097x | 0.709x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 480.375 | 770.485 | 499.625 | 1.604x | 1.040x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.223 | 0.919 | 1.555 | 4.118x | 6.965x |
| FP32 | DIV | inplace | 7 | NEON | 0.246 | 0.915 | 1.579 | 3.712x | 6.409x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 464.148 | 466.335 | 482.313 | 1.005x | 1.039x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 466.909 | 476.535 | 484.673 | 1.021x | 1.038x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.596 | 6.426 | 8.915 | 10.778x | 14.954x |
| INT8 | ADD | allocating | 7 | NEON | 0.643 | 6.427 | 8.206 | 10.000x | 12.767x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1014.693 | 572.215 | 601.788 | 0.564x | 0.593x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 110.521 | 549.872 | 616.402 | 4.975x | 5.577x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.221 | 6.369 | 7.371 | 28.868x | 33.406x |
| INT8 | ADD | inplace | 7 | NEON | 0.256 | 6.385 | 6.853 | 24.909x | 26.736x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 956.176 | 527.898 | 611.735 | 0.552x | 0.640x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 92.037 | 542.919 | 606.898 | 5.899x | 6.594x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.574 | 6.348 | 9.194 | 11.064x | 16.023x |
| INT8 | SUB | allocating | 7 | NEON | 0.651 | 6.370 | 9.689 | 9.789x | 14.889x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1012.862 | 535.484 | 601.076 | 0.529x | 0.593x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 109.621 | 549.109 | 592.666 | 5.009x | 5.407x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.223 | 6.358 | 8.071 | 28.567x | 36.260x |
| INT8 | SUB | inplace | 7 | NEON | 0.255 | 6.312 | 8.343 | 24.718x | 32.672x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 956.266 | 528.471 | 585.603 | 0.553x | 0.612x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 87.073 | 530.121 | 631.086 | 6.088x | 7.248x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.606 | 6.369 | 8.237 | 10.508x | 13.590x |
| INT8 | MUL | allocating | 7 | NEON | 0.656 | 6.371 | 8.154 | 9.707x | 12.423x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 966.772 | 552.540 | 583.428 | 0.572x | 0.603x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 117.987 | 550.883 | 580.791 | 4.669x | 4.923x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.225 | 6.342 | 6.562 | 28.188x | 29.167x |
| INT8 | MUL | inplace | 7 | NEON | 0.256 | 6.374 | 6.479 | 24.944x | 25.355x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 911.063 | 531.366 | 569.957 | 0.583x | 0.626x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 89.694 | 534.161 | 576.172 | 5.955x | 6.424x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.604 | 7.065 | 9.575 | 11.693x | 15.847x |
| INT8 | DIV | allocating | 7 | NEON | 0.652 | 7.052 | 10.628 | 10.823x | 16.311x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1679.684 | 2756.070 | 2061.365 | 1.641x | 1.227x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 568.095 | 2735.107 | 2054.012 | 4.815x | 3.616x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.224 | 7.038 | 8.649 | 31.352x | 38.527x |
| INT8 | DIV | inplace | 7 | NEON | 0.266 | 7.026 | 9.913 | 26.372x | 37.208x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1618.630 | 2758.584 | 2060.781 | 1.704x | 1.273x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 512.206 | 2782.259 | 2051.273 | 5.432x | 4.005x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.809 | 1.415 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.635 | 0.805 | 1.384 | 1.269x | 2.180x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1962.571 | 140.036 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1005.715 | 1975.075 | 142.477 | 1.964x | 0.142x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.930 | 1.263 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.253 | 0.918 | 1.262 | 3.630x | 4.990x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1882.374 | 120.522 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 898.341 | 1883.461 | 119.821 | 2.097x | 0.133x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.806 | 1.412 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.626 | 0.801 | 1.381 | 1.279x | 2.204x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1984.132 | 143.445 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1006.335 | 1982.254 | 139.397 | 1.970x | 0.139x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.926 | 1.314 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.249 | 0.923 | 1.295 | 3.701x | 5.196x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1901.544 | 120.288 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 895.102 | 1901.995 | 138.123 | 2.125x | 0.154x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.801 | 1.405 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.645 | 0.797 | 1.347 | 1.237x | 2.090x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1995.981 | 152.534 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1005.252 | 1980.455 | 140.418 | 1.970x | 0.140x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.926 | 1.231 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.260 | 0.922 | 1.235 | 3.543x | 4.744x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1898.814 | 130.610 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 895.152 | 1894.463 | 141.790 | 2.116x | 0.158x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.786 | 1.404 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.650 | 0.790 | 1.364 | 1.215x | 2.097x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2193.109 | 359.601 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1304.252 | 2188.348 | 359.974 | 1.678x | 0.276x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.911 | 1.610 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.252 | 0.914 | 1.543 | 3.627x | 6.123x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2102.662 | 348.534 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1191.790 | 2104.494 | 351.135 | 1.766x | 0.295x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.223 | 8.408 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.631 | 6.211 | 9.668 | 9.839x | 15.316x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1333.354 | 1419.575 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 242.133 | 1321.970 | 1424.709 | 5.460x | 5.884x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.232 | 7.225 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.260 | 6.223 | 6.752 | 23.933x | 25.969x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1315.750 | 1440.640 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 187.412 | 1314.786 | 1451.908 | 7.015x | 7.747x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.206 | 9.197 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.641 | 6.184 | 8.830 | 9.656x | 13.786x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1274.165 | 1432.070 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 238.840 | 1333.886 | 1445.891 | 5.585x | 6.054x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.214 | 7.078 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.259 | 6.246 | 8.029 | 24.082x | 30.956x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1316.716 | 1465.763 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 186.464 | 1256.822 | 1411.204 | 6.740x | 7.568x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.227 | 7.839 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.633 | 6.265 | 7.785 | 9.890x | 12.289x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1434.338 | 1398.820 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 249.662 | 1428.888 | 1430.317 | 5.723x | 5.729x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.274 | 6.621 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.255 | 6.269 | 6.497 | 24.597x | 25.491x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1386.238 | 1464.219 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 189.254 | 1357.757 | 1414.638 | 7.174x | 7.475x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.895 | 8.459 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.644 | 6.866 | 9.904 | 10.669x | 15.390x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3957.297 | 2702.494 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 615.671 | 3902.865 | 2685.124 | 6.339x | 4.361x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.888 | 8.489 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.260 | 6.915 | 8.009 | 26.612x | 30.823x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3847.422 | 2677.205 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 503.977 | 3934.838 | 2712.394 | 7.808x | 5.382x |

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
