# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-07T01:31:14.970749+00:00
- Commit: `d7f5ee0`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.499 | 0.793 | 1.461 | 1.589x | 2.929x |
| FP32 | ADD | allocating | 7 | NEON | 0.496 | 0.791 | 1.387 | 1.594x | 2.795x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 412.345 | 206.480 | 215.805 | 0.501x | 0.523x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 372.500 | 237.304 | 231.167 | 0.637x | 0.621x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.208 | 0.921 | 1.242 | 4.434x | 5.977x |
| FP32 | ADD | inplace | 7 | NEON | 0.209 | 0.925 | 1.248 | 4.427x | 5.971x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 311.792 | 184.527 | 197.108 | 0.592x | 0.632x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 312.658 | 197.820 | 207.341 | 0.633x | 0.663x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.510 | 0.780 | 1.380 | 1.530x | 2.705x |
| FP32 | SUB | allocating | 7 | NEON | 0.496 | 0.778 | 1.411 | 1.568x | 2.843x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 451.520 | 213.687 | 215.981 | 0.473x | 0.478x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 373.300 | 489.054 | 243.580 | 1.310x | 0.653x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.206 | 0.914 | 1.273 | 4.440x | 6.180x |
| FP32 | SUB | inplace | 7 | NEON | 0.198 | 0.912 | 1.289 | 4.601x | 6.502x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 310.786 | 194.494 | 202.628 | 0.626x | 0.652x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 320.082 | 199.967 | 209.885 | 0.625x | 0.656x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.496 | 0.788 | 1.377 | 1.586x | 2.773x |
| FP32 | MUL | allocating | 7 | NEON | 0.494 | 0.790 | 1.377 | 1.597x | 2.784x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 434.880 | 215.527 | 224.945 | 0.496x | 0.517x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 387.624 | 211.913 | 216.591 | 0.547x | 0.559x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.201 | 0.925 | 1.217 | 4.597x | 6.047x |
| FP32 | MUL | inplace | 7 | NEON | 0.203 | 0.919 | 1.198 | 4.518x | 5.892x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 320.398 | 188.614 | 183.526 | 0.589x | 0.573x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 343.021 | 184.530 | 186.722 | 0.538x | 0.544x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.483 | 0.769 | 1.366 | 1.592x | 2.828x |
| FP32 | DIV | allocating | 7 | NEON | 0.504 | 0.771 | 1.365 | 1.528x | 2.706x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 667.821 | 456.805 | 470.251 | 0.684x | 0.704x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 469.807 | 716.655 | 477.237 | 1.525x | 1.016x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.201 | 0.900 | 1.803 | 4.466x | 8.952x |
| FP32 | DIV | inplace | 7 | NEON | 0.199 | 0.901 | 1.683 | 4.532x | 8.469x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 457.837 | 453.767 | 461.509 | 0.991x | 1.008x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 464.447 | 464.499 | 477.949 | 1.000x | 1.029x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.509 | 6.376 | 8.867 | 12.523x | 17.417x |
| INT8 | ADD | allocating | 7 | NEON | 0.514 | 6.319 | 8.903 | 12.299x | 17.329x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1012.239 | 515.780 | 580.186 | 0.510x | 0.573x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 113.408 | 554.817 | 589.801 | 4.892x | 5.201x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.206 | 6.413 | 7.355 | 31.165x | 35.745x |
| INT8 | ADD | inplace | 7 | NEON | 0.211 | 6.324 | 7.463 | 29.989x | 35.390x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 954.173 | 518.253 | 565.885 | 0.543x | 0.593x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 85.957 | 522.042 | 575.415 | 6.073x | 6.694x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.511 | 6.315 | 9.239 | 12.354x | 18.074x |
| INT8 | SUB | allocating | 7 | NEON | 0.512 | 6.327 | 9.060 | 12.353x | 17.689x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1011.574 | 521.924 | 580.414 | 0.516x | 0.574x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 111.630 | 540.097 | 588.708 | 4.838x | 5.274x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.210 | 6.326 | 9.487 | 30.111x | 45.158x |
| INT8 | SUB | inplace | 7 | NEON | 0.212 | 6.310 | 8.690 | 29.808x | 41.049x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 955.545 | 517.027 | 575.926 | 0.541x | 0.603x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 90.191 | 524.561 | 577.135 | 5.816x | 6.399x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.510 | 6.314 | 7.967 | 12.390x | 15.633x |
| INT8 | MUL | allocating | 7 | NEON | 0.507 | 6.316 | 8.312 | 12.452x | 16.388x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 967.017 | 517.132 | 573.507 | 0.535x | 0.593x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 115.763 | 545.295 | 575.969 | 4.710x | 4.975x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.207 | 6.322 | 6.609 | 30.534x | 31.919x |
| INT8 | MUL | inplace | 7 | NEON | 0.209 | 6.332 | 6.877 | 30.277x | 32.887x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 910.117 | 521.130 | 559.007 | 0.573x | 0.614x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 83.514 | 525.086 | 566.413 | 6.287x | 6.782x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.502 | 7.062 | 10.369 | 14.055x | 20.638x |
| INT8 | DIV | allocating | 7 | NEON | 0.514 | 7.035 | 9.601 | 13.676x | 18.664x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1673.963 | 2664.155 | 2039.153 | 1.592x | 1.218x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 569.183 | 2718.188 | 2060.052 | 4.776x | 3.619x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.210 | 7.070 | 8.531 | 33.630x | 40.580x |
| INT8 | DIV | inplace | 7 | NEON | 0.210 | 7.009 | 8.285 | 33.337x | 39.405x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1614.268 | 2670.898 | 2041.459 | 1.655x | 1.265x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 510.507 | 2717.602 | 2046.958 | 5.323x | 4.010x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.799 | 1.377 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.512 | 0.808 | 1.407 | 1.580x | 2.751x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1949.022 | 126.263 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1003.927 | 1960.353 | 132.123 | 1.953x | 0.132x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.939 | 1.244 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.205 | 0.936 | 1.232 | 4.554x | 5.999x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1885.009 | 113.592 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 895.216 | 1887.676 | 115.716 | 2.109x | 0.129x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.811 | 1.390 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.512 | 0.805 | 1.384 | 1.572x | 2.701x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1904.489 | 125.934 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1003.589 | 1974.323 | 128.572 | 1.967x | 0.128x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.946 | 1.278 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.216 | 0.946 | 1.283 | 4.385x | 5.947x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1895.460 | 114.616 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 904.536 | 1899.903 | 117.179 | 2.100x | 0.130x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.798 | 1.362 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.518 | 0.799 | 1.397 | 1.541x | 2.696x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1897.962 | 126.832 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1004.008 | 1983.407 | 137.860 | 1.975x | 0.137x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.941 | 1.211 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.207 | 0.937 | 1.204 | 4.519x | 5.807x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1897.959 | 112.379 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 895.589 | 1898.730 | 115.566 | 2.120x | 0.129x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.798 | 1.387 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.508 | 0.798 | 1.367 | 1.573x | 2.693x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2115.077 | 358.789 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1298.234 | 2180.094 | 357.574 | 1.679x | 0.275x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.939 | 1.603 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.208 | 0.931 | 1.730 | 4.487x | 8.339x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2099.277 | 349.995 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1188.629 | 2102.087 | 352.535 | 1.768x | 0.297x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.201 | 8.476 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.521 | 6.233 | 8.939 | 11.960x | 17.153x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1231.087 | 1312.793 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 227.939 | 1247.105 | 1307.157 | 5.471x | 5.735x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.251 | 7.329 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.206 | 6.266 | 7.321 | 30.453x | 35.580x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1183.209 | 1334.260 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 148.902 | 1175.773 | 1322.323 | 7.896x | 8.880x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.194 | 9.639 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.511 | 6.220 | 10.748 | 12.179x | 21.044x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1237.359 | 1317.720 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 211.597 | 1231.905 | 1297.377 | 5.822x | 6.131x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.238 | 7.544 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.206 | 6.212 | 8.103 | 30.213x | 39.413x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1175.099 | 1307.220 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 148.654 | 1172.971 | 1300.656 | 7.891x | 8.750x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.193 | 7.638 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.523 | 6.228 | 7.654 | 11.908x | 14.634x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1328.914 | 1278.421 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 240.798 | 1336.203 | 1281.098 | 5.549x | 5.320x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.248 | 6.763 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.204 | 6.258 | 6.359 | 30.603x | 31.097x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1268.268 | 1301.051 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 157.179 | 1278.239 | 1273.041 | 8.132x | 8.099x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.849 | 10.491 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.508 | 6.883 | 8.512 | 13.553x | 16.760x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3459.807 | 2597.208 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 602.722 | 3499.728 | 2593.889 | 5.807x | 4.304x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.912 | 8.226 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.206 | 6.951 | 7.781 | 33.679x | 37.697x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3392.794 | 2595.265 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 491.615 | 3454.150 | 2621.337 | 7.026x | 5.332x |

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
