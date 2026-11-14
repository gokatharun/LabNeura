# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-14T11:04:16.250112+00:00
- Commit: `fe410e3`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.612 | 0.792 | 1.378 | 1.294x | 2.251x |
| FP32 | ADD | allocating | 7 | NEON | 0.599 | 0.803 | 1.369 | 1.340x | 2.284x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 512.950 | 591.701 | 341.054 | 1.154x | 0.665x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 464.308 | 599.635 | 370.697 | 1.291x | 0.798x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.221 | 0.917 | 1.263 | 4.143x | 5.706x |
| FP32 | ADD | inplace | 7 | NEON | 0.264 | 0.923 | 1.238 | 3.498x | 4.691x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 377.113 | 259.351 | 311.005 | 0.688x | 0.825x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 382.247 | 284.490 | 305.782 | 0.744x | 0.800x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.597 | 0.792 | 1.421 | 1.327x | 2.380x |
| FP32 | SUB | allocating | 7 | NEON | 0.592 | 0.791 | 1.387 | 1.336x | 2.342x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 504.490 | 597.371 | 367.391 | 1.184x | 0.728x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 428.427 | 530.556 | 327.625 | 1.238x | 0.765x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.220 | 0.931 | 1.322 | 4.236x | 6.013x |
| FP32 | SUB | inplace | 7 | NEON | 0.247 | 0.933 | 1.289 | 3.771x | 5.210x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 360.707 | 242.956 | 261.523 | 0.674x | 0.725x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 359.675 | 271.803 | 260.543 | 0.756x | 0.724x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.592 | 0.792 | 1.381 | 1.337x | 2.332x |
| FP32 | MUL | allocating | 7 | NEON | 0.585 | 0.794 | 1.367 | 1.357x | 2.336x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 502.635 | 579.488 | 321.841 | 1.153x | 0.640x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 420.099 | 560.204 | 310.609 | 1.334x | 0.739x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.223 | 0.934 | 1.244 | 4.198x | 5.589x |
| FP32 | MUL | inplace | 7 | NEON | 0.259 | 0.940 | 1.234 | 3.624x | 4.759x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 360.019 | 256.721 | 267.161 | 0.713x | 0.742x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 388.199 | 257.937 | 291.832 | 0.664x | 0.752x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.601 | 0.790 | 1.360 | 1.314x | 2.262x |
| FP32 | DIV | allocating | 7 | NEON | 0.585 | 0.798 | 1.378 | 1.365x | 2.357x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 693.300 | 727.293 | 487.617 | 1.049x | 0.703x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 503.953 | 743.474 | 484.197 | 1.475x | 0.961x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.233 | 0.934 | 1.557 | 4.014x | 6.692x |
| FP32 | DIV | inplace | 7 | NEON | 0.263 | 0.937 | 1.576 | 3.562x | 5.992x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 465.884 | 466.308 | 471.626 | 1.001x | 1.012x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 465.490 | 467.798 | 472.815 | 1.005x | 1.016x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.600 | 6.529 | 9.750 | 10.886x | 16.258x |
| INT8 | ADD | allocating | 7 | NEON | 0.651 | 6.384 | 8.561 | 9.806x | 13.149x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1017.164 | 547.513 | 592.629 | 0.538x | 0.583x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 114.374 | 554.854 | 607.508 | 4.851x | 5.312x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.248 | 6.429 | 6.978 | 25.871x | 28.085x |
| INT8 | ADD | inplace | 7 | NEON | 0.265 | 6.378 | 6.618 | 24.057x | 24.959x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 957.343 | 532.882 | 585.265 | 0.557x | 0.611x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 92.358 | 528.206 | 582.448 | 5.719x | 6.306x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.619 | 6.324 | 9.296 | 10.221x | 15.023x |
| INT8 | SUB | allocating | 7 | NEON | 0.661 | 6.331 | 9.477 | 9.571x | 14.327x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1014.128 | 554.014 | 597.231 | 0.546x | 0.589x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 112.321 | 538.802 | 592.918 | 4.797x | 5.279x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.234 | 6.375 | 7.254 | 27.196x | 30.947x |
| INT8 | SUB | inplace | 7 | NEON | 0.261 | 6.350 | 7.705 | 24.349x | 29.543x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 957.492 | 529.147 | 583.687 | 0.553x | 0.610x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 88.006 | 526.287 | 590.680 | 5.980x | 6.712x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.589 | 6.329 | 9.303 | 10.746x | 15.795x |
| INT8 | MUL | allocating | 7 | NEON | 0.674 | 6.350 | 8.147 | 9.421x | 12.086x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 970.169 | 543.323 | 584.606 | 0.560x | 0.603x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 118.919 | 544.858 | 589.714 | 4.582x | 4.959x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.231 | 6.382 | 6.338 | 27.620x | 27.432x |
| INT8 | MUL | inplace | 7 | NEON | 0.268 | 6.384 | 6.499 | 23.864x | 24.294x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 911.721 | 527.974 | 572.362 | 0.579x | 0.628x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 88.013 | 527.319 | 571.318 | 5.991x | 6.491x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.611 | 7.013 | 11.389 | 11.471x | 18.628x |
| INT8 | DIV | allocating | 7 | NEON | 0.649 | 7.002 | 9.869 | 10.784x | 15.200x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1672.781 | 2813.286 | 2046.254 | 1.682x | 1.223x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 572.196 | 2833.609 | 2059.678 | 4.952x | 3.600x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.241 | 7.036 | 9.333 | 29.163x | 38.680x |
| INT8 | DIV | inplace | 7 | NEON | 0.264 | 7.031 | 10.103 | 26.680x | 38.334x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1614.006 | 2717.038 | 2044.970 | 1.683x | 1.267x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 508.875 | 2718.473 | 2042.422 | 5.342x | 4.014x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.807 | 1.399 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.659 | 0.809 | 1.391 | 1.228x | 2.112x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1955.768 | 144.137 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1010.750 | 1974.162 | 137.869 | 1.953x | 0.136x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.942 | 1.247 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.260 | 0.944 | 1.233 | 3.637x | 4.750x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1885.743 | 127.472 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 898.952 | 1887.315 | 126.585 | 2.099x | 0.141x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.813 | 1.374 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.655 | 0.809 | 1.418 | 1.234x | 2.163x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1968.596 | 132.659 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1006.099 | 1967.819 | 130.580 | 1.956x | 0.130x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.949 | 1.285 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.259 | 0.943 | 1.335 | 3.642x | 5.152x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1895.280 | 115.117 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 904.035 | 1899.791 | 116.683 | 2.101x | 0.129x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.800 | 1.375 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.654 | 0.803 | 1.354 | 1.228x | 2.070x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1975.255 | 149.364 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1009.099 | 1978.276 | 143.461 | 1.960x | 0.142x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.945 | 1.219 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.260 | 0.951 | 1.218 | 3.662x | 4.691x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1899.667 | 112.671 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 898.579 | 1895.141 | 114.168 | 2.109x | 0.127x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.800 | 1.377 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.652 | 0.790 | 1.375 | 1.212x | 2.110x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2180.664 | 354.040 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1304.702 | 2171.947 | 357.628 | 1.665x | 0.274x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.933 | 1.515 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.264 | 0.931 | 1.545 | 3.531x | 5.860x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2101.651 | 347.818 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1194.725 | 2102.511 | 350.615 | 1.760x | 0.293x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.261 | 8.420 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.650 | 6.210 | 8.232 | 9.550x | 12.659x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1290.310 | 1381.939 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 233.049 | 1295.454 | 1424.207 | 5.559x | 6.111x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.323 | 6.705 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.268 | 6.276 | 6.637 | 23.402x | 24.747x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1231.642 | 1408.064 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 181.617 | 1229.540 | 1390.257 | 6.770x | 7.655x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.238 | 8.629 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.650 | 6.220 | 8.864 | 9.574x | 13.644x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1305.357 | 1631.749 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 236.210 | 1292.301 | 1415.038 | 5.471x | 5.991x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.274 | 7.317 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.262 | 6.215 | 7.346 | 23.742x | 28.062x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1463.175 | 1401.429 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 188.002 | 1283.294 | 1399.763 | 6.826x | 7.445x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.196 | 7.478 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.647 | 6.173 | 7.785 | 9.538x | 12.029x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1411.410 | 1351.024 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 246.114 | 1409.421 | 1356.156 | 5.727x | 5.510x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.237 | 7.020 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.282 | 6.236 | 6.476 | 22.081x | 22.929x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1675.747 | 1384.837 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 198.029 | 1340.254 | 1364.080 | 6.768x | 6.888x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.886 | 8.822 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.656 | 6.967 | 9.075 | 10.628x | 13.842x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3759.880 | 2701.494 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 635.065 | 3699.250 | 2652.325 | 5.825x | 4.176x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.887 | 7.724 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.270 | 7.045 | 7.838 | 26.081x | 29.020x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3763.505 | 2715.208 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 509.432 | 3769.767 | 3537.127 | 7.400x | 6.943x |

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
