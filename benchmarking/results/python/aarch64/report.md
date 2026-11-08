# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-08T06:54:14.305881+00:00
- Commit: `1089815`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.639 | 0.787 | 1.381 | 1.231x | 2.160x |
| FP32 | ADD | allocating | 7 | NEON | 0.619 | 0.791 | 1.402 | 1.278x | 2.266x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 460.094 | 493.582 | 257.550 | 1.073x | 0.560x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 392.175 | 544.675 | 306.928 | 1.389x | 0.783x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.242 | 0.923 | 1.232 | 3.809x | 5.084x |
| FP32 | ADD | inplace | 7 | NEON | 0.270 | 0.925 | 1.250 | 3.430x | 4.635x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 343.073 | 220.239 | 237.734 | 0.642x | 0.693x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 361.776 | 218.907 | 242.848 | 0.605x | 0.671x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.629 | 0.785 | 1.407 | 1.249x | 2.236x |
| FP32 | SUB | allocating | 7 | NEON | 0.629 | 0.783 | 1.440 | 1.245x | 2.290x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 467.271 | 501.508 | 255.494 | 1.073x | 0.547x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 407.599 | 527.755 | 282.525 | 1.295x | 0.693x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.238 | 0.935 | 1.354 | 3.929x | 5.686x |
| FP32 | SUB | inplace | 7 | NEON | 0.274 | 0.936 | 1.277 | 3.412x | 4.653x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 341.647 | 223.017 | 242.317 | 0.653x | 0.709x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 343.467 | 219.377 | 247.368 | 0.639x | 0.720x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.624 | 0.792 | 1.410 | 1.269x | 2.261x |
| FP32 | MUL | allocating | 7 | NEON | 0.628 | 0.792 | 1.372 | 1.262x | 2.185x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 467.801 | 516.246 | 303.251 | 1.104x | 0.648x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 399.041 | 519.349 | 296.269 | 1.301x | 0.742x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.246 | 0.942 | 1.235 | 3.824x | 5.016x |
| FP32 | MUL | inplace | 7 | NEON | 0.270 | 0.947 | 1.254 | 3.507x | 4.642x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 345.713 | 278.319 | 271.913 | 0.805x | 0.787x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 344.715 | 217.349 | 225.795 | 0.631x | 0.655x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.620 | 0.771 | 1.386 | 1.243x | 2.234x |
| FP32 | DIV | allocating | 7 | NEON | 0.611 | 0.773 | 1.388 | 1.264x | 2.271x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 680.836 | 746.617 | 492.260 | 1.097x | 0.723x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 473.317 | 772.859 | 489.156 | 1.633x | 1.033x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.240 | 0.920 | 1.543 | 3.829x | 6.425x |
| FP32 | DIV | inplace | 7 | NEON | 0.277 | 0.923 | 1.544 | 3.331x | 5.573x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 463.241 | 468.425 | 478.321 | 1.011x | 1.033x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 465.369 | 466.273 | 475.889 | 1.002x | 1.023x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.637 | 6.405 | 9.226 | 10.060x | 14.492x |
| INT8 | ADD | allocating | 7 | NEON | 0.704 | 6.329 | 8.873 | 8.996x | 12.612x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1016.536 | 555.741 | 598.907 | 0.547x | 0.589x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 113.542 | 548.689 | 596.610 | 4.832x | 5.255x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.254 | 6.356 | 6.637 | 25.046x | 26.155x |
| INT8 | ADD | inplace | 7 | NEON | 0.286 | 6.348 | 6.950 | 22.214x | 24.319x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 958.150 | 531.866 | 605.881 | 0.555x | 0.632x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 89.608 | 532.615 | 596.125 | 5.944x | 6.653x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.639 | 6.401 | 9.807 | 10.016x | 15.346x |
| INT8 | SUB | allocating | 7 | NEON | 0.707 | 6.274 | 8.510 | 8.872x | 12.034x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1013.625 | 542.941 | 601.000 | 0.536x | 0.593x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 113.268 | 549.722 | 628.860 | 4.853x | 5.552x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.253 | 6.303 | 7.640 | 24.949x | 30.240x |
| INT8 | SUB | inplace | 7 | NEON | 0.279 | 6.300 | 7.376 | 22.565x | 26.419x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 957.697 | 528.899 | 599.417 | 0.552x | 0.626x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 90.529 | 546.937 | 602.099 | 6.042x | 6.651x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.641 | 6.258 | 7.762 | 9.761x | 12.106x |
| INT8 | MUL | allocating | 7 | NEON | 0.698 | 6.281 | 8.396 | 8.997x | 12.028x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 972.408 | 557.057 | 598.389 | 0.573x | 0.615x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 123.816 | 568.204 | 601.231 | 4.589x | 4.856x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.250 | 6.319 | 6.692 | 25.251x | 26.739x |
| INT8 | MUL | inplace | 7 | NEON | 0.280 | 6.371 | 7.175 | 22.714x | 25.580x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 912.189 | 555.099 | 604.490 | 0.609x | 0.663x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 96.257 | 562.138 | 594.489 | 5.840x | 6.176x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.647 | 6.941 | 8.761 | 10.733x | 13.546x |
| INT8 | DIV | allocating | 7 | NEON | 0.701 | 6.941 | 8.748 | 9.900x | 12.477x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1675.856 | 2758.810 | 2066.793 | 1.646x | 1.233x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 570.681 | 2816.706 | 2071.150 | 4.936x | 3.629x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.255 | 6.943 | 8.185 | 27.220x | 32.091x |
| INT8 | DIV | inplace | 7 | NEON | 0.283 | 6.947 | 9.453 | 24.572x | 33.436x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1615.322 | 2761.505 | 2055.701 | 1.710x | 1.273x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 513.946 | 2775.903 | 2056.892 | 5.401x | 4.002x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.807 | 1.377 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.713 | 0.805 | 1.386 | 1.130x | 1.945x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1959.569 | 163.716 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1007.787 | 1951.925 | 143.656 | 1.937x | 0.143x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.945 | 1.261 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.275 | 0.941 | 1.239 | 3.424x | 4.507x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1888.897 | 134.683 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 899.088 | 1889.964 | 118.563 | 2.102x | 0.132x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.808 | 1.391 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.708 | 0.809 | 1.412 | 1.143x | 1.996x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1978.937 | 140.090 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1006.608 | 1976.071 | 130.074 | 1.963x | 0.129x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.949 | 1.292 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.281 | 0.948 | 1.332 | 3.370x | 4.737x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1900.610 | 117.879 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 898.681 | 1903.398 | 134.146 | 2.118x | 0.149x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.807 | 1.366 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.699 | 0.817 | 1.400 | 1.168x | 2.002x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1991.920 | 135.733 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1008.187 | 1984.584 | 144.089 | 1.968x | 0.143x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.945 | 1.225 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.274 | 0.943 | 1.221 | 3.442x | 4.454x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1899.621 | 154.321 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 899.453 | 1901.371 | 131.106 | 2.114x | 0.146x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.800 | 1.392 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.701 | 0.802 | 1.391 | 1.143x | 1.983x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2193.587 | 362.239 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1306.537 | 2177.539 | 361.158 | 1.667x | 0.276x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.937 | 1.548 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.280 | 0.941 | 1.598 | 3.355x | 5.697x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2103.893 | 352.775 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1192.712 | 2109.047 | 357.324 | 1.768x | 0.300x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.269 | 9.100 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.723 | 6.252 | 8.040 | 8.642x | 11.114x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1279.530 | 1469.476 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 244.900 | 1244.708 | 1410.852 | 5.083x | 5.761x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.322 | 6.946 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.287 | 6.261 | 6.867 | 21.838x | 23.949x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1259.346 | 1441.958 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 181.875 | 1226.231 | 1419.130 | 6.742x | 7.803x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.222 | 8.697 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.705 | 6.238 | 9.454 | 8.850x | 13.413x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1245.663 | 1362.030 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 236.575 | 1282.685 | 1411.509 | 5.422x | 5.966x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.261 | 7.843 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.280 | 6.290 | 7.193 | 22.440x | 25.660x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1236.454 | 1415.689 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 180.589 | 1278.296 | 1390.904 | 7.078x | 7.702x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.254 | 7.955 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.697 | 6.236 | 7.492 | 8.951x | 10.755x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1367.892 | 1364.012 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 252.746 | 1383.784 | 1399.901 | 5.475x | 5.539x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.255 | 6.263 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.282 | 6.234 | 6.952 | 22.088x | 24.632x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1351.355 | 1361.335 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 181.195 | 1387.943 | 1393.005 | 7.660x | 7.688x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.875 | 10.877 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.717 | 6.852 | 10.226 | 9.557x | 14.262x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3780.997 | 2673.580 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 611.706 | 3702.797 | 2659.290 | 6.053x | 4.347x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.879 | 9.828 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.289 | 6.986 | 8.618 | 24.133x | 29.772x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3749.096 | 2658.418 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 496.754 | 3978.587 | 2658.428 | 8.009x | 5.352x |

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
