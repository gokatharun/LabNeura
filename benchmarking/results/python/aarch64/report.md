# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-20T06:45:44.891093+00:00
- Commit: `f78502b`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.601 | 0.798 | 1.371 | 1.327x | 2.279x |
| FP32 | ADD | allocating | 7 | NEON | 0.576 | 0.798 | 1.397 | 1.385x | 2.426x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 507.546 | 621.554 | 390.540 | 1.225x | 0.769x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 401.110 | 734.164 | 394.228 | 1.830x | 0.983x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.226 | 0.926 | 1.254 | 4.101x | 5.555x |
| FP32 | ADD | inplace | 7 | NEON | 0.267 | 0.930 | 1.263 | 3.486x | 4.734x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 342.304 | 279.264 | 277.820 | 0.816x | 0.812x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 355.393 | 302.711 | 353.898 | 0.852x | 0.996x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.607 | 0.794 | 1.412 | 1.307x | 2.325x |
| FP32 | SUB | allocating | 7 | NEON | 0.572 | 0.804 | 1.391 | 1.407x | 2.434x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 497.525 | 695.605 | 439.144 | 1.398x | 0.883x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 469.159 | 699.286 | 387.212 | 1.491x | 0.825x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.223 | 0.937 | 1.393 | 4.206x | 6.248x |
| FP32 | SUB | inplace | 7 | NEON | 0.256 | 0.929 | 1.352 | 3.626x | 5.273x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 348.957 | 332.088 | 362.857 | 0.952x | 1.040x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 369.861 | 337.136 | 345.379 | 0.912x | 0.934x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.602 | 0.836 | 1.377 | 1.388x | 2.288x |
| FP32 | MUL | allocating | 7 | NEON | 0.586 | 0.833 | 1.394 | 1.421x | 2.378x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 474.061 | 682.181 | 392.500 | 1.439x | 0.828x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 393.561 | 645.014 | 379.684 | 1.639x | 0.965x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.220 | 0.974 | 1.249 | 4.426x | 5.675x |
| FP32 | MUL | inplace | 7 | NEON | 0.252 | 0.969 | 1.225 | 3.837x | 4.851x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 367.703 | 555.190 | 310.257 | 1.510x | 0.844x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 350.209 | 459.577 | 329.782 | 1.312x | 0.942x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.602 | 0.797 | 1.470 | 1.323x | 2.442x |
| FP32 | DIV | allocating | 7 | NEON | 0.581 | 0.799 | 1.382 | 1.376x | 2.381x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 710.788 | 800.647 | 520.917 | 1.126x | 0.733x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 490.026 | 845.524 | 510.213 | 1.725x | 1.041x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.217 | 0.922 | 1.599 | 4.242x | 7.353x |
| FP32 | DIV | inplace | 7 | NEON | 0.248 | 0.934 | 1.529 | 3.767x | 6.169x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 485.342 | 506.520 | 489.818 | 1.044x | 1.009x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 481.728 | 474.173 | 498.734 | 0.984x | 1.035x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.615 | 6.445 | 8.227 | 10.475x | 13.373x |
| INT8 | ADD | allocating | 7 | NEON | 0.675 | 6.379 | 8.771 | 9.452x | 12.995x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1025.513 | 722.745 | 653.665 | 0.705x | 0.637x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 115.418 | 595.337 | 631.776 | 5.158x | 5.474x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.243 | 6.494 | 7.244 | 26.718x | 29.806x |
| INT8 | ADD | inplace | 7 | NEON | 0.277 | 6.627 | 6.738 | 23.940x | 24.340x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 958.118 | 582.069 | 661.535 | 0.608x | 0.690x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 101.246 | 595.438 | 639.215 | 5.881x | 6.314x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.619 | 6.329 | 10.871 | 10.231x | 17.574x |
| INT8 | SUB | allocating | 7 | NEON | 0.645 | 6.356 | 10.205 | 9.850x | 15.814x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1022.170 | 686.108 | 784.296 | 0.671x | 0.767x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 114.494 | 602.450 | 654.423 | 5.262x | 5.716x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.236 | 6.426 | 8.472 | 27.227x | 35.894x |
| INT8 | SUB | inplace | 7 | NEON | 0.266 | 6.460 | 8.653 | 24.308x | 32.559x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 957.429 | 636.241 | 623.485 | 0.665x | 0.651x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 89.215 | 565.721 | 632.457 | 6.341x | 7.089x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.606 | 6.371 | 8.107 | 10.508x | 13.371x |
| INT8 | MUL | allocating | 7 | NEON | 0.658 | 6.432 | 8.240 | 9.772x | 12.518x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 972.438 | 628.022 | 631.168 | 0.646x | 0.649x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 130.330 | 616.506 | 624.076 | 4.730x | 4.788x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.230 | 6.405 | 6.710 | 27.878x | 29.202x |
| INT8 | MUL | inplace | 7 | NEON | 0.267 | 6.451 | 6.711 | 24.154x | 25.129x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 913.329 | 605.106 | 655.315 | 0.663x | 0.718x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 90.029 | 588.113 | 623.787 | 6.532x | 6.929x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.603 | 7.092 | 9.632 | 11.760x | 15.972x |
| INT8 | DIV | allocating | 7 | NEON | 0.654 | 7.090 | 9.309 | 10.847x | 14.242x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1687.649 | 3231.530 | 2134.795 | 1.915x | 1.265x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 555.533 | 3056.062 | 2184.896 | 5.501x | 3.933x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.245 | 7.108 | 7.745 | 28.978x | 31.577x |
| INT8 | DIV | inplace | 7 | NEON | 0.265 | 7.141 | 9.592 | 26.965x | 36.220x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1620.900 | 3178.317 | 2108.464 | 1.961x | 1.301x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 500.991 | 3226.734 | 2111.244 | 6.441x | 4.214x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.817 | 1.419 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.658 | 0.814 | 1.385 | 1.238x | 2.105x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1978.241 | 173.898 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1015.167 | 1968.335 | 166.650 | 1.939x | 0.164x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.945 | 1.275 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.255 | 0.954 | 1.236 | 3.749x | 4.855x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1903.899 | 227.403 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 905.026 | 1903.719 | 174.419 | 2.103x | 0.193x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.819 | 1.426 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.646 | 0.811 | 1.380 | 1.255x | 2.136x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 2020.310 | 218.530 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1012.381 | 1988.928 | 212.755 | 1.965x | 0.210x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.957 | 1.319 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.261 | 0.951 | 1.347 | 3.638x | 5.151x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1904.836 | 166.916 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 907.171 | 1911.423 | 177.376 | 2.107x | 0.196x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.813 | 1.377 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.645 | 0.816 | 1.414 | 1.265x | 2.190x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1996.548 | 202.031 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1019.117 | 2003.047 | 205.090 | 1.965x | 0.201x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.949 | 1.200 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.256 | 0.953 | 1.237 | 3.726x | 4.838x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1911.301 | 192.774 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 905.482 | 1913.550 | 195.435 | 2.113x | 0.216x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.803 | 1.394 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.655 | 0.810 | 1.371 | 1.236x | 2.092x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2201.847 | 371.074 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1307.711 | 2208.570 | 371.341 | 1.689x | 0.284x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.935 | 1.491 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.254 | 0.934 | 1.480 | 3.677x | 5.829x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2112.137 | 369.072 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1199.337 | 2116.211 | 356.265 | 1.764x | 0.297x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.310 | 9.462 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.653 | 6.310 | 10.469 | 9.659x | 16.025x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1439.131 | 1618.338 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 249.891 | 1516.414 | 1831.162 | 6.068x | 7.328x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.348 | 7.041 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.279 | 6.367 | 7.105 | 22.802x | 25.447x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1467.284 | 2301.417 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 206.375 | 1484.973 | 1575.753 | 7.196x | 7.635x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.275 | 9.207 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.655 | 6.288 | 8.873 | 9.599x | 13.546x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1516.406 | 1742.357 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 256.304 | 1455.946 | 1789.110 | 5.681x | 6.980x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.310 | 9.195 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.274 | 6.317 | 7.254 | 23.021x | 26.435x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1429.780 | 1648.044 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 197.429 | 1469.984 | 1798.939 | 7.446x | 9.112x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.343 | 8.768 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.658 | 6.300 | 8.306 | 9.568x | 12.615x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1551.480 | 1722.164 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 274.663 | 1941.574 | 1676.493 | 7.069x | 6.104x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.334 | 6.333 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.262 | 6.301 | 6.752 | 24.042x | 25.764x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1546.565 | 1723.588 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 219.261 | 1640.783 | 1593.248 | 7.483x | 7.266x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.936 | 9.482 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.679 | 6.934 | 9.075 | 10.207x | 13.358x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 4513.452 | 3056.871 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 618.952 | 4651.967 | 2924.142 | 7.516x | 4.724x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.987 | 7.548 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.262 | 6.995 | 8.644 | 26.674x | 32.961x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 4391.862 | 2981.519 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 500.100 | 4228.222 | 2827.263 | 8.455x | 5.653x |

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
