# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-05T19:31:13.885476+00:00
- Commit: `e8dca86`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.461 | 0.782 | 1.379 | 1.697x | 2.994x |
| FP32 | ADD | allocating | 7 | NEON | 0.474 | 0.776 | 1.367 | 1.636x | 2.881x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 485.392 | 206.199 | 301.276 | 0.425x | 0.621x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 489.370 | 205.341 | 243.802 | 0.420x | 0.498x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.207 | 0.922 | 1.247 | 4.457x | 6.026x |
| FP32 | ADD | inplace | 7 | NEON | 0.201 | 0.904 | 1.235 | 4.491x | 6.137x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 361.932 | 182.624 | 271.745 | 0.505x | 0.751x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 376.725 | 199.525 | 235.726 | 0.530x | 0.626x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.461 | 0.773 | 1.408 | 1.676x | 3.054x |
| FP32 | SUB | allocating | 7 | NEON | 0.467 | 0.770 | 1.403 | 1.648x | 3.002x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 494.839 | 211.665 | 305.566 | 0.428x | 0.618x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 495.920 | 217.591 | 258.336 | 0.439x | 0.521x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.201 | 0.920 | 1.253 | 4.568x | 6.220x |
| FP32 | SUB | inplace | 7 | NEON | 0.205 | 0.912 | 1.342 | 4.439x | 6.533x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 374.412 | 181.224 | 228.050 | 0.484x | 0.609x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 369.516 | 176.312 | 215.929 | 0.477x | 0.584x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.448 | 0.778 | 1.398 | 1.739x | 3.123x |
| FP32 | MUL | allocating | 7 | NEON | 0.479 | 0.779 | 1.393 | 1.627x | 2.908x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 487.156 | 212.175 | 293.626 | 0.436x | 0.603x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 485.334 | 208.349 | 289.534 | 0.429x | 0.597x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.201 | 0.923 | 1.207 | 4.583x | 5.993x |
| FP32 | MUL | inplace | 7 | NEON | 0.201 | 0.918 | 1.200 | 4.567x | 5.969x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 360.286 | 179.638 | 210.969 | 0.499x | 0.586x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 359.180 | 181.408 | 205.607 | 0.505x | 0.572x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.454 | 0.764 | 1.370 | 1.680x | 3.015x |
| FP32 | DIV | allocating | 7 | NEON | 0.472 | 0.762 | 1.369 | 1.614x | 2.901x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 695.185 | 455.091 | 476.302 | 0.655x | 0.685x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 622.183 | 458.561 | 478.654 | 0.737x | 0.769x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.205 | 0.904 | 1.572 | 4.419x | 7.682x |
| FP32 | DIV | inplace | 7 | NEON | 0.210 | 0.903 | 1.579 | 4.291x | 7.507x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 474.618 | 454.501 | 468.738 | 0.958x | 0.988x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 403.612 | 453.613 | 468.246 | 1.124x | 1.160x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.459 | 6.339 | 8.096 | 13.823x | 17.655x |
| INT8 | ADD | allocating | 7 | NEON | 0.476 | 6.322 | 8.085 | 13.276x | 16.978x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1019.068 | 531.775 | 600.521 | 0.522x | 0.589x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 124.122 | 545.775 | 623.904 | 4.397x | 5.027x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.212 | 6.293 | 7.134 | 29.619x | 33.573x |
| INT8 | ADD | inplace | 7 | NEON | 0.218 | 6.357 | 7.729 | 29.126x | 35.415x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 955.128 | 527.596 | 585.601 | 0.552x | 0.613x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 99.074 | 538.128 | 616.310 | 5.432x | 6.221x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.463 | 6.292 | 11.726 | 13.584x | 25.313x |
| INT8 | SUB | allocating | 7 | NEON | 0.479 | 6.304 | 9.599 | 13.160x | 20.038x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1018.240 | 532.825 | 598.155 | 0.523x | 0.587x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 117.939 | 547.949 | 627.350 | 4.646x | 5.319x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.210 | 6.359 | 7.951 | 30.245x | 37.814x |
| INT8 | SUB | inplace | 7 | NEON | 0.214 | 6.347 | 7.241 | 29.624x | 33.793x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 954.774 | 528.435 | 589.396 | 0.553x | 0.617x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 92.800 | 529.342 | 622.961 | 5.704x | 6.713x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.465 | 6.329 | 7.872 | 13.620x | 16.941x |
| INT8 | MUL | allocating | 7 | NEON | 0.473 | 6.393 | 7.685 | 13.525x | 16.259x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 973.235 | 528.556 | 586.021 | 0.543x | 0.602x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 127.887 | 543.926 | 609.700 | 4.253x | 4.767x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.210 | 6.360 | 6.529 | 30.319x | 31.127x |
| INT8 | MUL | inplace | 7 | NEON | 0.213 | 6.347 | 6.255 | 29.869x | 29.437x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 910.882 | 538.366 | 591.091 | 0.591x | 0.649x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 98.082 | 528.848 | 600.834 | 5.392x | 6.126x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.464 | 6.984 | 9.446 | 15.061x | 20.372x |
| INT8 | DIV | allocating | 7 | NEON | 0.459 | 7.022 | 9.792 | 15.286x | 21.317x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1677.911 | 2745.062 | 2061.524 | 1.636x | 1.229x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 568.386 | 2812.140 | 2080.072 | 4.948x | 3.660x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.214 | 7.048 | 7.468 | 32.899x | 34.856x |
| INT8 | DIV | inplace | 7 | NEON | 0.206 | 7.044 | 8.136 | 34.231x | 39.536x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1615.306 | 2736.604 | 2046.028 | 1.694x | 1.267x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 498.305 | 2797.708 | 2065.616 | 5.614x | 4.145x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.812 | 1.408 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.478 | 0.807 | 1.370 | 1.688x | 2.866x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1961.826 | 174.875 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1009.171 | 1966.099 | 186.467 | 1.948x | 0.185x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.942 | 1.251 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.203 | 0.933 | 1.220 | 4.590x | 5.998x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1884.208 | 156.310 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 903.349 | 1883.432 | 145.966 | 2.085x | 0.162x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.793 | 1.393 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.452 | 0.794 | 1.388 | 1.758x | 3.073x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1984.969 | 204.729 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1009.404 | 1977.556 | 169.101 | 1.959x | 0.168x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.924 | 1.305 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.208 | 0.924 | 1.265 | 4.441x | 6.075x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1901.144 | 171.976 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 896.575 | 1900.036 | 168.742 | 2.119x | 0.188x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.804 | 1.363 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.463 | 0.800 | 1.365 | 1.729x | 2.949x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1976.089 | 182.256 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1012.690 | 1973.933 | 184.115 | 1.949x | 0.182x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.929 | 1.218 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.208 | 0.925 | 1.211 | 4.454x | 5.832x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1897.867 | 144.510 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 897.303 | 1904.027 | 165.614 | 2.122x | 0.185x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.782 | 1.377 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.495 | 0.787 | 1.395 | 1.588x | 2.816x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2173.379 | 359.057 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 7960.310 | 2178.328 | 371.016 | 0.274x | 0.047x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.910 | 1.535 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.229 | 0.929 | 1.538 | 4.050x | 6.709x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2099.101 | 354.903 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 7831.472 | 2107.816 | 355.720 | 0.269x | 0.045x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.257 | 8.403 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.472 | 6.249 | 8.260 | 13.231x | 17.486x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1323.102 | 1503.150 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 251.275 | 1275.334 | 1492.649 | 5.075x | 5.940x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.285 | 6.777 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.211 | 6.289 | 6.950 | 29.776x | 32.904x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1277.442 | 1486.908 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 190.443 | 1245.048 | 1474.556 | 6.538x | 7.743x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.187 | 9.646 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.468 | 6.200 | 10.303 | 13.251x | 22.022x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1301.748 | 1479.185 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 253.510 | 1313.798 | 1510.544 | 5.182x | 5.959x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.237 | 8.030 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.206 | 6.383 | 7.793 | 30.994x | 37.841x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1223.978 | 1423.816 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 192.778 | 1297.030 | 1482.970 | 6.728x | 7.693x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.245 | 8.052 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.463 | 6.239 | 8.561 | 13.474x | 18.488x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1398.356 | 1405.961 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 257.102 | 1384.053 | 1452.379 | 5.383x | 5.649x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.248 | 6.178 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.212 | 6.268 | 6.658 | 29.584x | 31.428x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1336.751 | 1367.494 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 186.773 | 1370.540 | 1400.875 | 7.338x | 7.500x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.997 | 8.877 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.459 | 6.908 | 9.859 | 15.058x | 21.492x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3735.030 | 2672.174 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 614.043 | 3908.017 | 2703.557 | 6.364x | 4.403x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.953 | 7.592 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.209 | 6.934 | 7.591 | 33.110x | 36.246x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3805.887 | 2683.447 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 498.823 | 3763.274 | 2690.474 | 7.544x | 5.394x |

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
