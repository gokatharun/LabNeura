# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-14T11:43:55.794912+00:00
- Commit: `039c9d5`
- CPU: aarch64 (aarch64)
- OS: Linux-7.0.0-1017-azure-aarch64-with-glibc2.39
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.4.3
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.626 | 0.784 | 1.390 | 1.253x | 2.220x |
| FP32 | ADD | allocating | 7 | NEON | 0.622 | 0.793 | 1.391 | 1.275x | 2.235x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 484.408 | 503.699 | 384.678 | 1.040x | 0.794x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 463.185 | 497.923 | 281.212 | 1.075x | 0.607x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.243 | 0.912 | 1.247 | 3.747x | 5.122x |
| FP32 | ADD | inplace | 7 | NEON | 0.272 | 0.908 | 1.273 | 3.337x | 4.676x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 362.983 | 242.509 | 254.856 | 0.668x | 0.702x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 346.305 | 291.248 | 299.277 | 0.841x | 0.864x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.626 | 0.788 | 1.422 | 1.258x | 2.270x |
| FP32 | SUB | allocating | 7 | NEON | 0.618 | 0.785 | 1.388 | 1.269x | 2.246x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 479.000 | 555.922 | 323.500 | 1.161x | 0.675x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 434.799 | 492.077 | 262.774 | 1.132x | 0.604x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.245 | 0.918 | 1.657 | 3.740x | 6.751x |
| FP32 | SUB | inplace | 7 | NEON | 0.278 | 0.928 | 1.638 | 3.333x | 5.882x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 352.024 | 222.878 | 236.253 | 0.633x | 0.671x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 364.405 | 318.340 | 298.505 | 0.874x | 0.819x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.637 | 0.783 | 1.380 | 1.230x | 2.168x |
| FP32 | MUL | allocating | 7 | NEON | 0.614 | 0.790 | 1.402 | 1.287x | 2.283x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 486.366 | 600.601 | 357.402 | 1.235x | 0.735x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 394.468 | 502.757 | 320.066 | 1.275x | 0.811x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.250 | 0.926 | 1.205 | 3.707x | 4.823x |
| FP32 | MUL | inplace | 7 | NEON | 0.275 | 0.927 | 1.214 | 3.372x | 4.415x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 337.851 | 220.816 | 241.729 | 0.654x | 0.715x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 363.435 | 284.252 | 315.046 | 0.782x | 0.867x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.631 | 0.773 | 1.399 | 1.225x | 2.218x |
| FP32 | DIV | allocating | 7 | NEON | 0.617 | 0.769 | 1.386 | 1.247x | 2.246x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 697.916 | 750.557 | 503.609 | 1.075x | 0.722x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 481.740 | 743.184 | 565.403 | 1.543x | 1.174x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.245 | 0.912 | 1.602 | 3.721x | 6.538x |
| FP32 | DIV | inplace | 7 | NEON | 0.266 | 0.915 | 1.567 | 3.440x | 5.891x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 470.107 | 479.260 | 488.530 | 1.019x | 1.039x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 480.502 | 470.347 | 485.962 | 0.979x | 1.011x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.650 | 6.311 | 9.502 | 9.702x | 14.607x |
| INT8 | ADD | allocating | 7 | NEON | 0.699 | 6.268 | 8.870 | 8.966x | 12.688x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1013.104 | 578.222 | 705.650 | 0.571x | 0.697x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 115.262 | 574.181 | 647.598 | 4.982x | 5.618x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.249 | 6.290 | 6.872 | 25.219x | 27.553x |
| INT8 | ADD | inplace | 7 | NEON | 0.294 | 6.269 | 7.092 | 21.307x | 24.105x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 956.680 | 537.321 | 610.488 | 0.562x | 0.638x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 91.201 | 546.591 | 624.490 | 5.993x | 6.847x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.632 | 6.225 | 9.323 | 9.848x | 14.749x |
| INT8 | SUB | allocating | 7 | NEON | 0.693 | 6.226 | 9.393 | 8.979x | 13.546x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1015.871 | 605.988 | 624.750 | 0.597x | 0.615x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 114.172 | 572.529 | 658.080 | 5.015x | 5.764x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.250 | 6.307 | 8.031 | 25.220x | 32.116x |
| INT8 | SUB | inplace | 7 | NEON | 0.279 | 6.267 | 8.303 | 22.471x | 29.774x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 956.454 | 563.539 | 616.711 | 0.589x | 0.645x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 90.998 | 542.006 | 597.429 | 5.956x | 6.565x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.645 | 6.235 | 7.500 | 9.665x | 11.626x |
| INT8 | MUL | allocating | 7 | NEON | 0.693 | 6.206 | 7.683 | 8.950x | 11.080x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 972.855 | 579.455 | 613.359 | 0.596x | 0.630x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 150.234 | 552.717 | 584.626 | 3.679x | 3.891x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.248 | 6.295 | 6.181 | 25.417x | 24.957x |
| INT8 | MUL | inplace | 7 | NEON | 0.277 | 6.293 | 7.139 | 22.736x | 25.793x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 911.945 | 544.441 | 579.534 | 0.597x | 0.635x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 91.376 | 540.267 | 574.691 | 5.913x | 6.289x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.659 | 6.946 | 9.950 | 10.543x | 15.103x |
| INT8 | DIV | allocating | 7 | NEON | 0.701 | 6.934 | 10.164 | 9.886x | 14.491x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1678.219 | 2842.151 | 2192.026 | 1.694x | 1.306x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 556.718 | 2863.761 | 2057.977 | 5.144x | 3.697x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.244 | 6.987 | 7.840 | 28.618x | 32.111x |
| INT8 | DIV | inplace | 7 | NEON | 0.288 | 6.994 | 9.445 | 24.312x | 32.835x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1615.484 | 2830.211 | 2070.701 | 1.752x | 1.282x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 497.385 | 2766.735 | 2070.731 | 5.563x | 4.163x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.794 | 1.383 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.696 | 0.789 | 1.396 | 1.133x | 2.005x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1966.032 | 184.799 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1009.556 | 1948.364 | 174.424 | 1.930x | 0.173x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.935 | 1.232 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.280 | 0.958 | 1.261 | 3.421x | 4.503x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1890.516 | 140.935 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 898.274 | 1883.145 | 130.968 | 2.096x | 0.146x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.797 | 1.429 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.690 | 0.795 | 1.402 | 1.153x | 2.033x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1975.644 | 149.785 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1009.457 | 1979.003 | 173.308 | 1.960x | 0.172x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.939 | 1.601 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.279 | 0.947 | 1.576 | 3.394x | 5.645x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1901.500 | 116.634 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 896.750 | 1896.691 | 119.571 | 2.115x | 0.133x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.800 | 1.405 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.695 | 0.800 | 1.410 | 1.152x | 2.030x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1978.059 | 142.906 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1014.197 | 1991.793 | 135.920 | 1.964x | 0.134x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.937 | 1.237 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.277 | 0.951 | 1.229 | 3.432x | 4.438x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1901.810 | 139.092 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 901.964 | 1906.934 | 153.141 | 2.114x | 0.170x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.782 | 1.401 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.704 | 0.785 | 1.420 | 1.115x | 2.018x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2194.418 | 366.420 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1306.350 | 2250.499 | 360.012 | 1.723x | 0.276x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.929 | 1.574 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.276 | 0.924 | 1.514 | 3.355x | 5.497x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2101.911 | 353.083 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1193.282 | 2108.054 | 350.516 | 1.767x | 0.294x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.319 | 8.315 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.693 | 6.217 | 8.592 | 8.974x | 12.402x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1262.503 | 1341.203 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 238.961 | 1277.119 | 1596.841 | 5.344x | 6.682x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.245 | 7.251 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.282 | 6.268 | 6.646 | 22.247x | 23.590x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1207.657 | 1360.839 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 182.261 | 1347.137 | 1476.940 | 7.391x | 8.103x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.169 | 9.532 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.705 | 6.206 | 9.306 | 8.807x | 13.204x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1386.837 | 1462.539 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 245.318 | 1354.193 | 1408.524 | 5.520x | 5.742x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.243 | 9.018 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.275 | 6.266 | 7.872 | 22.783x | 28.625x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1359.127 | 1480.201 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 230.801 | 1241.830 | 1395.414 | 5.381x | 6.046x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.187 | 8.288 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.690 | 6.188 | 9.267 | 8.973x | 13.439x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1368.705 | 1383.361 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 254.552 | 1468.494 | 1440.550 | 5.769x | 5.659x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.210 | 6.593 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.274 | 6.198 | 6.669 | 22.614x | 24.333x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1377.223 | 1465.780 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 177.962 | 1399.264 | 1490.470 | 7.863x | 8.375x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.819 | 9.513 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.713 | 6.861 | 9.205 | 9.623x | 12.912x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3993.848 | 2801.324 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 615.771 | 3625.406 | 2647.761 | 5.888x | 4.300x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.912 | 7.767 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.282 | 6.920 | 7.948 | 24.505x | 28.146x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3954.332 | 2628.740 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 496.909 | 3976.507 | 2757.765 | 8.002x | 5.550x |

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
