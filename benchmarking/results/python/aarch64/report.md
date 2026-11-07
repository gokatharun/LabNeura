# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-07T13:52:47.653920+00:00
- Commit: `ded92fd`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.580 | 0.793 | 1.451 | 1.366x | 2.501x |
| FP32 | ADD | allocating | 7 | NEON | 0.581 | 0.795 | 1.391 | 1.367x | 2.392x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 474.375 | 534.423 | 310.008 | 1.127x | 0.654x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 406.816 | 526.254 | 309.952 | 1.294x | 0.762x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.222 | 0.924 | 1.258 | 4.156x | 5.659x |
| FP32 | ADD | inplace | 7 | NEON | 0.251 | 0.918 | 1.239 | 3.651x | 4.928x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 352.385 | 264.265 | 278.778 | 0.750x | 0.791x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 369.894 | 303.772 | 286.170 | 0.821x | 0.774x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.589 | 0.790 | 1.401 | 1.342x | 2.381x |
| FP32 | SUB | allocating | 7 | NEON | 0.583 | 0.786 | 1.393 | 1.348x | 2.390x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 505.395 | 593.707 | 333.647 | 1.175x | 0.660x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 440.001 | 558.629 | 357.156 | 1.270x | 0.812x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.221 | 0.929 | 1.274 | 4.209x | 5.775x |
| FP32 | SUB | inplace | 7 | NEON | 0.252 | 0.925 | 1.330 | 3.672x | 5.276x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 363.882 | 290.830 | 332.001 | 0.799x | 0.912x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 378.602 | 262.993 | 302.823 | 0.695x | 0.800x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.593 | 0.816 | 1.363 | 1.376x | 2.298x |
| FP32 | MUL | allocating | 7 | NEON | 0.583 | 0.815 | 1.384 | 1.398x | 2.373x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 480.764 | 586.544 | 340.578 | 1.220x | 0.708x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 408.898 | 572.867 | 325.655 | 1.401x | 0.796x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.222 | 0.956 | 1.224 | 4.299x | 5.502x |
| FP32 | MUL | inplace | 7 | NEON | 0.253 | 0.959 | 1.206 | 3.794x | 4.771x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 363.964 | 276.289 | 264.059 | 0.759x | 0.726x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 371.160 | 268.579 | 271.173 | 0.724x | 0.731x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.576 | 0.776 | 1.377 | 1.347x | 2.390x |
| FP32 | DIV | allocating | 7 | NEON | 0.577 | 0.781 | 1.373 | 1.353x | 2.378x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 694.266 | 728.269 | 489.025 | 1.049x | 0.704x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 489.614 | 736.750 | 497.147 | 1.505x | 1.015x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.231 | 0.920 | 1.574 | 3.989x | 6.821x |
| FP32 | DIV | inplace | 7 | NEON | 0.255 | 0.914 | 1.633 | 3.582x | 6.398x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 467.739 | 466.441 | 478.640 | 0.997x | 1.023x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 473.569 | 478.113 | 526.903 | 1.010x | 1.113x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.614 | 6.388 | 8.517 | 10.400x | 13.865x |
| INT8 | ADD | allocating | 7 | NEON | 0.645 | 6.330 | 8.228 | 9.822x | 12.766x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1015.974 | 567.669 | 633.517 | 0.559x | 0.624x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 116.857 | 570.750 | 623.108 | 4.884x | 5.332x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.238 | 6.422 | 6.812 | 27.026x | 28.666x |
| INT8 | ADD | inplace | 7 | NEON | 0.266 | 6.383 | 7.158 | 24.029x | 26.948x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 956.177 | 548.788 | 603.931 | 0.574x | 0.632x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 93.187 | 560.748 | 678.404 | 6.017x | 7.280x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.588 | 6.291 | 9.723 | 10.695x | 16.530x |
| INT8 | SUB | allocating | 7 | NEON | 0.640 | 6.309 | 8.713 | 9.862x | 13.620x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1019.198 | 558.715 | 614.480 | 0.548x | 0.603x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 118.388 | 561.530 | 620.202 | 4.743x | 5.239x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.225 | 6.323 | 7.800 | 28.086x | 34.646x |
| INT8 | SUB | inplace | 7 | NEON | 0.264 | 6.292 | 7.972 | 23.788x | 30.142x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 957.544 | 553.065 | 603.909 | 0.578x | 0.631x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 92.665 | 546.358 | 610.831 | 5.896x | 6.592x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.595 | 6.304 | 9.154 | 10.597x | 15.386x |
| INT8 | MUL | allocating | 7 | NEON | 0.636 | 6.310 | 7.976 | 9.926x | 12.546x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 970.009 | 581.211 | 600.485 | 0.599x | 0.619x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 128.273 | 574.680 | 607.315 | 4.480x | 4.735x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.229 | 6.349 | 6.335 | 27.729x | 27.670x |
| INT8 | MUL | inplace | 7 | NEON | 0.260 | 6.356 | 6.786 | 24.428x | 26.081x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 911.030 | 553.324 | 593.779 | 0.607x | 0.652x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 98.879 | 573.900 | 611.112 | 5.804x | 6.180x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.603 | 6.989 | 10.616 | 11.589x | 17.604x |
| INT8 | DIV | allocating | 7 | NEON | 0.651 | 7.011 | 9.977 | 10.766x | 15.320x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1682.637 | 2861.126 | 2078.813 | 1.700x | 1.235x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 575.118 | 2842.313 | 2068.553 | 4.942x | 3.597x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.232 | 7.049 | 8.324 | 30.383x | 35.878x |
| INT8 | DIV | inplace | 7 | NEON | 0.264 | 7.013 | 7.536 | 26.561x | 28.543x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1620.169 | 2854.153 | 2065.059 | 1.762x | 1.275x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 513.690 | 2942.456 | 2077.681 | 5.728x | 4.045x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.812 | 1.422 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.655 | 0.815 | 1.366 | 1.245x | 2.085x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1966.203 | 184.560 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1011.882 | 1963.804 | 161.546 | 1.941x | 0.160x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.945 | 1.224 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.260 | 0.952 | 1.258 | 3.659x | 4.836x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1888.838 | 150.235 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 897.060 | 1883.168 | 134.128 | 2.099x | 0.150x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.806 | 1.397 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.648 | 0.807 | 1.377 | 1.246x | 2.125x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1995.015 | 154.318 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1011.045 | 1978.377 | 161.100 | 1.957x | 0.159x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.960 | 1.288 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.261 | 0.949 | 1.328 | 3.636x | 5.089x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1904.486 | 124.629 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 900.444 | 1905.591 | 144.102 | 2.116x | 0.160x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.802 | 1.346 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.650 | 0.810 | 1.347 | 1.246x | 2.072x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1977.187 | 151.135 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1015.747 | 1983.019 | 150.048 | 1.952x | 0.148x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.940 | 1.209 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.255 | 0.939 | 1.211 | 3.688x | 4.755x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1903.618 | 132.703 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 896.376 | 1898.786 | 135.764 | 2.118x | 0.151x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.789 | 1.358 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.656 | 0.790 | 1.365 | 1.205x | 2.082x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2203.597 | 359.015 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1309.455 | 2178.563 | 359.111 | 1.664x | 0.274x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.934 | 1.527 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.256 | 0.941 | 1.545 | 3.683x | 6.047x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2106.805 | 355.366 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1194.877 | 2102.235 | 352.132 | 1.759x | 0.295x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.240 | 8.881 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.639 | 6.246 | 9.355 | 9.772x | 14.635x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1304.064 | 1447.656 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 242.635 | 1321.961 | 1429.154 | 5.448x | 5.890x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.314 | 6.540 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.260 | 6.255 | 7.317 | 24.088x | 28.177x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1279.547 | 1447.646 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 183.091 | 1266.126 | 1406.801 | 6.915x | 7.684x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.227 | 9.180 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.643 | 6.226 | 9.276 | 9.687x | 14.431x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1316.011 | 1415.474 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 246.326 | 1356.734 | 1438.288 | 5.508x | 5.839x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.249 | 7.532 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.260 | 6.271 | 7.249 | 24.148x | 27.914x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1298.458 | 1433.615 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 186.986 | 1265.472 | 1477.054 | 6.768x | 7.899x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.212 | 7.746 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.648 | 6.228 | 8.021 | 9.603x | 12.369x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1450.043 | 1400.638 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 261.065 | 1420.329 | 1399.777 | 5.441x | 5.362x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.254 | 6.443 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.257 | 6.260 | 6.269 | 24.316x | 24.350x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1370.761 | 1373.701 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 183.306 | 1400.306 | 1366.367 | 7.639x | 7.454x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.911 | 8.861 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.649 | 7.040 | 8.661 | 10.843x | 13.339x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3959.869 | 2659.955 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 615.442 | 3916.544 | 2698.101 | 6.364x | 4.384x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.986 | 7.818 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.257 | 6.930 | 7.644 | 26.919x | 29.691x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3716.585 | 2686.990 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 496.319 | 3768.842 | 2702.212 | 7.594x | 5.445x |

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
