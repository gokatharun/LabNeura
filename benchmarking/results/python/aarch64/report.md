# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-07T04:15:58.135316+00:00
- Commit: `c169436`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.564 | 0.790 | 1.412 | 1.400x | 2.501x |
| FP32 | ADD | allocating | 7 | NEON | 0.552 | 0.787 | 1.378 | 1.424x | 2.495x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 477.598 | 519.675 | 290.960 | 1.088x | 0.609x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 392.277 | 488.124 | 306.429 | 1.244x | 0.781x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.212 | 0.930 | 1.249 | 4.378x | 5.879x |
| FP32 | ADD | inplace | 7 | NEON | 0.214 | 0.936 | 1.247 | 4.382x | 5.839x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 346.738 | 237.219 | 234.683 | 0.684x | 0.677x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 347.924 | 222.758 | 254.245 | 0.640x | 0.731x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.557 | 0.789 | 1.397 | 1.417x | 2.510x |
| FP32 | SUB | allocating | 7 | NEON | 0.553 | 0.790 | 1.379 | 1.427x | 2.492x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 469.943 | 503.312 | 274.072 | 1.071x | 0.583x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 404.413 | 478.729 | 252.211 | 1.184x | 0.624x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.217 | 0.919 | 1.283 | 4.240x | 5.923x |
| FP32 | SUB | inplace | 7 | NEON | 0.230 | 0.914 | 1.273 | 3.971x | 5.530x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 338.462 | 219.294 | 215.220 | 0.648x | 0.636x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 350.304 | 214.144 | 214.756 | 0.611x | 0.613x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.564 | 0.795 | 1.351 | 1.409x | 2.394x |
| FP32 | MUL | allocating | 7 | NEON | 0.559 | 0.788 | 1.374 | 1.408x | 2.456x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 470.469 | 510.749 | 270.864 | 1.086x | 0.576x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 404.329 | 538.879 | 299.324 | 1.333x | 0.740x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.218 | 0.921 | 1.204 | 4.221x | 5.519x |
| FP32 | MUL | inplace | 7 | NEON | 0.213 | 0.925 | 1.209 | 4.347x | 5.682x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 353.182 | 230.406 | 232.036 | 0.652x | 0.657x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 343.880 | 215.990 | 219.008 | 0.628x | 0.637x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.560 | 0.772 | 1.383 | 1.378x | 2.468x |
| FP32 | DIV | allocating | 7 | NEON | 0.558 | 0.774 | 1.365 | 1.386x | 2.446x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 680.047 | 698.238 | 480.982 | 1.027x | 0.707x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 470.165 | 728.125 | 476.292 | 1.549x | 1.013x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.211 | 0.910 | 1.583 | 4.311x | 7.502x |
| FP32 | DIV | inplace | 7 | NEON | 0.216 | 0.905 | 1.556 | 4.190x | 7.205x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 461.389 | 461.897 | 476.456 | 1.001x | 1.033x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 464.881 | 466.055 | 468.727 | 1.003x | 1.008x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.568 | 6.365 | 8.284 | 11.202x | 14.579x |
| INT8 | ADD | allocating | 7 | NEON | 0.579 | 6.324 | 8.976 | 10.930x | 15.514x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1017.529 | 549.265 | 593.716 | 0.540x | 0.583x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 113.407 | 553.205 | 596.796 | 4.878x | 5.262x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.218 | 6.435 | 7.781 | 29.525x | 35.700x |
| INT8 | ADD | inplace | 7 | NEON | 0.220 | 6.414 | 7.899 | 29.131x | 35.877x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 955.544 | 534.176 | 587.134 | 0.559x | 0.614x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 90.682 | 544.321 | 589.678 | 6.003x | 6.503x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.570 | 6.348 | 9.754 | 11.145x | 17.123x |
| INT8 | SUB | allocating | 7 | NEON | 0.587 | 6.343 | 10.819 | 10.799x | 18.419x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1016.362 | 549.680 | 615.344 | 0.541x | 0.605x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 113.114 | 560.462 | 593.980 | 4.955x | 5.251x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.225 | 6.384 | 7.467 | 28.378x | 33.192x |
| INT8 | SUB | inplace | 7 | NEON | 0.223 | 6.371 | 9.384 | 28.607x | 42.133x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 956.044 | 536.961 | 591.336 | 0.562x | 0.619x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 90.156 | 557.629 | 593.299 | 6.185x | 6.581x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.574 | 6.346 | 7.548 | 11.058x | 13.151x |
| INT8 | MUL | allocating | 7 | NEON | 0.580 | 6.318 | 7.989 | 10.889x | 13.769x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 968.868 | 553.652 | 590.934 | 0.571x | 0.610x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 122.090 | 561.946 | 594.465 | 4.603x | 4.869x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.222 | 6.392 | 6.263 | 28.804x | 28.221x |
| INT8 | MUL | inplace | 7 | NEON | 0.230 | 6.439 | 6.278 | 27.964x | 27.264x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 910.329 | 544.102 | 582.505 | 0.598x | 0.640x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 91.036 | 541.024 | 587.287 | 5.943x | 6.451x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.565 | 7.029 | 10.634 | 12.434x | 18.811x |
| INT8 | DIV | allocating | 7 | NEON | 0.580 | 7.061 | 10.246 | 12.174x | 17.665x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1680.428 | 2775.447 | 2061.479 | 1.652x | 1.227x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 568.590 | 2746.261 | 2054.804 | 4.830x | 3.614x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.221 | 7.080 | 8.956 | 32.067x | 40.562x |
| INT8 | DIV | inplace | 7 | NEON | 0.223 | 7.048 | 9.451 | 31.555x | 42.311x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1618.606 | 2854.445 | 2055.732 | 1.764x | 1.270x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 511.826 | 2836.989 | 2054.931 | 5.543x | 4.015x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.808 | 1.415 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.592 | 0.801 | 1.383 | 1.353x | 2.338x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1955.206 | 153.131 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1009.759 | 1958.673 | 156.285 | 1.940x | 0.155x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.935 | 1.274 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.218 | 0.936 | 1.251 | 4.293x | 5.740x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1882.802 | 146.321 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 898.978 | 1883.717 | 128.392 | 2.095x | 0.143x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.805 | 1.427 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.585 | 0.806 | 1.385 | 1.379x | 2.368x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1982.729 | 139.405 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1011.470 | 1973.147 | 152.565 | 1.951x | 0.151x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.941 | 1.293 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.224 | 0.931 | 1.293 | 4.163x | 5.787x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1900.842 | 127.844 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 895.415 | 1899.798 | 125.539 | 2.122x | 0.140x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.800 | 1.368 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.580 | 0.796 | 1.354 | 1.372x | 2.336x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1982.061 | 151.064 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1009.357 | 1969.462 | 142.692 | 1.951x | 0.141x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.933 | 1.194 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.224 | 0.932 | 1.216 | 4.168x | 5.440x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1893.360 | 134.096 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 895.952 | 1900.065 | 119.686 | 2.121x | 0.134x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.788 | 1.375 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.580 | 0.791 | 1.404 | 1.363x | 2.421x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2182.708 | 362.271 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1306.698 | 2183.000 | 361.100 | 1.671x | 0.276x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.930 | 1.549 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.227 | 0.928 | 1.613 | 4.094x | 7.121x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2105.960 | 354.013 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1191.222 | 2106.738 | 350.890 | 1.769x | 0.295x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.346 | 8.757 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.575 | 6.258 | 8.215 | 10.876x | 14.277x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1276.055 | 1400.197 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 240.283 | 1301.956 | 1412.066 | 5.418x | 5.877x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.263 | 6.818 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.228 | 6.291 | 7.224 | 27.554x | 31.639x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1263.787 | 1439.503 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 179.240 | 1223.532 | 1386.880 | 6.826x | 7.738x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.250 | 9.174 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.578 | 6.215 | 9.594 | 10.754x | 16.600x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1269.728 | 1422.611 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 239.409 | 1336.028 | 1439.424 | 5.581x | 6.012x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.264 | 8.120 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.221 | 6.301 | 8.265 | 28.492x | 37.375x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1212.147 | 1352.330 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 190.951 | 1261.767 | 1471.051 | 6.608x | 7.704x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.231 | 7.733 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.595 | 6.227 | 7.898 | 10.473x | 13.284x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1440.481 | 1408.537 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 248.466 | 1395.533 | 1336.705 | 5.617x | 5.380x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.288 | 6.765 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.225 | 6.318 | 6.675 | 28.064x | 29.651x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1357.897 | 1393.397 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 176.501 | 1333.256 | 1329.282 | 7.554x | 7.531x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.941 | 9.031 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.582 | 6.953 | 8.879 | 11.948x | 15.259x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3636.316 | 2692.759 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 614.657 | 3844.139 | 2683.022 | 6.254x | 4.365x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.945 | 7.346 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.228 | 6.991 | 7.155 | 30.729x | 31.448x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3740.865 | 2652.996 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 500.842 | 3727.199 | 2659.654 | 7.442x | 5.310x |

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
