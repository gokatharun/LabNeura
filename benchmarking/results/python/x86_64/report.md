# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-07T03:09:45.785925+00:00
- Commit: `88dce7a`
- CPU: AMD EPYC 7763 64-Core Processor (x86_64)
- OS: Linux-7.0.0-1017-azure-x86_64-with-glibc2.39
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.4.1
- Actual SIMD backend selected: **AVX2**
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.789 | 1.052 | 1.890 | 1.334x | 2.395x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.600 | 1.051 | 1.887 | 1.752x | 3.146x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 475.352 | 724.801 | 747.739 | 1.525x | 1.573x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 375.014 | 724.180 | 730.314 | 1.931x | 1.947x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.199 | 1.170 | 1.662 | 5.883x | 8.354x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.201 | 1.182 | 1.682 | 5.890x | 8.378x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 280.513 | 263.981 | 297.960 | 0.941x | 1.062x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 282.817 | 266.559 | 271.886 | 0.943x | 0.961x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.599 | 1.045 | 1.889 | 1.745x | 3.153x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.602 | 1.038 | 1.884 | 1.724x | 3.129x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 475.725 | 743.977 | 718.798 | 1.564x | 1.511x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 376.886 | 770.993 | 733.889 | 2.046x | 1.947x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.206 | 1.168 | 1.676 | 5.676x | 8.145x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.215 | 1.165 | 1.688 | 5.417x | 7.850x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 285.647 | 258.895 | 261.772 | 0.906x | 0.916x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 316.708 | 282.489 | 265.114 | 0.892x | 0.837x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.586 | 1.046 | 1.844 | 1.783x | 3.146x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.591 | 1.035 | 1.873 | 1.752x | 3.170x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 502.841 | 745.055 | 708.649 | 1.482x | 1.409x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 377.915 | 727.254 | 777.322 | 1.924x | 2.057x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.202 | 1.164 | 1.631 | 5.764x | 8.076x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.210 | 1.157 | 1.634 | 5.499x | 7.768x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 280.232 | 262.849 | 309.302 | 0.938x | 1.104x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 299.661 | 268.460 | 280.871 | 0.896x | 0.937x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.609 | 1.024 | 1.855 | 1.683x | 3.048x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.618 | 1.021 | 1.853 | 1.654x | 3.001x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 549.701 | 719.897 | 753.022 | 1.310x | 1.370x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 392.928 | 730.870 | 766.484 | 1.860x | 1.951x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.211 | 1.144 | 1.646 | 5.430x | 7.816x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.207 | 1.156 | 1.640 | 5.584x | 7.925x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 299.884 | 249.499 | 259.699 | 0.832x | 0.866x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 296.322 | 261.219 | 266.406 | 0.882x | 0.899x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.593 | 8.088 | 8.382 | 13.650x | 14.147x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.630 | 8.111 | 8.371 | 12.866x | 13.279x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 3310.702 | 535.913 | 429.505 | 0.162x | 0.130x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 104.932 | 594.355 | 431.020 | 5.664x | 4.108x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.209 | 8.214 | 6.683 | 39.262x | 31.944x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.211 | 8.079 | 6.699 | 38.329x | 31.781x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 3258.136 | 483.782 | 422.900 | 0.148x | 0.130x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 73.022 | 486.926 | 454.618 | 6.668x | 6.226x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.597 | 8.332 | 8.544 | 13.953x | 14.309x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.615 | 8.331 | 8.633 | 13.557x | 14.048x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 3234.248 | 516.100 | 481.853 | 0.160x | 0.149x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 103.538 | 499.197 | 416.333 | 4.821x | 4.021x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.213 | 8.418 | 6.714 | 39.482x | 31.490x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.209 | 8.191 | 6.665 | 39.154x | 31.859x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 3184.531 | 451.889 | 410.397 | 0.142x | 0.129x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 70.477 | 464.335 | 395.308 | 6.588x | 5.609x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.591 | 8.299 | 8.299 | 14.036x | 14.036x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.609 | 8.161 | 8.317 | 13.412x | 13.666x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4810.554 | 498.534 | 426.578 | 0.104x | 0.089x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 152.718 | 510.090 | 479.732 | 3.340x | 3.141x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.207 | 8.136 | 6.620 | 39.307x | 31.982x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.208 | 8.260 | 6.600 | 39.637x | 31.672x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4759.553 | 487.092 | 402.094 | 0.102x | 0.084x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 104.030 | 496.327 | 453.546 | 4.771x | 4.360x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.609 | 9.235 | 8.604 | 15.172x | 14.134x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.636 | 9.162 | 8.272 | 14.397x | 12.998x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1932.196 | 1492.903 | 2257.158 | 0.773x | 1.168x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 757.919 | 1478.320 | 2230.610 | 1.950x | 2.943x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.210 | 9.202 | 6.491 | 43.737x | 30.849x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.214 | 9.064 | 6.515 | 42.271x | 30.382x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1876.076 | 1488.041 | 2205.065 | 0.793x | 1.175x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 704.555 | 1455.702 | 2205.609 | 2.066x | 3.131x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 1.055 | 2.014 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.643 | 1.053 | 1.880 | 1.637x | 2.923x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 6691.250 | 357.493 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 261.021 | 6676.930 | 379.185 | 25.580x | 1.453x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 1.202 | 1.688 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.208 | 1.197 | 1.683 | 5.760x | 8.100x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 6583.459 | 243.046 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 153.195 | 6690.012 | 244.414 | 43.670x | 1.595x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 1.087 | 1.890 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.647 | 1.087 | 1.997 | 1.680x | 3.086x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 6982.954 | 363.131 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 258.242 | 7005.959 | 354.324 | 27.129x | 1.372x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 1.184 | 1.685 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.211 | 1.230 | 1.681 | 5.827x | 7.965x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 6883.168 | 244.772 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 155.328 | 6882.490 | 246.890 | 44.309x | 1.589x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 1.095 | 1.876 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.631 | 1.082 | 1.866 | 1.715x | 2.957x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 6664.442 | 326.843 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 250.424 | 6800.758 | 326.688 | 27.157x | 1.305x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 1.197 | 1.634 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.217 | 1.189 | 1.634 | 5.484x | 7.536x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 6639.411 | 220.218 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 152.984 | 6581.911 | 213.607 | 43.024x | 1.396x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 1.085 | 1.863 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.654 | 1.085 | 1.867 | 1.658x | 2.854x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 6678.701 | 324.353 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 278.978 | 6685.464 | 317.408 | 23.964x | 1.138x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 1.188 | 1.639 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.212 | 1.202 | 1.638 | 5.664x | 7.720x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 6580.492 | 209.309 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 158.500 | 6613.313 | 218.279 | 41.724x | 1.377x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 8.094 | 8.456 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.626 | 8.182 | 8.639 | 13.062x | 13.793x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1115.037 | 1401.837 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 224.963 | 1188.206 | 1418.283 | 5.282x | 6.305x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 8.173 | 6.704 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.212 | 7.978 | 6.687 | 37.703x | 31.603x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1031.340 | 1303.915 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 142.442 | 1088.715 | 1327.031 | 7.643x | 9.316x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 8.068 | 8.424 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.632 | 8.127 | 8.323 | 12.863x | 13.173x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1066.890 | 1421.910 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 227.727 | 1257.943 | 1376.748 | 5.524x | 6.046x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 8.356 | 6.786 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.211 | 8.031 | 6.767 | 38.103x | 32.106x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1030.774 | 1318.665 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 139.456 | 1114.934 | 1301.916 | 7.995x | 9.336x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 8.171 | 8.165 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.637 | 7.916 | 8.355 | 12.432x | 13.121x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1198.286 | 1517.194 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 302.678 | 1252.092 | 1464.627 | 4.137x | 4.839x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 8.234 | 6.599 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.214 | 8.090 | 6.553 | 37.734x | 30.566x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1100.159 | 1350.681 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 198.711 | 1083.144 | 1392.925 | 5.451x | 7.010x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 9.300 | 8.527 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.655 | 9.169 | 8.494 | 13.998x | 12.968x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 2860.470 | 3043.380 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 759.868 | 2888.333 | 2959.534 | 3.801x | 3.895x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 9.009 | 6.752 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.214 | 9.094 | 6.539 | 42.537x | 30.586x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 2888.276 | 2850.615 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 647.528 | 2779.039 | 2865.458 | 4.292x | 4.425x |

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
