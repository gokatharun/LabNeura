# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-20T12:04:34.179276+00:00
- Commit: `2c5ef6c`
- CPU: AMD EPYC 7763 64-Core Processor (x86_64)
- OS: Linux-7.0.0-1019-azure-x86_64-with-glibc2.43
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.5.1
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.554 | 0.995 | 1.832 | 1.797x | 3.308x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.529 | 1.007 | 1.789 | 1.903x | 3.380x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 461.732 | 737.947 | 775.515 | 1.598x | 1.680x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 360.436 | 716.151 | 732.421 | 1.987x | 2.032x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.196 | 1.141 | 1.597 | 5.812x | 8.135x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.201 | 1.157 | 1.609 | 5.756x | 8.005x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 274.965 | 274.070 | 286.499 | 0.997x | 1.042x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 316.099 | 261.605 | 277.158 | 0.828x | 0.877x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.532 | 1.001 | 1.833 | 1.883x | 3.445x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.526 | 0.996 | 1.791 | 1.894x | 3.407x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 475.165 | 741.115 | 722.448 | 1.560x | 1.520x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 383.610 | 725.506 | 735.405 | 1.891x | 1.917x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.196 | 1.135 | 1.589 | 5.796x | 8.119x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.198 | 1.145 | 1.590 | 5.785x | 8.033x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 276.349 | 270.258 | 274.704 | 0.978x | 0.994x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 310.067 | 280.042 | 285.952 | 0.903x | 0.922x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.545 | 1.006 | 1.741 | 1.847x | 3.197x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.521 | 1.002 | 1.757 | 1.923x | 3.372x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 481.860 | 294.105 | 316.798 | 0.610x | 0.657x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 409.988 | 699.019 | 763.064 | 1.705x | 1.861x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.200 | 1.124 | 1.580 | 5.632x | 7.918x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.209 | 1.131 | 1.587 | 5.405x | 7.588x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 274.569 | 268.760 | 276.207 | 0.979x | 1.006x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 311.820 | 268.188 | 263.214 | 0.860x | 0.844x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.549 | 0.986 | 1.768 | 1.796x | 3.221x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.528 | 0.986 | 1.743 | 1.867x | 3.302x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 473.756 | 702.723 | 740.216 | 1.483x | 1.562x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 417.503 | 695.851 | 724.342 | 1.667x | 1.735x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.210 | 1.114 | 1.584 | 5.309x | 7.551x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.209 | 1.130 | 1.577 | 5.408x | 7.547x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 274.730 | 277.446 | 298.097 | 1.010x | 1.085x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 318.384 | 287.430 | 272.943 | 0.903x | 0.857x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.570 | 8.131 | 8.128 | 14.273x | 14.268x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.575 | 7.724 | 8.332 | 13.430x | 14.488x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 2907.354 | 495.123 | 412.524 | 0.170x | 0.142x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 105.354 | 507.186 | 450.244 | 4.814x | 4.274x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.212 | 7.725 | 6.435 | 36.473x | 30.381x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.210 | 7.801 | 6.427 | 37.181x | 30.632x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 2850.313 | 524.787 | 413.221 | 0.184x | 0.145x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 71.609 | 507.680 | 413.812 | 7.090x | 5.779x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.559 | 7.674 | 8.144 | 13.727x | 14.567x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.579 | 7.655 | 8.216 | 13.224x | 14.192x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 2910.179 | 552.027 | 452.670 | 0.190x | 0.156x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 104.205 | 495.136 | 423.835 | 4.752x | 4.067x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.211 | 7.936 | 6.730 | 37.649x | 31.925x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.204 | 7.764 | 6.468 | 38.102x | 31.741x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 2850.559 | 491.262 | 457.205 | 0.172x | 0.160x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 72.584 | 577.745 | 472.279 | 7.960x | 6.507x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.548 | 8.082 | 8.047 | 14.746x | 14.684x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.576 | 7.819 | 8.231 | 13.563x | 14.279x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4894.527 | 508.505 | 408.209 | 0.104x | 0.083x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 148.540 | 505.615 | 414.906 | 3.404x | 2.793x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.206 | 7.901 | 6.443 | 38.433x | 31.340x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.209 | 7.970 | 6.407 | 38.175x | 30.688x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4839.944 | 459.991 | 473.532 | 0.095x | 0.098x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 102.214 | 519.843 | 406.316 | 5.086x | 3.975x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.561 | 8.813 | 8.299 | 15.719x | 14.802x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.585 | 8.842 | 8.323 | 15.121x | 14.235x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1928.979 | 1463.803 | 2196.849 | 0.759x | 1.139x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 766.982 | 1497.833 | 2218.002 | 1.953x | 2.892x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.205 | 8.708 | 6.388 | 42.525x | 31.194x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.207 | 8.782 | 6.351 | 42.429x | 30.685x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1875.912 | 1506.270 | 2214.513 | 0.803x | 1.180x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 714.360 | 1450.218 | 2219.556 | 2.030x | 3.107x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 1.054 | 1.796 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.571 | 1.055 | 1.803 | 1.846x | 3.155x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 6671.845 | 350.517 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 272.265 | 6659.919 | 359.151 | 24.461x | 1.319x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 1.203 | 1.632 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.213 | 1.186 | 1.631 | 5.564x | 7.651x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 6651.352 | 242.281 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 156.614 | 6579.419 | 243.106 | 42.010x | 1.552x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 1.045 | 1.829 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.572 | 1.037 | 1.805 | 1.813x | 3.156x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 6986.344 | 341.361 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 278.060 | 6990.436 | 359.773 | 25.140x | 1.294x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 1.189 | 1.624 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.211 | 1.184 | 1.633 | 5.605x | 7.731x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 6894.054 | 242.203 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 160.567 | 6898.247 | 247.770 | 42.962x | 1.543x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 1.061 | 1.756 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.582 | 1.041 | 1.769 | 1.788x | 3.040x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 6657.402 | 320.116 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 267.726 | 6665.439 | 318.259 | 24.896x | 1.189x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 1.185 | 1.609 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.209 | 1.171 | 1.607 | 5.594x | 7.674x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 6576.283 | 212.187 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 153.118 | 6564.655 | 223.134 | 42.873x | 1.457x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 1.038 | 1.758 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.569 | 1.034 | 1.760 | 1.816x | 3.093x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 6659.477 | 304.148 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 276.122 | 6673.574 | 312.815 | 24.169x | 1.133x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 1.169 | 1.596 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.210 | 1.170 | 1.583 | 5.580x | 7.553x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 6657.802 | 209.114 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 165.249 | 6598.323 | 216.931 | 39.929x | 1.313x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 7.656 | 8.065 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.572 | 7.657 | 8.077 | 13.385x | 14.120x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1219.055 | 1422.949 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 223.980 | 1198.811 | 1381.795 | 5.352x | 6.169x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 7.698 | 6.427 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.206 | 7.768 | 6.431 | 37.786x | 31.282x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1097.256 | 1323.881 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 141.331 | 1084.975 | 1398.555 | 7.677x | 9.896x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 7.639 | 8.557 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.569 | 7.611 | 8.620 | 13.371x | 15.143x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1088.667 | 1493.403 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 234.826 | 1185.083 | 1398.939 | 5.047x | 5.957x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 7.717 | 6.619 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.206 | 7.681 | 6.441 | 37.252x | 31.242x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1022.058 | 1307.979 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 138.220 | 1168.409 | 1557.935 | 8.453x | 11.271x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 7.675 | 7.916 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.577 | 7.848 | 7.926 | 13.600x | 13.735x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1195.082 | 1441.286 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 309.600 | 1308.756 | 1457.395 | 4.227x | 4.707x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 7.742 | 6.366 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.205 | 7.872 | 6.385 | 38.369x | 31.120x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1123.955 | 1380.066 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 196.112 | 1138.492 | 1370.724 | 5.805x | 6.990x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 8.586 | 8.159 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.574 | 8.740 | 8.088 | 15.236x | 14.098x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 2752.520 | 2932.662 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 772.748 | 2798.760 | 2942.010 | 3.622x | 3.807x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 8.585 | 6.404 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.209 | 8.742 | 7.009 | 41.792x | 33.509x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 2796.456 | 2849.903 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 645.364 | 2696.961 | 2901.287 | 4.179x | 4.496x |

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
