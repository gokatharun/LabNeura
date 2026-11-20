# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-20T16:26:28.859625+00:00
- Commit: `e807bf9`
- CPU: AMD EPYC 7763 64-Core Processor (x86_64)
- OS: Linux-7.0.0-1019-azure-x86_64-with-glibc2.43
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.5.2
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.568 | 1.038 | 1.846 | 1.828x | 3.252x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.537 | 1.027 | 1.841 | 1.912x | 3.425x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 509.258 | 755.077 | 782.273 | 1.483x | 1.536x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 396.515 | 781.704 | 780.750 | 1.971x | 1.969x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.201 | 1.204 | 1.695 | 5.978x | 8.415x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.204 | 1.185 | 1.708 | 5.804x | 8.364x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 317.583 | 318.061 | 334.477 | 1.002x | 1.053x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 356.285 | 326.302 | 337.312 | 0.916x | 0.947x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.550 | 1.036 | 1.848 | 1.885x | 3.362x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.531 | 1.052 | 1.854 | 1.980x | 3.491x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 528.467 | 753.797 | 782.782 | 1.426x | 1.481x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 368.430 | 793.867 | 792.933 | 2.155x | 2.152x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.200 | 1.193 | 1.699 | 5.979x | 8.513x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.211 | 1.164 | 1.704 | 5.517x | 8.078x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 299.196 | 307.883 | 340.696 | 1.029x | 1.139x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 373.069 | 330.507 | 357.776 | 0.886x | 0.959x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.567 | 1.018 | 1.828 | 1.796x | 3.224x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.532 | 1.028 | 1.821 | 1.933x | 3.423x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 453.892 | 794.271 | 779.540 | 1.750x | 1.717x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 384.184 | 835.960 | 791.574 | 2.176x | 2.060x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.203 | 1.158 | 1.660 | 5.701x | 8.172x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.215 | 1.158 | 1.635 | 5.394x | 7.619x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 321.433 | 313.888 | 426.406 | 0.977x | 1.327x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 340.485 | 326.747 | 340.223 | 0.960x | 0.999x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.546 | 1.010 | 1.807 | 1.849x | 3.308x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.533 | 1.014 | 1.840 | 1.902x | 3.451x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 497.503 | 838.256 | 805.100 | 1.685x | 1.618x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 385.319 | 1266.595 | 814.266 | 3.287x | 2.113x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.204 | 1.147 | 1.642 | 5.612x | 8.034x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.212 | 1.163 | 1.632 | 5.493x | 7.707x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 318.657 | 322.141 | 344.218 | 1.011x | 1.080x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 372.626 | 329.431 | 314.170 | 0.884x | 0.843x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.591 | 8.163 | 8.654 | 13.815x | 14.645x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.574 | 7.921 | 8.579 | 13.787x | 14.933x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 2914.981 | 559.629 | 481.338 | 0.192x | 0.165x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 115.792 | 507.866 | 474.989 | 4.386x | 4.102x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.211 | 7.970 | 6.780 | 37.778x | 32.134x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.214 | 8.196 | 6.762 | 38.227x | 31.539x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 2846.714 | 518.058 | 452.722 | 0.182x | 0.159x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 81.767 | 483.519 | 437.088 | 5.913x | 5.346x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.576 | 7.987 | 8.438 | 13.855x | 14.637x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.576 | 7.891 | 8.356 | 13.693x | 14.500x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 2916.180 | 578.951 | 464.363 | 0.199x | 0.159x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 119.369 | 521.980 | 466.709 | 4.373x | 3.910x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.221 | 8.152 | 6.892 | 36.954x | 31.243x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.212 | 8.192 | 6.728 | 38.567x | 31.678x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 2850.804 | 499.308 | 455.310 | 0.175x | 0.160x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 81.823 | 502.781 | 438.979 | 6.145x | 5.365x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.569 | 8.095 | 8.532 | 14.226x | 14.994x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.587 | 7.820 | 8.408 | 13.329x | 14.330x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4903.819 | 512.903 | 408.498 | 0.105x | 0.083x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 152.841 | 493.941 | 461.668 | 3.232x | 3.021x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.209 | 7.901 | 6.587 | 37.806x | 31.522x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.210 | 8.010 | 6.745 | 38.146x | 32.123x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4833.705 | 513.000 | 435.658 | 0.106x | 0.090x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 104.251 | 480.791 | 398.978 | 4.612x | 3.827x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.561 | 9.120 | 8.269 | 16.245x | 14.729x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.591 | 9.013 | 8.358 | 15.243x | 14.136x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1938.701 | 1564.905 | 2220.637 | 0.807x | 1.145x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 775.543 | 1565.193 | 2231.125 | 2.018x | 2.877x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.215 | 9.271 | 6.618 | 43.040x | 30.724x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.209 | 9.274 | 6.548 | 44.291x | 31.271x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1876.839 | 1598.280 | 2213.205 | 0.852x | 1.179x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 714.149 | 1484.846 | 2206.726 | 2.079x | 3.090x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 1.099 | 1.809 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.589 | 1.096 | 1.815 | 1.860x | 3.079x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 6656.981 | 347.763 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 241.420 | 6676.685 | 349.588 | 27.656x | 1.448x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 1.228 | 1.716 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.221 | 1.202 | 1.701 | 5.440x | 7.694x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 6574.639 | 248.603 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 172.297 | 6586.540 | 251.300 | 38.228x | 1.459x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 1.107 | 1.834 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.593 | 1.100 | 1.808 | 1.854x | 3.048x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 6993.552 | 353.257 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 240.350 | 6976.990 | 355.714 | 29.028x | 1.480x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 1.222 | 1.682 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.231 | 1.199 | 1.670 | 5.187x | 7.222x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 6969.591 | 249.812 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 168.944 | 7039.205 | 250.434 | 41.666x | 1.482x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 1.090 | 1.800 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.595 | 1.087 | 1.841 | 1.829x | 3.097x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 6657.716 | 319.812 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 239.366 | 6689.503 | 316.029 | 27.947x | 1.320x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 1.196 | 1.627 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.221 | 1.215 | 1.619 | 5.502x | 7.334x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 6581.154 | 213.309 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 168.492 | 6757.844 | 225.711 | 40.108x | 1.340x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 1.071 | 1.786 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.594 | 1.077 | 1.812 | 1.813x | 3.050x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 6944.709 | 313.403 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 266.020 | 6690.369 | 320.142 | 25.150x | 1.203x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 1.192 | 1.625 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.222 | 1.193 | 1.638 | 5.367x | 7.373x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 6585.882 | 208.337 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 170.772 | 6617.229 | 217.439 | 38.749x | 1.273x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 7.931 | 8.479 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.585 | 8.075 | 8.373 | 13.812x | 14.321x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1113.735 | 1398.980 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 237.702 | 1180.739 | 1425.037 | 4.967x | 5.995x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 7.904 | 6.748 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.218 | 7.807 | 6.732 | 35.748x | 30.822x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1051.953 | 1355.000 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 147.733 | 1071.249 | 1336.677 | 7.251x | 9.048x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 7.690 | 8.340 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.591 | 7.763 | 8.242 | 13.142x | 13.953x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1142.157 | 1355.982 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 227.827 | 1209.805 | 1381.237 | 5.310x | 6.063x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 7.950 | 6.696 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.216 | 7.919 | 6.655 | 36.732x | 30.867x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1063.820 | 1312.146 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 151.377 | 1083.968 | 1616.634 | 7.161x | 10.680x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 7.866 | 8.189 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.586 | 7.811 | 8.399 | 13.332x | 14.335x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1239.460 | 1476.727 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 304.013 | 1325.799 | 1467.559 | 4.361x | 4.827x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 7.880 | 6.783 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.211 | 7.881 | 6.552 | 37.388x | 31.083x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1123.076 | 1405.377 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 197.990 | 1079.777 | 1359.787 | 5.454x | 6.868x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 8.987 | 8.399 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.591 | 8.948 | 8.455 | 15.133x | 14.299x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 2807.819 | 2977.134 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 738.066 | 2881.164 | 2920.847 | 3.904x | 3.957x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 9.055 | 6.533 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.225 | 8.921 | 6.557 | 39.682x | 29.164x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 2746.204 | 2874.792 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 643.578 | 2656.012 | 2882.930 | 4.127x | 4.480x |

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
