# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-07T13:54:16.499439+00:00
- Commit: `ded92fd`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.735 | 1.024 | 1.821 | 1.392x | 2.478x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.605 | 1.017 | 1.825 | 1.682x | 3.019x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 519.203 | 756.578 | 745.263 | 1.457x | 1.435x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 411.082 | 750.492 | 752.250 | 1.826x | 1.830x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.219 | 1.129 | 1.616 | 5.166x | 7.392x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.250 | 1.141 | 1.623 | 4.559x | 6.483x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 286.376 | 269.570 | 285.588 | 0.941x | 0.997x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 315.261 | 295.929 | 301.189 | 0.939x | 0.955x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.615 | 1.016 | 1.834 | 1.653x | 2.984x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.604 | 1.019 | 1.834 | 1.686x | 3.036x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 506.244 | 742.532 | 735.955 | 1.467x | 1.454x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 398.005 | 772.298 | 792.952 | 1.940x | 1.992x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.223 | 1.132 | 1.666 | 5.079x | 7.475x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.252 | 1.126 | 1.631 | 4.471x | 6.474x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 298.307 | 302.194 | 279.183 | 1.013x | 0.936x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 315.589 | 271.992 | 323.573 | 0.862x | 1.025x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.613 | 1.015 | 1.911 | 1.656x | 3.115x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.602 | 1.020 | 1.835 | 1.694x | 3.047x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 524.123 | 762.076 | 747.632 | 1.454x | 1.426x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 390.458 | 751.757 | 750.517 | 1.925x | 1.922x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.222 | 1.127 | 1.589 | 5.065x | 7.146x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.257 | 1.130 | 1.596 | 4.400x | 6.218x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 289.868 | 272.892 | 290.457 | 0.941x | 1.002x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 321.895 | 262.953 | 255.149 | 0.817x | 0.793x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.761 | 1.009 | 1.830 | 1.325x | 2.404x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.610 | 1.005 | 1.815 | 1.648x | 2.976x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 557.713 | 737.395 | 787.083 | 1.322x | 1.411x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 380.186 | 797.240 | 784.095 | 2.097x | 2.062x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.229 | 1.128 | 1.615 | 4.929x | 7.056x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.254 | 1.102 | 1.596 | 4.332x | 6.276x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 304.701 | 272.373 | 311.884 | 0.894x | 1.024x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 308.669 | 271.109 | 288.279 | 0.878x | 0.934x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.622 | 8.109 | 8.290 | 13.038x | 13.329x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.682 | 7.927 | 8.494 | 11.619x | 12.449x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 3244.193 | 504.921 | 477.345 | 0.156x | 0.147x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 109.242 | 510.673 | 497.469 | 4.675x | 4.554x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.236 | 7.925 | 6.591 | 33.577x | 27.923x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.262 | 7.973 | 6.608 | 30.467x | 25.251x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 3185.409 | 472.519 | 447.656 | 0.148x | 0.141x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 73.765 | 544.143 | 411.925 | 7.377x | 5.584x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.627 | 8.148 | 8.206 | 13.000x | 13.092x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.687 | 7.905 | 8.261 | 11.502x | 12.019x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 3313.908 | 486.444 | 473.322 | 0.147x | 0.143x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 108.709 | 509.427 | 480.561 | 4.686x | 4.421x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.232 | 8.236 | 6.710 | 35.528x | 28.945x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.274 | 7.960 | 6.583 | 29.039x | 24.017x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 3256.047 | 488.690 | 446.329 | 0.150x | 0.137x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 72.761 | 470.661 | 489.493 | 6.469x | 6.727x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.631 | 7.930 | 8.309 | 12.571x | 13.172x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.690 | 7.992 | 8.550 | 11.588x | 12.398x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4950.365 | 531.318 | 482.710 | 0.107x | 0.098x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 153.101 | 504.784 | 466.877 | 3.297x | 3.049x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.252 | 7.989 | 6.680 | 31.667x | 26.480x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.272 | 8.043 | 6.547 | 29.560x | 24.061x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4900.559 | 476.605 | 453.993 | 0.097x | 0.093x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 104.897 | 469.127 | 411.482 | 4.472x | 3.923x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.646 | 9.131 | 7.935 | 14.134x | 12.282x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.707 | 8.963 | 8.011 | 12.679x | 11.332x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1931.854 | 1529.530 | 2230.215 | 0.792x | 1.154x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 761.167 | 1555.457 | 2245.736 | 2.044x | 2.950x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.245 | 9.236 | 6.389 | 37.718x | 26.094x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.276 | 9.276 | 6.428 | 33.594x | 23.279x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1882.314 | 1488.534 | 2209.073 | 0.791x | 1.174x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 705.287 | 1510.386 | 2224.432 | 2.142x | 3.154x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 1.054 | 1.842 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.675 | 1.052 | 1.835 | 1.558x | 2.717x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 6954.297 | 356.247 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 257.671 | 6706.892 | 357.149 | 26.029x | 1.386x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 1.172 | 1.633 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.264 | 1.167 | 1.635 | 4.426x | 6.201x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 6586.596 | 242.913 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 155.826 | 6607.166 | 240.973 | 42.401x | 1.546x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 1.061 | 1.840 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.677 | 1.055 | 1.852 | 1.558x | 2.733x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 7012.489 | 356.458 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 251.877 | 6987.763 | 361.181 | 27.743x | 1.434x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 1.167 | 1.639 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.259 | 1.164 | 1.643 | 4.500x | 6.353x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 7021.800 | 242.825 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 158.835 | 6971.792 | 245.590 | 43.893x | 1.546x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 1.060 | 1.828 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.702 | 1.059 | 1.826 | 1.509x | 2.603x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 6677.845 | 321.369 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 252.071 | 6686.447 | 330.816 | 26.526x | 1.312x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 1.168 | 1.612 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.269 | 1.170 | 1.610 | 4.346x | 5.982x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 6587.320 | 218.145 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 160.008 | 6886.380 | 213.516 | 43.038x | 1.334x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 1.043 | 1.832 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.707 | 1.051 | 1.823 | 1.487x | 2.578x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 6679.358 | 320.624 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 283.727 | 6703.830 | 326.258 | 23.628x | 1.150x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 1.162 | 1.615 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.269 | 1.161 | 1.621 | 4.308x | 6.016x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 6593.328 | 207.469 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 164.392 | 6598.136 | 216.937 | 40.137x | 1.320x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 7.830 | 8.197 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.686 | 7.830 | 8.280 | 11.419x | 12.076x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1175.133 | 1494.215 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 214.719 | 1212.395 | 1523.407 | 5.646x | 7.095x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 7.869 | 6.583 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.261 | 7.896 | 6.512 | 30.245x | 24.944x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1117.666 | 1401.800 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 145.543 | 1087.622 | 1370.441 | 7.473x | 9.416x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 7.820 | 8.108 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.690 | 7.805 | 8.187 | 11.307x | 11.860x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1264.011 | 1415.267 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 216.523 | 1182.105 | 1475.315 | 5.459x | 6.814x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 7.897 | 6.570 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.263 | 8.066 | 6.601 | 30.638x | 25.071x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1095.939 | 1396.667 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 144.588 | 1088.343 | 1349.631 | 7.527x | 9.334x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 7.767 | 8.201 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.686 | 7.829 | 8.373 | 11.415x | 12.207x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1161.994 | 1476.377 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 291.674 | 1226.599 | 1522.174 | 4.205x | 5.219x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 8.113 | 6.496 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.277 | 7.886 | 6.553 | 28.457x | 23.645x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1071.521 | 1400.943 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 196.634 | 1139.273 | 1436.427 | 5.794x | 7.305x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 8.850 | 7.977 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.707 | 8.969 | 8.113 | 12.688x | 11.476x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 2922.672 | 3027.859 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 763.987 | 2871.964 | 2997.167 | 3.759x | 3.923x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 9.081 | 6.632 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.276 | 8.924 | 6.368 | 32.391x | 23.114x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 2730.179 | 2883.271 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 649.189 | 2749.340 | 2956.550 | 4.235x | 4.554x |

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
