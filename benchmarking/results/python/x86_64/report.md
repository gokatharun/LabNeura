# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-08T06:59:28.291716+00:00
- Commit: `1089815`
- CPU: Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz (x86_64)
- OS: Linux-7.0.0-1017-azure-x86_64-with-glibc2.39
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.4.2
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.723 | 0.999 | 2.312 | 1.381x | 3.199x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.624 | 1.007 | 2.255 | 1.612x | 3.610x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 1190.788 | 1705.509 | 1414.372 | 1.432x | 1.188x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 1244.999 | 1800.147 | 1346.558 | 1.446x | 1.082x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.221 | 1.150 | 1.691 | 5.205x | 7.657x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.412 | 1.136 | 1.692 | 2.757x | 4.106x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 766.126 | 714.834 | 777.680 | 0.933x | 1.015x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 696.108 | 793.021 | 753.923 | 1.139x | 1.083x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.726 | 0.994 | 2.225 | 1.369x | 3.066x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.617 | 0.988 | 2.256 | 1.603x | 3.659x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 1218.325 | 1883.448 | 1370.360 | 1.546x | 1.125x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 1276.221 | 1823.527 | 1402.028 | 1.429x | 1.099x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.215 | 1.112 | 1.697 | 5.184x | 7.911x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.254 | 1.110 | 1.956 | 4.379x | 7.714x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 778.153 | 715.928 | 773.367 | 0.920x | 0.994x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 702.704 | 746.475 | 762.249 | 1.062x | 1.085x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.611 | 0.989 | 2.183 | 1.619x | 3.575x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.616 | 0.991 | 2.155 | 1.608x | 3.497x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 1174.461 | 1736.342 | 1291.885 | 1.478x | 1.100x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 1293.831 | 1548.096 | 1010.735 | 1.197x | 0.781x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.210 | 1.113 | 1.672 | 5.295x | 7.952x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.256 | 1.172 | 1.810 | 4.585x | 7.078x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 774.756 | 688.889 | 709.678 | 0.889x | 0.916x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 680.752 | 694.749 | 707.135 | 1.021x | 1.039x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.726 | 1.144 | 2.423 | 1.576x | 3.339x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.604 | 0.977 | 2.185 | 1.619x | 3.621x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 1196.292 | 1754.121 | 1354.334 | 1.466x | 1.132x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 1202.003 | 1878.478 | 1377.144 | 1.563x | 1.146x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.209 | 1.095 | 1.707 | 5.226x | 8.151x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.253 | 1.092 | 1.691 | 4.317x | 6.686x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 781.564 | 750.759 | 749.229 | 0.961x | 0.959x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 697.070 | 741.564 | 757.064 | 1.064x | 1.086x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.635 | 8.409 | 9.584 | 13.247x | 15.098x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.699 | 8.313 | 9.686 | 11.898x | 13.863x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 3976.659 | 1408.190 | 1455.238 | 0.354x | 0.366x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 263.558 | 1347.360 | 1451.958 | 5.112x | 5.509x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.224 | 8.441 | 7.466 | 37.709x | 33.356x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.261 | 8.378 | 7.357 | 32.091x | 28.181x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 3904.073 | 1373.921 | 1477.670 | 0.352x | 0.378x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 172.919 | 1272.993 | 1299.354 | 7.362x | 7.514x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.637 | 8.606 | 9.646 | 13.501x | 15.133x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.700 | 8.335 | 9.530 | 11.911x | 13.618x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 3900.917 | 1210.898 | 1353.986 | 0.310x | 0.347x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 255.789 | 1208.224 | 1297.833 | 4.724x | 5.074x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.220 | 8.481 | 7.500 | 38.479x | 34.028x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.258 | 8.522 | 7.461 | 33.078x | 28.961x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 3769.258 | 1236.594 | 1260.893 | 0.328x | 0.335x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 170.863 | 1200.254 | 1251.705 | 7.025x | 7.326x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.671 | 8.444 | 9.577 | 12.586x | 14.275x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.722 | 8.320 | 9.511 | 11.516x | 13.165x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 5547.402 | 1191.141 | 1170.503 | 0.215x | 0.211x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 272.573 | 1225.261 | 1211.371 | 4.495x | 4.444x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.219 | 8.462 | 7.453 | 38.593x | 33.993x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.256 | 8.428 | 7.389 | 32.935x | 28.873x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 5392.139 | 1183.425 | 1208.481 | 0.219x | 0.224x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 202.740 | 1210.116 | 1209.958 | 5.969x | 5.968x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.618 | 9.362 | 9.523 | 15.144x | 15.403x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.950 | 10.478 | 9.743 | 11.031x | 10.257x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1879.291 | 5013.102 | 2702.410 | 2.668x | 1.438x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 870.681 | 5713.223 | 2764.608 | 6.562x | 3.175x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.219 | 9.490 | 7.391 | 43.388x | 33.792x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.261 | 9.522 | 7.517 | 36.479x | 28.800x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1754.121 | 5368.319 | 2709.328 | 3.060x | 1.545x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 759.550 | 5744.884 | 2762.462 | 7.564x | 3.637x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 1.056 | 2.186 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.703 | 1.059 | 2.192 | 1.506x | 3.118x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 5565.476 | 669.889 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 563.536 | 5587.413 | 631.271 | 9.915x | 1.120x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 1.452 | 1.694 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.273 | 1.247 | 1.732 | 4.570x | 6.348x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 5299.728 | 472.376 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 399.170 | 5333.121 | 477.731 | 13.361x | 1.197x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 1.057 | 2.379 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.848 | 1.058 | 2.243 | 1.247x | 2.644x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 6022.922 | 579.153 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 552.950 | 6026.612 | 581.350 | 10.899x | 1.051x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 1.267 | 1.738 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.273 | 1.198 | 1.751 | 4.391x | 6.416x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 5771.453 | 443.375 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 401.119 | 5789.031 | 467.720 | 14.432x | 1.166x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 1.220 | 2.249 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.729 | 1.064 | 2.186 | 1.459x | 2.998x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 5781.956 | 570.343 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 541.338 | 5777.983 | 646.555 | 10.674x | 1.194x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 1.256 | 1.714 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.265 | 1.262 | 1.688 | 4.759x | 6.365x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 5354.403 | 433.859 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 395.497 | 5350.912 | 449.766 | 13.530x | 1.137x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 1.072 | 2.377 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.724 | 1.095 | 2.217 | 1.513x | 3.062x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 5941.121 | 562.203 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 554.665 | 5949.082 | 558.836 | 10.726x | 1.008x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 1.212 | 1.739 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.265 | 1.225 | 1.732 | 4.619x | 6.527x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 5712.370 | 464.970 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 401.607 | 5721.548 | 470.589 | 14.247x | 1.172x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 8.197 | 9.705 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.700 | 8.260 | 9.665 | 11.805x | 13.814x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 3704.749 | 4114.840 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 486.588 | 3554.424 | 3694.313 | 7.305x | 7.592x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 8.169 | 7.471 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.257 | 8.400 | 7.402 | 32.648x | 28.767x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 3451.625 | 3665.685 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 318.820 | 3446.020 | 3580.348 | 10.809x | 11.230x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 8.089 | 9.631 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.726 | 8.245 | 9.576 | 11.354x | 13.188x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 3333.855 | 3703.845 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 492.889 | 3277.689 | 3572.305 | 6.650x | 7.248x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 8.301 | 7.435 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.262 | 8.236 | 7.394 | 31.457x | 28.241x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 3323.853 | 3584.834 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 323.390 | 3127.345 | 3940.278 | 9.671x | 12.184x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 8.261 | 9.459 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.726 | 8.151 | 9.552 | 11.233x | 13.163x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 3398.675 | 3745.610 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 513.169 | 3276.451 | 3750.088 | 6.385x | 7.308x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 8.296 | 7.225 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.258 | 8.201 | 7.370 | 31.745x | 28.529x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 3490.622 | 3472.858 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 384.581 | 3549.352 | 3587.501 | 9.229x | 9.328x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 9.154 | 9.550 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.704 | 9.132 | 9.663 | 12.966x | 13.720x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 9252.275 | 6009.629 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 993.145 | 8745.162 | 6101.693 | 8.806x | 6.144x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 9.292 | 7.357 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.266 | 9.341 | 7.396 | 35.081x | 27.777x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 8709.177 | 5632.456 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 793.794 | 9218.571 | 6142.522 | 11.613x | 7.738x |

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
