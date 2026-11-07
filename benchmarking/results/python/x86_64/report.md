# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-07T16:25:57.612895+00:00
- Commit: `895b763`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.655 | 1.036 | 1.942 | 1.581x | 2.963x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.637 | 1.036 | 1.897 | 1.626x | 2.979x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 525.611 | 774.387 | 755.592 | 1.473x | 1.438x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 385.209 | 757.238 | 792.847 | 1.966x | 2.058x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.223 | 1.176 | 1.694 | 5.273x | 7.596x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.245 | 1.167 | 1.706 | 4.763x | 6.962x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 296.651 | 277.851 | 310.619 | 0.937x | 1.047x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 330.359 | 282.299 | 289.735 | 0.855x | 0.877x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.649 | 1.038 | 1.923 | 1.599x | 2.961x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.640 | 1.034 | 1.930 | 1.616x | 3.017x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 526.610 | 843.909 | 817.690 | 1.603x | 1.553x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 411.395 | 341.637 | 345.354 | 0.830x | 0.839x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.220 | 1.151 | 1.736 | 5.234x | 7.896x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.245 | 1.160 | 1.723 | 4.733x | 7.030x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 299.303 | 295.201 | 400.732 | 0.986x | 1.339x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 343.468 | 264.265 | 285.693 | 0.769x | 0.832x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.782 | 1.026 | 1.890 | 1.312x | 2.416x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.639 | 1.037 | 1.899 | 1.624x | 2.973x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 523.379 | 755.807 | 762.798 | 1.444x | 1.457x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 416.043 | 365.409 | 347.035 | 0.878x | 0.834x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.227 | 1.158 | 1.678 | 5.093x | 7.380x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.253 | 1.156 | 1.681 | 4.570x | 6.647x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 303.724 | 267.877 | 284.709 | 0.882x | 0.937x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 331.891 | 281.956 | 283.937 | 0.850x | 0.856x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.657 | 1.024 | 1.890 | 1.558x | 2.874x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.652 | 1.023 | 1.891 | 1.569x | 2.899x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 570.923 | 826.559 | 792.468 | 1.448x | 1.388x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 375.683 | 743.790 | 748.859 | 1.980x | 1.993x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.224 | 1.144 | 1.691 | 5.113x | 7.555x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.253 | 1.158 | 1.694 | 4.571x | 6.687x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 312.360 | 278.479 | 342.619 | 0.892x | 1.097x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 336.894 | 267.987 | 292.949 | 0.795x | 0.870x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.670 | 8.215 | 8.744 | 12.256x | 13.046x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.719 | 7.962 | 9.029 | 11.078x | 12.563x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 3243.070 | 478.369 | 426.839 | 0.148x | 0.132x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 113.591 | 461.642 | 430.798 | 4.064x | 3.793x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.227 | 8.117 | 6.809 | 35.753x | 29.991x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.262 | 7.999 | 6.824 | 30.518x | 26.036x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 3185.008 | 438.139 | 396.499 | 0.138x | 0.124x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 76.867 | 435.262 | 409.732 | 5.663x | 5.330x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.658 | 7.982 | 8.454 | 12.123x | 12.839x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.724 | 8.019 | 8.505 | 11.076x | 11.747x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 3324.629 | 474.483 | 430.347 | 0.143x | 0.129x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 114.687 | 472.513 | 416.043 | 4.120x | 3.628x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.234 | 8.045 | 6.810 | 34.376x | 29.099x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.262 | 8.405 | 6.772 | 32.093x | 25.857x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 3259.097 | 457.632 | 398.924 | 0.140x | 0.122x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 78.084 | 481.196 | 511.147 | 6.163x | 6.546x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.664 | 7.937 | 8.574 | 11.946x | 12.905x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.704 | 8.278 | 8.506 | 11.757x | 12.081x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4972.038 | 493.764 | 423.958 | 0.099x | 0.085x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 154.674 | 463.991 | 456.908 | 3.000x | 2.954x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.238 | 8.120 | 6.705 | 34.082x | 28.143x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.261 | 8.107 | 6.718 | 31.025x | 25.711x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4898.396 | 457.792 | 406.626 | 0.093x | 0.083x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 106.661 | 480.273 | 464.458 | 4.503x | 4.355x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.656 | 9.237 | 8.517 | 14.076x | 12.980x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.723 | 9.332 | 8.550 | 12.901x | 11.820x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1945.702 | 1604.671 | 2243.168 | 0.825x | 1.153x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 770.509 | 1509.956 | 2229.447 | 1.960x | 2.893x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.242 | 9.419 | 6.717 | 38.852x | 27.706x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.261 | 9.396 | 6.693 | 35.960x | 25.617x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1881.403 | 1517.192 | 2224.893 | 0.806x | 1.183x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 705.786 | 1502.330 | 2221.632 | 2.129x | 3.148x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 1.059 | 1.904 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.716 | 1.097 | 1.906 | 1.533x | 2.663x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 6682.915 | 358.096 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 265.511 | 6704.309 | 364.815 | 25.251x | 1.374x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 1.208 | 1.692 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.263 | 1.178 | 1.700 | 4.475x | 6.458x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 6611.525 | 245.610 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 156.700 | 6598.018 | 248.339 | 42.106x | 1.585x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 1.107 | 1.933 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.704 | 1.103 | 1.944 | 1.566x | 2.760x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 7006.044 | 371.387 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 262.546 | 7002.160 | 351.973 | 26.670x | 1.341x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 1.225 | 1.745 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.276 | 1.227 | 1.730 | 4.448x | 6.271x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 6975.765 | 253.519 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 162.622 | 7093.972 | 246.031 | 43.622x | 1.513x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 1.101 | 1.905 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.723 | 1.096 | 1.919 | 1.517x | 2.654x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 6672.909 | 332.638 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 265.415 | 6686.492 | 328.937 | 25.193x | 1.239x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 1.217 | 1.671 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.274 | 1.216 | 1.692 | 4.447x | 6.184x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 6617.658 | 229.359 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 169.323 | 6679.255 | 220.425 | 39.447x | 1.302x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 1.092 | 1.998 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.719 | 1.103 | 1.885 | 1.534x | 2.622x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 6709.821 | 339.726 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 284.403 | 6685.576 | 327.255 | 23.507x | 1.151x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 1.223 | 1.700 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.263 | 1.224 | 1.690 | 4.652x | 6.422x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 6636.123 | 211.239 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 165.928 | 6600.976 | 208.137 | 39.782x | 1.254x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 7.886 | 8.582 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.722 | 8.033 | 8.714 | 11.127x | 12.071x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1113.750 | 1393.825 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 219.272 | 1167.707 | 1407.734 | 5.325x | 6.420x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 7.989 | 6.884 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.262 | 8.315 | 6.792 | 31.748x | 25.934x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1020.025 | 1292.559 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 137.361 | 1062.911 | 1324.528 | 7.738x | 9.643x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 7.936 | 8.720 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.732 | 7.893 | 8.791 | 10.777x | 12.003x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1137.915 | 1398.367 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 234.144 | 1178.138 | 1386.908 | 5.032x | 5.923x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 8.033 | 6.778 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.277 | 7.965 | 6.787 | 28.723x | 24.475x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1037.114 | 1305.652 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 137.419 | 1075.682 | 1307.714 | 7.828x | 9.516x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 8.120 | 8.531 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.733 | 7.932 | 8.498 | 10.819x | 11.592x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1160.785 | 1468.668 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 304.712 | 1210.551 | 1441.822 | 3.973x | 4.732x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 7.986 | 6.689 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.280 | 8.160 | 6.747 | 29.129x | 24.086x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1113.788 | 1358.451 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 192.414 | 1093.553 | 1359.835 | 5.683x | 7.067x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 9.244 | 8.603 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.719 | 9.264 | 8.332 | 12.893x | 11.595x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 2761.174 | 2929.654 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 761.990 | 2719.278 | 2946.104 | 3.569x | 3.866x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 9.299 | 6.689 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.270 | 9.286 | 6.661 | 34.356x | 24.645x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 2685.668 | 2849.762 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 648.131 | 2652.590 | 2871.617 | 4.093x | 4.431x |

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
