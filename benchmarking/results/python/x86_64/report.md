# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-07T05:44:53.110830+00:00
- Commit: `3d8d879`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.776 | 1.008 | 1.860 | 1.298x | 2.396x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.658 | 1.030 | 1.866 | 1.564x | 2.834x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 515.396 | 787.096 | 793.596 | 1.527x | 1.540x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 450.968 | 764.301 | 751.407 | 1.695x | 1.666x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.241 | 1.142 | 1.642 | 4.731x | 6.800x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.264 | 1.151 | 1.636 | 4.358x | 6.193x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 320.942 | 263.016 | 292.814 | 0.820x | 0.912x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 347.039 | 291.868 | 299.272 | 0.841x | 0.862x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.646 | 1.019 | 1.863 | 1.577x | 2.884x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.658 | 1.024 | 1.894 | 1.557x | 2.880x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 470.299 | 769.563 | 762.678 | 1.636x | 1.622x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 369.168 | 779.024 | 787.817 | 2.110x | 2.134x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.228 | 1.137 | 1.643 | 4.987x | 7.205x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.270 | 1.123 | 1.638 | 4.154x | 6.060x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 304.980 | 303.294 | 298.648 | 0.994x | 0.979x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 332.820 | 288.529 | 303.705 | 0.867x | 0.913x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.656 | 1.024 | 1.838 | 1.560x | 2.801x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.670 | 1.029 | 1.845 | 1.537x | 2.755x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 500.566 | 795.725 | 777.621 | 1.590x | 1.553x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 409.753 | 372.833 | 389.785 | 0.910x | 0.951x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.236 | 1.132 | 1.625 | 4.804x | 6.898x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.271 | 1.134 | 1.628 | 4.180x | 6.003x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 341.258 | 333.170 | 349.720 | 0.976x | 1.025x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 335.959 | 296.627 | 322.536 | 0.883x | 0.960x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.661 | 1.022 | 1.829 | 1.547x | 2.769x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.672 | 1.014 | 1.829 | 1.509x | 2.721x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 564.409 | 795.472 | 779.637 | 1.409x | 1.381x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 439.128 | 784.707 | 775.933 | 1.787x | 1.767x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.245 | 1.138 | 1.645 | 4.650x | 6.722x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.282 | 1.131 | 1.637 | 4.018x | 5.816x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 330.425 | 301.596 | 300.251 | 0.913x | 0.909x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 339.675 | 289.348 | 315.357 | 0.852x | 0.928x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.651 | 8.235 | 8.610 | 12.657x | 13.234x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.739 | 8.015 | 8.711 | 10.848x | 11.791x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 3239.746 | 512.374 | 484.646 | 0.158x | 0.150x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 113.116 | 545.265 | 497.448 | 4.820x | 4.398x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.237 | 7.991 | 6.821 | 33.683x | 28.753x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.266 | 8.101 | 6.873 | 30.442x | 25.828x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 3173.430 | 481.033 | 468.219 | 0.152x | 0.148x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 76.077 | 497.307 | 456.739 | 6.537x | 6.004x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.650 | 8.198 | 9.013 | 12.605x | 13.857x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.726 | 7.991 | 8.632 | 11.013x | 11.897x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 3333.567 | 517.613 | 486.392 | 0.155x | 0.146x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 112.629 | 525.281 | 485.138 | 4.664x | 4.307x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.238 | 8.045 | 6.776 | 33.855x | 28.515x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.270 | 7.952 | 6.775 | 29.442x | 25.083x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 3277.454 | 477.799 | 467.149 | 0.146x | 0.143x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 74.636 | 491.904 | 467.106 | 6.591x | 6.258x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.660 | 8.020 | 8.587 | 12.158x | 13.018x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.724 | 7.958 | 8.632 | 10.989x | 11.921x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4968.602 | 496.383 | 474.334 | 0.100x | 0.095x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 153.385 | 519.685 | 467.587 | 3.388x | 3.048x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.243 | 7.979 | 6.835 | 32.827x | 28.121x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.269 | 8.029 | 6.938 | 29.813x | 25.763x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4907.895 | 506.286 | 445.656 | 0.103x | 0.091x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 105.340 | 500.592 | 452.377 | 4.752x | 4.294x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.657 | 9.011 | 8.738 | 13.722x | 13.307x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.741 | 9.060 | 8.516 | 12.224x | 11.490x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1934.409 | 1534.706 | 2237.430 | 0.793x | 1.157x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 764.786 | 1589.957 | 2232.614 | 2.079x | 2.919x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.235 | 9.172 | 6.877 | 39.091x | 29.307x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.268 | 9.132 | 6.700 | 34.111x | 25.027x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1877.878 | 1579.280 | 2206.417 | 0.841x | 1.175x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 705.812 | 1579.202 | 2221.270 | 2.237x | 3.147x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 1.074 | 1.870 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.747 | 1.068 | 1.845 | 1.430x | 2.471x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 6676.734 | 357.186 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 257.590 | 6677.490 | 375.075 | 25.923x | 1.456x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 1.193 | 1.660 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.279 | 1.184 | 1.636 | 4.250x | 5.874x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 6581.413 | 245.370 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 169.572 | 6620.092 | 251.863 | 39.040x | 1.485x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 1.076 | 1.846 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.742 | 1.068 | 1.860 | 1.439x | 2.506x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 7002.241 | 373.661 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 257.604 | 6989.584 | 353.414 | 27.133x | 1.372x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 1.195 | 1.665 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.278 | 1.187 | 1.655 | 4.277x | 5.965x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 6951.481 | 246.008 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 180.343 | 6942.046 | 259.085 | 38.494x | 1.437x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 1.072 | 1.831 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.737 | 1.066 | 1.841 | 1.446x | 2.499x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 6743.707 | 339.848 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 265.717 | 6701.966 | 354.000 | 25.222x | 1.332x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 1.191 | 1.621 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.287 | 1.189 | 1.622 | 4.142x | 5.650x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 6606.861 | 225.747 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 172.504 | 6602.697 | 223.304 | 38.276x | 1.294x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 1.050 | 1.868 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.753 | 1.050 | 1.842 | 1.394x | 2.447x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 6697.194 | 326.738 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 275.074 | 6695.093 | 349.915 | 24.339x | 1.272x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 1.168 | 1.619 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.286 | 1.181 | 1.636 | 4.129x | 5.717x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 6616.978 | 214.035 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 184.989 | 6596.700 | 220.695 | 35.660x | 1.193x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 8.101 | 8.878 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.751 | 7.984 | 8.689 | 10.636x | 11.575x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1224.400 | 1455.260 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 226.614 | 1258.098 | 1464.522 | 5.552x | 6.463x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 7.900 | 6.965 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.268 | 7.930 | 6.852 | 29.533x | 25.519x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1119.221 | 1369.423 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 155.065 | 1137.387 | 1397.142 | 7.335x | 9.010x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 7.899 | 8.852 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.761 | 7.995 | 8.887 | 10.503x | 11.674x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1204.306 | 1481.008 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 235.299 | 1331.454 | 1446.409 | 5.659x | 6.147x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 8.139 | 6.914 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.271 | 7.981 | 6.916 | 29.502x | 25.567x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1110.924 | 1373.823 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 146.156 | 1131.866 | 1362.008 | 7.744x | 9.319x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 7.934 | 8.678 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.744 | 7.915 | 8.693 | 10.633x | 11.677x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1248.999 | 1504.058 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 329.797 | 1299.496 | 1420.871 | 3.940x | 4.308x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 7.946 | 6.756 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.275 | 7.917 | 6.969 | 28.756x | 25.312x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1129.069 | 1404.282 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 201.147 | 1189.140 | 1388.512 | 5.912x | 6.903x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 8.830 | 8.404 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.757 | 9.117 | 8.710 | 12.040x | 11.503x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 2902.745 | 2985.429 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 764.767 | 2917.728 | 3016.424 | 3.815x | 3.944x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 9.022 | 6.793 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.289 | 9.130 | 6.634 | 31.554x | 22.927x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 2855.366 | 2935.828 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 649.562 | 2897.831 | 2905.148 | 4.461x | 4.472x |

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
