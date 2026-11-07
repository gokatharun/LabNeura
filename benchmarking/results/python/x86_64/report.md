# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-07T04:17:14.329131+00:00
- Commit: `c169436`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.587 | 1.062 | 1.912 | 1.809x | 3.257x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.591 | 1.057 | 1.858 | 1.789x | 3.147x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 488.011 | 735.503 | 732.295 | 1.507x | 1.501x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 389.237 | 727.862 | 735.872 | 1.870x | 1.891x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.204 | 1.175 | 1.658 | 5.755x | 8.122x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.195 | 1.186 | 1.664 | 6.079x | 8.527x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 288.020 | 262.848 | 280.130 | 0.913x | 0.973x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 304.193 | 260.341 | 280.836 | 0.856x | 0.923x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.592 | 1.024 | 1.960 | 1.730x | 3.311x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.601 | 1.055 | 1.843 | 1.755x | 3.066x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 490.448 | 699.909 | 707.348 | 1.427x | 1.442x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 396.978 | 701.215 | 699.532 | 1.766x | 1.762x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.199 | 1.178 | 1.648 | 5.912x | 8.272x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.198 | 1.168 | 1.655 | 5.889x | 8.341x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 302.034 | 290.735 | 299.424 | 0.963x | 0.991x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 323.855 | 266.358 | 261.206 | 0.822x | 0.807x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.597 | 1.067 | 1.837 | 1.788x | 3.079x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.603 | 1.066 | 1.817 | 1.768x | 3.014x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 509.048 | 687.011 | 715.622 | 1.350x | 1.406x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 356.441 | 724.064 | 702.665 | 2.031x | 1.971x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.198 | 1.178 | 1.616 | 5.952x | 8.163x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.201 | 1.169 | 1.624 | 5.817x | 8.079x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 288.711 | 248.193 | 290.397 | 0.860x | 1.006x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 317.145 | 262.573 | 261.794 | 0.828x | 0.825x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.599 | 1.049 | 1.835 | 1.752x | 3.064x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.595 | 1.056 | 1.822 | 1.774x | 3.062x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 534.904 | 722.922 | 702.642 | 1.351x | 1.314x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 338.435 | 704.535 | 780.542 | 2.082x | 2.306x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.200 | 1.155 | 1.618 | 5.764x | 8.077x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.196 | 1.152 | 1.637 | 5.886x | 8.362x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 316.243 | 273.498 | 262.668 | 0.865x | 0.831x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 313.048 | 266.100 | 261.043 | 0.850x | 0.834x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.619 | 8.189 | 8.541 | 13.235x | 13.803x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.622 | 8.074 | 8.481 | 12.986x | 13.640x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 3235.089 | 496.755 | 413.731 | 0.154x | 0.128x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 103.273 | 494.071 | 443.142 | 4.784x | 4.291x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.207 | 8.238 | 6.718 | 39.880x | 32.520x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.213 | 8.088 | 6.710 | 38.007x | 31.534x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 3184.584 | 472.877 | 433.943 | 0.148x | 0.136x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 71.944 | 482.675 | 437.545 | 6.709x | 6.082x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.614 | 7.990 | 8.479 | 13.010x | 13.807x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.636 | 8.254 | 8.355 | 12.970x | 13.128x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 3330.780 | 522.656 | 453.047 | 0.157x | 0.136x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 104.082 | 516.961 | 433.403 | 4.967x | 4.164x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.214 | 8.219 | 6.629 | 38.372x | 30.948x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.208 | 8.148 | 6.645 | 39.248x | 32.009x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 3273.682 | 513.526 | 394.784 | 0.157x | 0.121x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 71.077 | 526.894 | 497.922 | 7.413x | 7.005x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.607 | 8.112 | 8.572 | 13.370x | 14.128x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.614 | 8.109 | 8.457 | 13.213x | 13.780x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4958.594 | 506.320 | 415.354 | 0.102x | 0.084x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 149.146 | 500.985 | 419.962 | 3.359x | 2.816x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.203 | 8.274 | 6.710 | 40.722x | 33.027x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.208 | 8.115 | 6.678 | 38.939x | 32.043x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4900.324 | 500.557 | 445.534 | 0.102x | 0.091x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 102.782 | 516.684 | 405.903 | 5.027x | 3.949x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.603 | 9.315 | 8.722 | 15.445x | 14.462x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.627 | 8.926 | 8.376 | 14.227x | 13.351x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1928.839 | 1522.474 | 2214.203 | 0.789x | 1.148x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 757.619 | 1450.978 | 2224.803 | 1.915x | 2.937x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.203 | 9.328 | 6.539 | 46.007x | 32.252x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.209 | 9.317 | 6.485 | 44.580x | 31.029x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1878.633 | 1439.099 | 2183.314 | 0.766x | 1.162x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 705.390 | 1443.822 | 2195.288 | 2.047x | 3.112x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 1.106 | 1.868 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.627 | 1.108 | 1.844 | 1.767x | 2.940x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 6580.196 | 243.811 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 253.011 | 6684.025 | 350.114 | 26.418x | 1.384x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 1.223 | 1.666 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.209 | 1.240 | 1.667 | 5.945x | 7.990x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 6572.851 | 238.994 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 149.349 | 6634.183 | 238.377 | 44.421x | 1.596x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 1.110 | 1.871 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.629 | 1.104 | 1.866 | 1.755x | 2.967x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 6999.412 | 360.497 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 254.175 | 6994.382 | 356.952 | 27.518x | 1.404x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 1.217 | 1.670 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.215 | 1.231 | 1.658 | 5.723x | 7.711x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 6898.636 | 242.722 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 147.183 | 6894.446 | 241.101 | 46.843x | 1.638x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 1.099 | 1.865 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.628 | 1.079 | 1.835 | 1.718x | 2.921x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 6677.435 | 320.617 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 250.509 | 6678.966 | 321.657 | 26.662x | 1.284x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 1.226 | 1.621 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.211 | 1.224 | 1.615 | 5.807x | 7.664x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 6572.249 | 215.643 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 145.555 | 6579.189 | 215.699 | 45.201x | 1.482x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 1.092 | 1.816 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.627 | 1.093 | 1.821 | 1.743x | 2.906x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 6671.794 | 316.560 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 270.166 | 6678.791 | 313.800 | 24.721x | 1.162x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 1.195 | 1.625 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.210 | 1.213 | 1.614 | 5.766x | 7.671x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 6585.204 | 209.917 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 161.531 | 6580.837 | 210.546 | 40.740x | 1.303x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 7.873 | 8.798 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.633 | 7.901 | 8.399 | 12.491x | 13.278x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1085.106 | 1416.119 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 222.280 | 1171.994 | 1386.702 | 5.273x | 6.239x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 7.960 | 6.702 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.213 | 7.894 | 6.722 | 36.991x | 31.501x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1016.535 | 1280.259 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 133.081 | 1073.187 | 1297.581 | 8.064x | 9.750x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 7.944 | 8.397 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.630 | 7.876 | 8.435 | 12.502x | 13.390x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1107.584 | 1389.642 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 219.907 | 1155.878 | 1427.550 | 5.256x | 6.492x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 8.222 | 6.693 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.219 | 8.212 | 6.994 | 37.460x | 31.905x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1090.355 | 1277.071 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 138.342 | 1091.958 | 1311.836 | 7.893x | 9.483x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 8.140 | 8.683 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.628 | 7.971 | 8.720 | 12.690x | 13.882x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1131.383 | 1456.349 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 299.534 | 1206.342 | 1445.722 | 4.027x | 4.827x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 8.042 | 6.894 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.214 | 8.091 | 6.595 | 37.843x | 30.845x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1074.125 | 1393.247 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 196.435 | 1143.002 | 1353.731 | 5.819x | 6.891x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 9.075 | 8.432 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.644 | 9.008 | 8.274 | 13.979x | 12.840x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 2769.539 | 2947.852 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 772.007 | 2966.097 | 2960.819 | 3.842x | 3.835x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 8.997 | 6.752 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.215 | 9.143 | 6.522 | 42.565x | 30.361x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 2752.555 | 2843.854 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 650.956 | 2788.373 | 2892.443 | 4.284x | 4.443x |

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
