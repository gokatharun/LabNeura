# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-07T01:31:27.152636+00:00
- Commit: `d7f5ee0`
- CPU: AMD EPYC 9V74 80-Core Processor (x86_64)
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.333 | 0.794 | 1.406 | 2.381x | 4.217x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.335 | 0.786 | 1.383 | 2.345x | 4.125x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 397.027 | 227.894 | 246.269 | 0.574x | 0.620x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 318.151 | 229.714 | 237.779 | 0.722x | 0.747x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.142 | 0.887 | 1.287 | 6.225x | 9.033x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.142 | 0.878 | 1.273 | 6.203x | 8.987x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 232.453 | 179.751 | 187.092 | 0.773x | 0.805x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 300.917 | 184.984 | 195.032 | 0.615x | 0.648x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.349 | 0.769 | 1.394 | 2.207x | 3.999x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.337 | 0.786 | 1.393 | 2.330x | 4.130x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 399.070 | 233.432 | 274.765 | 0.585x | 0.689x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 322.187 | 229.692 | 245.368 | 0.713x | 0.762x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.142 | 0.879 | 1.296 | 6.180x | 9.115x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.140 | 0.919 | 1.308 | 6.548x | 9.318x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 228.723 | 185.312 | 200.064 | 0.810x | 0.875x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 300.857 | 178.019 | 192.195 | 0.592x | 0.639x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.336 | 0.788 | 1.358 | 2.344x | 4.040x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.332 | 0.782 | 1.365 | 2.357x | 4.113x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 387.184 | 233.968 | 265.163 | 0.604x | 0.685x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 338.058 | 253.700 | 262.166 | 0.750x | 0.776x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.141 | 0.892 | 1.214 | 6.320x | 8.597x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.143 | 0.898 | 1.215 | 6.298x | 8.519x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 230.718 | 176.812 | 183.798 | 0.766x | 0.797x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 312.967 | 185.650 | 185.563 | 0.593x | 0.593x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.339 | 0.767 | 1.405 | 2.260x | 4.141x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.331 | 0.766 | 1.438 | 2.315x | 4.346x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 424.942 | 227.363 | 230.691 | 0.535x | 0.543x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 340.667 | 225.225 | 237.834 | 0.661x | 0.698x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.140 | 0.866 | 1.262 | 6.168x | 8.988x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.145 | 0.879 | 1.267 | 6.047x | 8.712x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 233.595 | 189.781 | 179.248 | 0.812x | 0.767x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 302.687 | 178.978 | 186.331 | 0.591x | 0.616x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.351 | 5.982 | 6.340 | 17.048x | 18.067x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.360 | 6.008 | 6.302 | 16.701x | 17.517x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 2888.229 | 324.499 | 293.344 | 0.112x | 0.102x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 73.732 | 335.471 | 314.256 | 4.550x | 4.262x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.153 | 6.163 | 4.976 | 40.378x | 32.602x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.148 | 6.091 | 4.959 | 41.092x | 33.458x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 2853.446 | 345.336 | 303.657 | 0.121x | 0.106x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 55.608 | 333.463 | 295.174 | 5.997x | 5.308x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.344 | 6.056 | 6.306 | 17.628x | 18.356x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.359 | 6.062 | 6.193 | 16.880x | 17.243x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 2676.663 | 319.791 | 293.208 | 0.119x | 0.110x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 73.925 | 319.768 | 293.299 | 4.326x | 3.968x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.147 | 6.073 | 5.049 | 41.195x | 34.249x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.152 | 6.070 | 5.079 | 39.868x | 33.362x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 2645.836 | 323.809 | 294.595 | 0.122x | 0.111x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 55.272 | 321.793 | 293.286 | 5.822x | 5.306x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.347 | 6.030 | 6.229 | 17.370x | 17.945x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.353 | 6.020 | 6.145 | 17.056x | 17.413x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4308.474 | 331.838 | 279.809 | 0.077x | 0.065x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 82.058 | 333.897 | 279.903 | 4.069x | 3.411x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.148 | 6.042 | 4.986 | 40.875x | 33.732x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.150 | 6.129 | 4.956 | 40.858x | 33.033x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4267.247 | 337.877 | 283.042 | 0.079x | 0.066x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 59.017 | 370.659 | 293.571 | 6.281x | 4.974x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.347 | 6.677 | 6.329 | 19.257x | 18.253x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.368 | 6.780 | 6.330 | 18.416x | 17.196x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1681.204 | 1171.640 | 1843.428 | 0.697x | 1.096x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 582.559 | 1182.129 | 1842.790 | 2.029x | 3.163x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.149 | 6.862 | 4.991 | 45.983x | 33.450x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.151 | 6.748 | 4.991 | 44.795x | 33.130x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1643.990 | 1178.589 | 1842.727 | 0.717x | 1.121x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 547.180 | 1214.434 | 1845.445 | 2.219x | 3.373x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.813 | 1.380 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.362 | 0.794 | 1.388 | 2.193x | 3.834x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 5560.886 | 253.629 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 187.180 | 5481.743 | 159.771 | 29.286x | 0.854x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.938 | 1.260 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.148 | 0.963 | 1.273 | 6.523x | 8.623x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 5493.521 | 142.911 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 116.719 | 5476.465 | 140.079 | 46.920x | 1.200x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.814 | 1.398 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.361 | 0.809 | 1.391 | 2.242x | 3.854x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 5851.088 | 249.219 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 185.539 | 5753.400 | 160.850 | 31.009x | 0.867x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.942 | 1.285 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.148 | 0.925 | 1.295 | 6.234x | 8.724x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 5760.813 | 141.730 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 116.572 | 5747.587 | 141.376 | 49.305x | 1.213x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.818 | 1.396 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.368 | 0.817 | 1.368 | 2.223x | 3.721x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 5565.434 | 245.673 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 181.631 | 5485.159 | 148.083 | 30.199x | 0.815x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.938 | 1.212 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.149 | 0.949 | 1.211 | 6.384x | 8.148x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 5486.800 | 124.507 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 115.694 | 5521.384 | 124.713 | 47.724x | 1.078x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.799 | 1.380 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.361 | 0.800 | 1.370 | 2.218x | 3.800x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 5564.161 | 242.740 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 198.089 | 5480.235 | 151.933 | 27.666x | 0.767x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.928 | 1.257 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.147 | 0.935 | 1.246 | 6.376x | 8.499x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 5512.020 | 133.295 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 122.963 | 5480.015 | 131.176 | 44.566x | 1.067x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 5.959 | 6.321 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.354 | 5.921 | 6.136 | 16.738x | 17.348x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 891.699 | 967.800 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 159.914 | 843.239 | 881.186 | 5.273x | 5.510x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.078 | 4.951 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.146 | 6.003 | 5.010 | 41.000x | 34.219x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 868.947 | 946.900 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 112.362 | 902.711 | 941.929 | 8.034x | 8.383x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 5.989 | 6.189 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.362 | 6.058 | 6.192 | 16.736x | 17.107x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 906.355 | 969.658 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 170.062 | 819.474 | 884.546 | 4.819x | 5.201x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 5.985 | 5.061 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.151 | 6.040 | 5.089 | 39.885x | 33.603x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 925.125 | 933.593 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 114.002 | 876.034 | 946.486 | 7.684x | 8.302x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.039 | 6.349 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.355 | 5.934 | 6.137 | 16.737x | 17.309x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 869.078 | 955.092 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 178.277 | 953.272 | 955.303 | 5.347x | 5.359x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.029 | 4.980 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.150 | 6.128 | 5.005 | 40.791x | 33.316x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 897.664 | 918.145 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 117.335 | 878.240 | 932.296 | 7.485x | 7.946x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.631 | 6.133 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.368 | 6.649 | 6.136 | 18.059x | 16.666x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 2381.579 | 2407.740 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 597.661 | 2399.544 | 2376.428 | 4.015x | 3.976x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.662 | 4.914 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.150 | 6.722 | 4.906 | 44.743x | 32.656x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 2442.360 | 2356.908 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 511.805 | 2385.010 | 2340.910 | 4.660x | 4.574x |

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
