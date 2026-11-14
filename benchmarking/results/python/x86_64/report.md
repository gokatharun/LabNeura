# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-14T11:05:55.108218+00:00
- Commit: `fe410e3`
- CPU: INTEL(R) XEON(R) PLATINUM 8573C (x86_64)
- OS: Linux-7.0.0-1019-azure-x86_64-with-glibc2.43
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.492 | 0.773 | 1.406 | 1.573x | 2.860x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.445 | 0.775 | 1.437 | 1.741x | 3.230x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 832.610 | 926.961 | 929.536 | 1.113x | 1.116x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 708.062 | 959.632 | 915.663 | 1.355x | 1.293x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.175 | 0.947 | 1.448 | 5.425x | 8.296x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.176 | 0.928 | 1.460 | 5.277x | 8.301x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 493.117 | 518.816 | 539.554 | 1.052x | 1.094x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 484.968 | 518.739 | 539.987 | 1.070x | 1.113x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.491 | 0.777 | 1.403 | 1.583x | 2.857x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.439 | 0.770 | 1.414 | 1.753x | 3.221x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 820.724 | 931.342 | 895.399 | 1.135x | 1.091x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 686.308 | 919.367 | 911.337 | 1.340x | 1.328x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.177 | 0.923 | 1.322 | 5.210x | 7.459x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.174 | 0.897 | 1.319 | 5.158x | 7.588x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 488.532 | 522.886 | 537.620 | 1.070x | 1.100x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 488.015 | 523.184 | 536.462 | 1.072x | 1.099x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.493 | 0.782 | 1.396 | 1.587x | 2.833x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.441 | 0.778 | 1.414 | 1.763x | 3.207x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 826.690 | 941.590 | 914.231 | 1.139x | 1.106x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 676.404 | 967.495 | 872.173 | 1.430x | 1.289x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.171 | 0.897 | 1.345 | 5.232x | 7.849x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.173 | 0.901 | 1.335 | 5.206x | 7.712x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 489.155 | 519.259 | 526.532 | 1.062x | 1.076x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 486.280 | 519.274 | 523.134 | 1.068x | 1.076x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.479 | 0.756 | 1.442 | 1.579x | 3.011x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.446 | 0.759 | 1.453 | 1.703x | 3.259x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 787.493 | 965.956 | 891.420 | 1.227x | 1.132x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 673.491 | 982.403 | 897.012 | 1.459x | 1.332x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.176 | 0.887 | 1.380 | 5.033x | 7.825x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.174 | 0.891 | 1.400 | 5.117x | 8.043x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 482.190 | 521.367 | 514.035 | 1.081x | 1.066x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 491.139 | 520.747 | 540.813 | 1.060x | 1.101x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.501 | 6.688 | 7.014 | 13.354x | 14.005x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.490 | 6.853 | 7.080 | 13.974x | 14.437x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 3511.242 | 881.369 | 912.967 | 0.251x | 0.260x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 174.160 | 872.005 | 894.354 | 5.007x | 5.135x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.177 | 6.905 | 5.924 | 38.968x | 33.430x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.178 | 6.805 | 5.781 | 38.318x | 32.551x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 3412.978 | 933.836 | 902.696 | 0.274x | 0.264x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 126.053 | 873.126 | 885.915 | 6.927x | 7.028x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.489 | 6.804 | 7.139 | 13.907x | 14.591x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.484 | 6.855 | 6.989 | 14.161x | 14.438x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 3304.311 | 874.934 | 886.476 | 0.265x | 0.268x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 174.645 | 877.202 | 903.490 | 5.023x | 5.173x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.181 | 6.861 | 5.687 | 37.938x | 31.444x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.181 | 6.873 | 5.689 | 37.910x | 31.379x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 3225.826 | 894.673 | 900.550 | 0.277x | 0.279x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 128.806 | 888.248 | 891.721 | 6.896x | 6.923x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.498 | 6.782 | 7.109 | 13.607x | 14.262x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.488 | 6.762 | 6.902 | 13.856x | 14.144x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 6753.755 | 886.751 | 916.886 | 0.131x | 0.136x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 218.751 | 881.940 | 905.608 | 4.032x | 4.140x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.179 | 6.679 | 5.615 | 37.334x | 31.384x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.182 | 6.896 | 5.927 | 37.803x | 32.493x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 6647.437 | 894.559 | 905.087 | 0.135x | 0.136x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 159.803 | 871.227 | 893.695 | 5.452x | 5.592x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.492 | 7.640 | 7.092 | 15.528x | 14.415x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.497 | 7.623 | 7.250 | 15.325x | 14.575x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 2171.448 | 2462.520 | 2631.340 | 1.134x | 1.212x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 858.068 | 2559.174 | 2669.813 | 2.982x | 3.111x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.186 | 7.689 | 5.739 | 41.397x | 30.899x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.187 | 7.602 | 5.784 | 40.754x | 31.009x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 2054.257 | 2489.770 | 2640.981 | 1.212x | 1.286x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 763.276 | 2541.591 | 2647.222 | 3.330x | 3.468x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.880 | 1.404 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.494 | 0.830 | 1.426 | 1.681x | 2.888x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 7130.264 | 467.714 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 469.636 | 6855.384 | 444.183 | 14.597x | 0.946x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.973 | 1.476 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.178 | 0.942 | 1.476 | 5.302x | 8.305x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 6745.282 | 386.374 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 326.778 | 7408.348 | 385.442 | 22.671x | 1.180x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.859 | 1.466 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.493 | 0.822 | 1.414 | 1.666x | 2.866x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 6300.909 | 446.277 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 477.057 | 6163.562 | 458.146 | 12.920x | 0.960x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.964 | 1.337 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.182 | 0.961 | 1.331 | 5.289x | 7.322x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 5986.979 | 388.674 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 330.511 | 6048.726 | 379.414 | 18.301x | 1.148x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.843 | 1.426 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.501 | 0.826 | 1.397 | 1.650x | 2.791x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 6982.279 | 448.853 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 481.405 | 7041.260 | 449.273 | 14.626x | 0.933x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.964 | 1.367 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.191 | 0.962 | 1.352 | 5.026x | 7.063x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 6991.827 | 375.236 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 327.496 | 7026.753 | 367.573 | 21.456x | 1.122x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.810 | 1.418 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.498 | 0.814 | 1.413 | 1.634x | 2.837x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 7128.816 | 454.876 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 490.318 | 7038.682 | 439.398 | 14.355x | 0.896x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.942 | 1.377 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.185 | 0.954 | 1.351 | 5.147x | 7.293x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 7008.543 | 377.928 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 329.049 | 6968.102 | 402.540 | 21.176x | 1.223x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.563 | 6.988 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.495 | 6.607 | 7.018 | 13.360x | 14.191x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1909.661 | 2076.274 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 406.815 | 1843.860 | 2058.235 | 4.532x | 5.059x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.575 | 5.735 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.180 | 6.766 | 5.686 | 37.530x | 31.538x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1847.752 | 1984.245 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 249.218 | 1827.910 | 1987.565 | 7.335x | 7.975x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.562 | 7.050 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.500 | 6.691 | 7.083 | 13.380x | 14.165x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1838.273 | 1996.466 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 400.501 | 1897.632 | 1968.716 | 4.738x | 4.916x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.553 | 5.689 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.182 | 6.628 | 5.721 | 36.473x | 31.484x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1810.294 | 1945.167 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 249.325 | 1880.018 | 1983.674 | 7.540x | 7.956x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.672 | 6.928 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.489 | 6.700 | 7.012 | 13.702x | 14.341x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1889.068 | 2178.582 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 475.679 | 1877.795 | 2049.441 | 3.948x | 4.308x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.762 | 5.776 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.185 | 6.697 | 5.630 | 36.258x | 30.479x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1850.691 | 2075.860 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 317.381 | 1826.463 | 2062.958 | 5.755x | 6.500x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 7.576 | 6.981 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.492 | 7.422 | 6.994 | 15.098x | 14.227x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3897.813 | 4655.857 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 956.238 | 3936.895 | 4711.380 | 4.117x | 4.927x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 7.571 | 5.756 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.188 | 7.479 | 5.580 | 39.755x | 29.659x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 4097.214 | 4715.919 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 754.838 | 3843.836 | 4637.284 | 5.092x | 6.143x |

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
