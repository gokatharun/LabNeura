# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-20T16:59:56.604586+00:00
- Commit: `12feb95`
- CPU: AMD EPYC 7763 64-Core Processor (x86_64)
- OS: Linux-7.0.0-1020-azure-x86_64-with-glibc2.43
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.607 | 1.041 | 1.924 | 1.716x | 3.173x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.523 | 1.039 | 1.980 | 1.988x | 3.787x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 449.890 | 740.677 | 740.381 | 1.646x | 1.646x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 359.465 | 744.803 | 767.120 | 2.072x | 2.134x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.205 | 1.184 | 1.688 | 5.763x | 8.217x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.213 | 1.198 | 1.680 | 5.632x | 7.894x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 287.390 | 299.515 | 294.097 | 1.042x | 1.023x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 334.501 | 274.756 | 298.192 | 0.821x | 0.891x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.591 | 1.033 | 1.859 | 1.748x | 3.148x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.522 | 1.033 | 1.856 | 1.979x | 3.555x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 453.122 | 773.444 | 754.904 | 1.707x | 1.666x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 377.258 | 792.252 | 794.786 | 2.100x | 2.107x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.208 | 1.182 | 1.685 | 5.687x | 8.110x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.206 | 1.175 | 1.678 | 5.702x | 8.145x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 286.211 | 299.083 | 309.637 | 1.045x | 1.082x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 332.665 | 310.172 | 300.830 | 0.932x | 0.904x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.593 | 1.029 | 1.866 | 1.735x | 3.147x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.523 | 1.034 | 1.874 | 1.977x | 3.584x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 458.332 | 749.732 | 782.296 | 1.636x | 1.707x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 353.828 | 705.466 | 788.085 | 1.994x | 2.227x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.203 | 1.170 | 1.654 | 5.756x | 8.138x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.206 | 1.171 | 1.655 | 5.674x | 8.018x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 289.037 | 291.580 | 300.113 | 1.009x | 1.038x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 311.164 | 292.334 | 310.160 | 0.939x | 0.997x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.609 | 1.019 | 1.829 | 1.673x | 3.005x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.528 | 1.020 | 1.841 | 1.932x | 3.488x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 462.364 | 708.376 | 437.798 | 1.532x | 0.947x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 368.296 | 769.932 | 472.563 | 2.091x | 1.283x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.206 | 1.164 | 1.685 | 5.649x | 8.182x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.205 | 1.159 | 1.657 | 5.650x | 8.074x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 295.830 | 310.413 | 298.917 | 1.049x | 1.010x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 334.034 | 293.196 | 345.569 | 0.878x | 1.035x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.606 | 7.895 | 8.814 | 13.025x | 14.542x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.630 | 7.911 | 8.487 | 12.550x | 13.464x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 2912.308 | 528.358 | 468.537 | 0.181x | 0.161x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 105.305 | 496.275 | 423.347 | 4.713x | 4.020x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.220 | 8.262 | 6.701 | 37.553x | 30.458x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.221 | 8.164 | 6.705 | 36.977x | 30.367x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 2853.569 | 487.910 | 396.229 | 0.171x | 0.139x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 70.719 | 481.182 | 439.120 | 6.804x | 6.209x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.607 | 8.094 | 8.449 | 13.336x | 13.920x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.612 | 8.221 | 8.640 | 13.439x | 14.124x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 2912.032 | 510.374 | 466.168 | 0.175x | 0.160x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 104.598 | 492.465 | 462.270 | 4.708x | 4.419x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.228 | 8.311 | 6.844 | 36.415x | 29.986x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.212 | 8.127 | 6.692 | 38.407x | 31.624x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 2851.005 | 483.400 | 445.814 | 0.170x | 0.156x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 72.040 | 463.270 | 436.207 | 6.431x | 6.055x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.614 | 7.902 | 8.590 | 12.861x | 13.981x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.625 | 7.903 | 8.511 | 12.642x | 13.614x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4927.355 | 538.430 | 421.554 | 0.109x | 0.086x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 153.448 | 516.983 | 483.604 | 3.369x | 3.152x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.216 | 8.153 | 6.819 | 37.781x | 31.599x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.219 | 8.239 | 6.781 | 37.554x | 30.907x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4837.440 | 487.176 | 453.880 | 0.101x | 0.094x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 104.175 | 472.376 | 463.519 | 4.534x | 4.449x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.612 | 9.243 | 8.716 | 15.105x | 14.244x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.623 | 9.127 | 8.604 | 14.655x | 13.816x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1934.242 | 1536.407 | 2256.396 | 0.794x | 1.167x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 775.273 | 1533.610 | 2255.500 | 1.978x | 2.909x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.219 | 9.248 | 6.676 | 42.267x | 30.511x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.209 | 9.260 | 6.677 | 44.391x | 32.008x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1875.946 | 1557.252 | 2228.623 | 0.830x | 1.188x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 704.139 | 1533.160 | 2219.279 | 2.177x | 3.152x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 1.097 | 1.858 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.629 | 1.100 | 1.911 | 1.750x | 3.040x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 6687.222 | 355.176 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 242.302 | 6677.989 | 367.616 | 27.561x | 1.517x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 1.225 | 1.694 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.212 | 1.224 | 1.687 | 5.777x | 7.966x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 6595.916 | 243.740 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 161.163 | 6848.904 | 247.493 | 42.497x | 1.536x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 1.094 | 1.844 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.636 | 1.093 | 1.858 | 1.718x | 2.920x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 7179.154 | 400.343 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 252.023 | 6998.929 | 356.524 | 27.771x | 1.415x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 1.223 | 1.702 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.217 | 1.215 | 1.712 | 5.608x | 7.906x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 7570.190 | 253.678 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 168.514 | 6961.425 | 245.330 | 41.311x | 1.456x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 1.094 | 1.815 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.630 | 1.080 | 1.823 | 1.715x | 2.895x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 6696.323 | 331.758 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 254.388 | 6974.697 | 334.651 | 27.418x | 1.316x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 1.216 | 1.663 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.218 | 1.215 | 1.654 | 5.579x | 7.595x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 6588.835 | 223.524 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 170.547 | 6651.397 | 220.994 | 39.000x | 1.296x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 1.075 | 1.830 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.632 | 1.077 | 1.844 | 1.703x | 2.916x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 6823.570 | 309.194 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 271.643 | 6691.228 | 321.828 | 24.632x | 1.185x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 1.214 | 1.641 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.218 | 1.209 | 1.637 | 5.543x | 7.503x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 6570.532 | 203.453 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 172.954 | 6856.061 | 207.986 | 39.641x | 1.203x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 7.877 | 8.520 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.643 | 7.828 | 8.349 | 12.174x | 12.983x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1215.149 | 1467.445 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 230.500 | 1235.994 | 1475.775 | 5.362x | 6.402x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 8.055 | 6.649 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.213 | 7.937 | 6.757 | 37.299x | 31.755x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1128.770 | 1380.990 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 140.268 | 1159.758 | 1438.297 | 8.268x | 10.254x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 7.790 | 8.599 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.639 | 7.861 | 9.080 | 12.302x | 14.210x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1321.890 | 1519.812 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 226.094 | 1491.608 | 1557.682 | 6.597x | 6.890x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 7.928 | 6.887 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.221 | 8.037 | 6.667 | 36.402x | 30.192x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1126.437 | 1448.244 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 145.659 | 1108.786 | 1326.228 | 7.612x | 9.105x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 8.116 | 8.458 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.630 | 7.823 | 8.675 | 12.410x | 13.761x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1199.918 | 1494.090 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 299.770 | 1233.018 | 1458.659 | 4.113x | 4.866x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 7.902 | 6.708 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.221 | 7.940 | 6.794 | 35.923x | 30.739x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1117.768 | 1404.838 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 192.890 | 1136.732 | 1367.015 | 5.893x | 7.087x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 8.763 | 8.382 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.636 | 8.887 | 8.486 | 13.965x | 13.335x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 2878.850 | 2967.451 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 769.050 | 2819.189 | 2967.051 | 3.666x | 3.858x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 8.945 | 6.659 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.219 | 9.140 | 6.653 | 41.806x | 30.430x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 2804.944 | 2846.785 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 643.045 | 2813.458 | 2864.607 | 4.375x | 4.455x |

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
