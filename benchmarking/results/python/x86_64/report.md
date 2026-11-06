# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-06T06:10:37.427187+00:00
- Commit: `cbf1d9d`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.352 | 0.774 | 1.395 | 2.197x | 3.960x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.345 | 0.771 | 1.373 | 2.234x | 3.980x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 386.890 | 558.353 | 679.997 | 1.443x | 1.758x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 334.225 | 185.333 | 245.497 | 0.555x | 0.735x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.144 | 0.907 | 1.281 | 6.289x | 8.880x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.145 | 0.907 | 1.289 | 6.244x | 8.873x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 245.362 | 149.936 | 239.012 | 0.611x | 0.974x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 312.030 | 144.400 | 229.228 | 0.463x | 0.735x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.352 | 0.765 | 1.396 | 2.174x | 3.970x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.353 | 0.762 | 1.393 | 2.156x | 3.942x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 395.753 | 196.580 | 648.371 | 0.497x | 1.638x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 319.676 | 186.167 | 620.139 | 0.582x | 1.940x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.145 | 0.901 | 1.312 | 6.221x | 9.059x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.143 | 0.907 | 1.305 | 6.339x | 9.124x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 232.563 | 145.189 | 207.243 | 0.624x | 0.891x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 288.538 | 144.072 | 241.243 | 0.499x | 0.836x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.352 | 0.761 | 1.377 | 2.165x | 3.914x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.356 | 0.765 | 1.372 | 2.150x | 3.860x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 376.615 | 189.762 | 249.716 | 0.504x | 0.663x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 301.161 | 200.439 | 251.365 | 0.666x | 0.835x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.143 | 0.913 | 1.245 | 6.402x | 8.732x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.145 | 0.906 | 1.240 | 6.233x | 8.528x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 234.886 | 154.565 | 196.047 | 0.658x | 0.835x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 321.379 | 146.781 | 223.890 | 0.457x | 0.697x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.368 | 0.758 | 1.382 | 2.062x | 3.759x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.342 | 0.754 | 1.381 | 2.203x | 4.037x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 413.141 | 186.978 | 273.233 | 0.453x | 0.661x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 298.155 | 190.308 | 265.283 | 0.638x | 0.890x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.144 | 0.892 | 1.256 | 6.177x | 8.696x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.144 | 0.892 | 1.252 | 6.186x | 8.680x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 236.794 | 145.730 | 188.533 | 0.615x | 0.796x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 316.098 | 144.533 | 181.878 | 0.457x | 0.575x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.367 | 6.042 | 6.293 | 16.475x | 17.158x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.367 | 5.994 | 6.348 | 16.318x | 17.281x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 2888.205 | 330.071 | 298.027 | 0.114x | 0.103x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 77.724 | 327.254 | 302.036 | 4.210x | 3.886x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.149 | 6.093 | 5.108 | 40.832x | 34.228x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.150 | 6.053 | 5.037 | 40.401x | 33.622x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 2852.310 | 322.675 | 297.657 | 0.113x | 0.104x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 59.920 | 320.996 | 299.592 | 5.357x | 5.000x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.363 | 5.985 | 6.221 | 16.482x | 17.131x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.374 | 5.969 | 6.327 | 15.953x | 16.909x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 2678.141 | 326.287 | 290.767 | 0.122x | 0.109x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 77.921 | 322.931 | 297.197 | 4.144x | 3.814x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.150 | 6.101 | 5.088 | 40.719x | 33.960x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.154 | 6.046 | 5.156 | 39.299x | 33.518x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 2645.978 | 316.725 | 297.515 | 0.120x | 0.112x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 57.754 | 333.525 | 298.208 | 5.775x | 5.163x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.359 | 5.975 | 6.102 | 16.665x | 17.019x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.378 | 5.978 | 6.147 | 15.810x | 16.257x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4306.817 | 326.026 | 287.463 | 0.076x | 0.067x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 87.046 | 332.487 | 292.661 | 3.820x | 3.362x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.148 | 6.118 | 5.017 | 41.339x | 33.901x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.151 | 6.041 | 5.019 | 40.002x | 33.233x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4267.418 | 322.655 | 296.196 | 0.076x | 0.069x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 61.130 | 333.276 | 297.828 | 5.452x | 4.872x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.375 | 6.696 | 6.170 | 17.838x | 16.438x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.379 | 6.704 | 6.214 | 17.708x | 16.414x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1683.724 | 1182.797 | 1849.193 | 0.702x | 1.098x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 585.171 | 1164.401 | 1848.754 | 1.990x | 3.159x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.151 | 6.834 | 4.995 | 45.308x | 33.117x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.152 | 6.795 | 5.005 | 44.757x | 32.964x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1644.449 | 1217.665 | 1846.320 | 0.740x | 1.123x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 546.510 | 1175.653 | 1845.034 | 2.151x | 3.376x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.802 | 1.379 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.371 | 0.805 | 1.389 | 2.168x | 3.742x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 5475.904 | 155.906 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 186.360 | 5487.589 | 162.061 | 29.446x | 0.870x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.927 | 1.286 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.150 | 0.931 | 1.294 | 6.189x | 8.605x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 5471.782 | 141.478 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 119.208 | 5473.815 | 140.608 | 45.918x | 1.180x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.803 | 1.396 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.374 | 0.797 | 1.405 | 2.132x | 3.760x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 5754.037 | 159.848 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 186.576 | 5746.090 | 162.834 | 30.798x | 0.873x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.921 | 1.309 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.146 | 0.923 | 1.307 | 6.305x | 8.926x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 5738.570 | 141.750 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 118.880 | 5740.932 | 141.699 | 48.292x | 1.192x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.807 | 1.366 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.372 | 0.812 | 1.372 | 2.186x | 3.694x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 5473.526 | 153.176 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 182.892 | 5481.201 | 159.526 | 29.970x | 0.872x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.930 | 1.247 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.150 | 0.929 | 1.242 | 6.185x | 8.268x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 5470.282 | 124.639 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 117.917 | 5467.733 | 125.465 | 46.369x | 1.064x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.780 | 1.372 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.376 | 0.781 | 1.380 | 2.076x | 3.666x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 5476.259 | 152.968 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 208.899 | 5474.220 | 158.320 | 26.205x | 0.758x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.914 | 1.240 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.148 | 0.910 | 1.241 | 6.129x | 8.362x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 5468.874 | 130.723 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 124.984 | 5471.180 | 129.123 | 43.775x | 1.033x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 5.975 | 6.274 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.395 | 6.012 | 6.308 | 15.238x | 15.986x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 808.889 | 894.833 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 160.933 | 770.549 | 938.786 | 4.788x | 5.833x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.004 | 5.046 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.149 | 5.969 | 5.026 | 40.000x | 33.685x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 797.430 | 920.840 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 115.689 | 811.683 | 908.350 | 7.016x | 7.852x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 5.911 | 6.310 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.368 | 5.893 | 6.220 | 16.007x | 16.897x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 763.791 | 1023.827 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 166.319 | 770.880 | 1063.194 | 4.635x | 6.393x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 5.982 | 5.082 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.147 | 5.986 | 5.115 | 40.662x | 34.744x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 786.283 | 898.852 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 117.636 | 849.751 | 916.835 | 7.224x | 7.794x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 5.883 | 6.162 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.370 | 6.109 | 6.117 | 16.503x | 16.526x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 822.630 | 937.735 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 180.686 | 895.926 | 929.060 | 4.958x | 5.142x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 5.985 | 5.006 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.149 | 5.985 | 5.030 | 40.109x | 33.711x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 842.381 | 937.339 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 120.148 | 852.662 | 927.565 | 7.097x | 7.720x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.655 | 6.262 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.385 | 6.650 | 6.167 | 17.274x | 16.019x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 2348.806 | 2328.949 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 598.476 | 2616.970 | 2336.617 | 4.373x | 3.904x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.719 | 4.950 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.153 | 6.739 | 4.959 | 44.152x | 32.491x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 2365.349 | 2328.603 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 511.471 | 2358.746 | 2333.126 | 4.612x | 4.562x |

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
