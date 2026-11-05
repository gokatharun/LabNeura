# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-05T21:38:04.082852+00:00
- Commit: `659f17b`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.427 | 1.017 | 1.816 | 2.378x | 4.249x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.440 | 1.020 | 1.824 | 2.317x | 4.144x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 395.210 | 231.912 | 287.163 | 0.587x | 0.727x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 379.750 | 223.667 | 287.092 | 0.589x | 0.756x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.188 | 1.145 | 1.672 | 6.092x | 8.901x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.188 | 1.145 | 1.670 | 6.096x | 8.889x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 239.328 | 173.742 | 247.175 | 0.726x | 1.033x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 312.606 | 161.195 | 218.722 | 0.516x | 0.700x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.443 | 1.026 | 1.836 | 2.316x | 4.145x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.440 | 1.019 | 1.839 | 2.315x | 4.179x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 388.360 | 225.855 | 274.090 | 0.582x | 0.706x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 424.186 | 221.474 | 285.592 | 0.522x | 0.673x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.187 | 1.154 | 1.680 | 6.163x | 8.972x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.191 | 1.160 | 1.695 | 6.080x | 8.890x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 235.159 | 158.857 | 227.711 | 0.676x | 0.968x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 311.780 | 156.556 | 230.625 | 0.502x | 0.740x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.432 | 1.016 | 1.807 | 2.348x | 4.178x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.448 | 1.017 | 1.816 | 2.270x | 4.052x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 406.618 | 225.282 | 286.577 | 0.554x | 0.705x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 415.393 | 224.776 | 276.316 | 0.541x | 0.665x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.188 | 1.162 | 1.625 | 6.184x | 8.649x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.190 | 1.162 | 1.627 | 6.100x | 8.543x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 237.911 | 150.574 | 224.233 | 0.633x | 0.943x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 306.474 | 161.378 | 220.480 | 0.527x | 0.719x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.434 | 0.997 | 1.816 | 2.296x | 4.183x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.447 | 1.007 | 1.817 | 2.251x | 4.061x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 428.603 | 227.318 | 278.244 | 0.530x | 0.649x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 462.462 | 221.246 | 717.694 | 0.478x | 1.552x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.189 | 1.139 | 1.631 | 6.036x | 8.643x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.190 | 1.141 | 1.613 | 6.016x | 8.505x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 241.955 | 163.479 | 220.197 | 0.676x | 0.910x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 323.101 | 173.792 | 235.640 | 0.538x | 0.729x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.436 | 7.895 | 8.146 | 18.089x | 18.663x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.448 | 7.926 | 8.053 | 17.689x | 17.974x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 3455.461 | 418.309 | 384.292 | 0.121x | 0.111x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 81.850 | 415.802 | 378.471 | 5.080x | 4.624x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.194 | 7.975 | 6.699 | 41.043x | 34.475x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.201 | 7.942 | 6.551 | 39.533x | 32.607x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 3413.789 | 420.160 | 382.137 | 0.123x | 0.112x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 59.225 | 420.299 | 381.499 | 7.097x | 6.442x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.449 | 7.887 | 8.014 | 17.579x | 17.862x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.448 | 7.819 | 8.161 | 17.459x | 18.223x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 3700.844 | 414.183 | 380.926 | 0.112x | 0.103x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 82.471 | 413.251 | 391.257 | 5.011x | 4.744x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.198 | 7.959 | 6.622 | 40.257x | 33.497x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.198 | 7.936 | 6.612 | 40.184x | 33.476x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 3670.617 | 421.094 | 381.539 | 0.115x | 0.104x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 59.073 | 408.744 | 380.675 | 6.919x | 6.444x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.444 | 7.912 | 8.071 | 17.834x | 18.191x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.443 | 7.971 | 8.209 | 17.990x | 18.528x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 5589.974 | 414.179 | 366.458 | 0.074x | 0.066x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 96.353 | 420.246 | 366.319 | 4.362x | 3.802x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.195 | 7.905 | 6.556 | 40.605x | 33.675x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.198 | 7.998 | 6.557 | 40.373x | 33.100x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 5549.455 | 425.299 | 369.300 | 0.077x | 0.067x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 62.009 | 425.691 | 371.522 | 6.865x | 5.991x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.436 | 8.927 | 8.026 | 20.482x | 18.414x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.458 | 8.801 | 7.974 | 19.230x | 17.422x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 2164.764 | 1593.976 | 2378.575 | 0.736x | 1.099x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 737.488 | 1628.636 | 2385.390 | 2.208x | 3.234x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.200 | 9.049 | 6.632 | 45.225x | 33.144x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.201 | 8.938 | 6.475 | 44.402x | 32.167x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 2120.884 | 1614.437 | 2379.441 | 0.761x | 1.122x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 695.207 | 1989.728 | 2386.912 | 2.862x | 3.433x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 1.072 | 1.854 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.477 | 1.074 | 1.851 | 2.252x | 3.884x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 7188.662 | 305.310 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 199.402 | 7066.299 | 189.387 | 35.438x | 0.950x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 1.199 | 1.669 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.199 | 1.197 | 1.665 | 6.005x | 8.355x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 7061.980 | 182.605 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 124.523 | 7056.358 | 170.965 | 56.667x | 1.373x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 1.055 | 1.841 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.468 | 1.058 | 1.836 | 2.260x | 3.921x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 7511.458 | 310.757 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 196.474 | 7411.775 | 188.460 | 37.724x | 0.959x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 1.212 | 1.700 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.198 | 1.194 | 1.708 | 6.017x | 8.604x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 7424.091 | 180.422 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 123.291 | 7410.081 | 175.249 | 60.102x | 1.421x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 1.071 | 1.817 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.463 | 1.069 | 1.809 | 2.307x | 3.904x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 7070.572 | 175.746 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 186.820 | 7081.028 | 180.481 | 37.903x | 0.966x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 1.190 | 1.631 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.200 | 1.199 | 1.642 | 5.996x | 8.212x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 7059.142 | 157.278 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 124.280 | 7060.871 | 150.606 | 56.814x | 1.212x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 1.060 | 1.822 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.463 | 1.054 | 1.828 | 2.276x | 3.948x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 7196.096 | 307.578 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 229.283 | 7167.029 | 305.249 | 31.258x | 1.331x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 1.192 | 1.619 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.198 | 1.184 | 1.615 | 5.995x | 8.178x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 7089.184 | 171.686 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 145.274 | 7058.222 | 162.805 | 48.586x | 1.121x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 7.931 | 8.209 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.461 | 7.729 | 8.160 | 16.763x | 17.697x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1181.208 | 1291.922 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 171.377 | 1060.322 | 1146.806 | 6.187x | 6.692x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 7.908 | 6.636 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.198 | 7.922 | 6.484 | 39.992x | 32.731x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1081.421 | 1165.039 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 124.173 | 1117.241 | 1142.438 | 8.997x | 9.200x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 7.905 | 7.969 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.459 | 7.831 | 8.140 | 17.043x | 17.716x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1168.927 | 1266.250 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 174.246 | 1063.071 | 1137.169 | 6.101x | 6.526x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 7.852 | 6.559 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.197 | 7.918 | 6.599 | 40.213x | 33.513x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1076.095 | 1140.015 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 122.939 | 1029.806 | 1143.894 | 8.377x | 9.305x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 7.832 | 8.039 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.458 | 7.731 | 8.029 | 16.876x | 17.528x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1200.333 | 1301.082 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 196.460 | 1074.256 | 1180.531 | 5.468x | 6.009x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 7.923 | 6.519 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.199 | 7.918 | 6.561 | 39.769x | 32.956x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1141.446 | 1180.275 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 128.373 | 1050.725 | 1187.729 | 8.185x | 9.252x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 8.951 | 7.970 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.461 | 8.936 | 7.973 | 19.397x | 17.307x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3145.828 | 3184.715 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 729.443 | 2991.759 | 3031.103 | 4.101x | 4.155x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 8.970 | 6.374 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.200 | 8.948 | 6.586 | 44.719x | 32.913x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3141.771 | 3050.962 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 642.082 | 2996.204 | 3006.155 | 4.666x | 4.682x |

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
