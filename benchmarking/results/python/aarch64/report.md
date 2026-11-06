# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-06T06:10:13.031452+00:00
- Commit: `cbf1d9d`
- CPU: aarch64 (aarch64)
- OS: Linux-7.0.0-1017-azure-aarch64-with-glibc2.39
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.4.1
- Actual SIMD backend selected: **NEON**
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.509 | 0.785 | 1.390 | 1.543x | 2.732x |
| FP32 | ADD | allocating | 7 | NEON | 0.493 | 0.780 | 1.367 | 1.583x | 2.772x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 466.067 | 209.725 | 237.875 | 0.450x | 0.510x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 410.458 | 204.597 | 233.015 | 0.498x | 0.568x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.215 | 0.908 | 1.231 | 4.231x | 5.738x |
| FP32 | ADD | inplace | 7 | NEON | 0.211 | 0.904 | 1.232 | 4.295x | 5.852x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 345.064 | 177.423 | 208.615 | 0.514x | 0.605x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 351.822 | 181.373 | 218.801 | 0.516x | 0.622x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.498 | 0.778 | 1.404 | 1.561x | 2.818x |
| FP32 | SUB | allocating | 7 | NEON | 0.494 | 0.778 | 1.371 | 1.575x | 2.775x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 462.173 | 208.012 | 322.135 | 0.450x | 0.697x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 411.878 | 460.145 | 298.158 | 1.117x | 0.724x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.212 | 0.921 | 1.274 | 4.336x | 5.996x |
| FP32 | SUB | inplace | 7 | NEON | 0.216 | 0.912 | 1.260 | 4.222x | 5.834x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 342.082 | 176.494 | 215.467 | 0.516x | 0.630x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 345.486 | 191.065 | 231.120 | 0.553x | 0.669x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.520 | 0.813 | 1.364 | 1.563x | 2.624x |
| FP32 | MUL | allocating | 7 | NEON | 0.491 | 0.814 | 1.363 | 1.658x | 2.774x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 463.902 | 208.401 | 229.787 | 0.449x | 0.495x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 406.292 | 211.330 | 233.504 | 0.520x | 0.575x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.207 | 0.947 | 1.185 | 4.587x | 5.736x |
| FP32 | MUL | inplace | 7 | NEON | 0.211 | 0.951 | 1.183 | 4.503x | 5.601x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 345.713 | 178.053 | 197.462 | 0.515x | 0.571x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 344.382 | 177.976 | 205.178 | 0.517x | 0.596x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.526 | 0.765 | 1.392 | 1.456x | 2.647x |
| FP32 | DIV | allocating | 7 | NEON | 0.507 | 0.762 | 1.357 | 1.503x | 2.674x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 678.243 | 458.161 | 473.704 | 0.676x | 0.698x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 472.160 | 673.255 | 486.537 | 1.426x | 1.030x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.213 | 0.902 | 1.564 | 4.237x | 7.346x |
| FP32 | DIV | inplace | 7 | NEON | 0.213 | 0.903 | 1.587 | 4.232x | 7.438x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 462.970 | 452.273 | 464.042 | 0.977x | 1.002x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 470.226 | 458.143 | 472.288 | 0.974x | 1.004x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.513 | 6.304 | 11.439 | 12.290x | 22.302x |
| INT8 | ADD | allocating | 7 | NEON | 0.528 | 6.275 | 8.977 | 11.881x | 16.997x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1013.404 | 523.652 | 586.861 | 0.517x | 0.579x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 115.888 | 540.119 | 594.684 | 4.661x | 5.132x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.216 | 6.291 | 7.291 | 29.126x | 33.753x |
| INT8 | ADD | inplace | 7 | NEON | 0.219 | 6.320 | 7.205 | 28.879x | 32.919x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 957.357 | 522.938 | 572.483 | 0.546x | 0.598x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 90.069 | 523.127 | 577.086 | 5.808x | 6.407x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.517 | 6.344 | 9.770 | 12.272x | 18.900x |
| INT8 | SUB | allocating | 7 | NEON | 0.518 | 6.332 | 9.076 | 12.227x | 17.524x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1013.846 | 522.283 | 586.031 | 0.515x | 0.578x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 111.810 | 536.637 | 591.797 | 4.800x | 5.293x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.217 | 6.272 | 7.729 | 28.929x | 35.650x |
| INT8 | SUB | inplace | 7 | NEON | 0.217 | 6.303 | 7.913 | 28.989x | 36.396x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 956.991 | 524.809 | 571.693 | 0.548x | 0.597x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 86.999 | 523.527 | 587.251 | 6.018x | 6.750x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.514 | 6.262 | 8.624 | 12.186x | 16.781x |
| INT8 | MUL | allocating | 7 | NEON | 0.520 | 6.258 | 7.829 | 12.031x | 15.052x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 968.721 | 522.810 | 571.376 | 0.540x | 0.590x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 123.768 | 545.615 | 582.686 | 4.408x | 4.708x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.216 | 6.253 | 6.148 | 28.972x | 28.487x |
| INT8 | MUL | inplace | 7 | NEON | 0.212 | 6.272 | 6.357 | 29.653x | 30.053x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 910.705 | 520.678 | 562.035 | 0.572x | 0.617x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 87.930 | 530.343 | 574.842 | 6.031x | 6.537x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.508 | 7.049 | 9.206 | 13.881x | 18.129x |
| INT8 | DIV | allocating | 7 | NEON | 0.527 | 6.879 | 9.312 | 13.062x | 17.680x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1677.803 | 2691.805 | 2052.430 | 1.604x | 1.223x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 554.030 | 2730.332 | 2061.566 | 4.928x | 3.721x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.219 | 6.964 | 7.800 | 31.844x | 35.666x |
| INT8 | DIV | inplace | 7 | NEON | 0.213 | 6.945 | 7.021 | 32.641x | 32.996x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1619.505 | 2692.033 | 2043.255 | 1.662x | 1.262x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 511.927 | 2720.104 | 2042.083 | 5.313x | 3.989x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.803 | 1.361 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.512 | 0.799 | 1.359 | 1.559x | 2.652x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1939.855 | 124.650 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1003.763 | 1951.577 | 125.352 | 1.944x | 0.125x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.922 | 1.205 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.219 | 0.928 | 1.254 | 4.239x | 5.723x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1881.078 | 107.952 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 896.100 | 1883.782 | 112.014 | 2.102x | 0.125x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.804 | 1.422 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.520 | 0.813 | 1.397 | 1.565x | 2.689x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1974.539 | 133.436 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1007.286 | 1900.029 | 143.381 | 1.886x | 0.142x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.915 | 1.334 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.213 | 0.920 | 1.249 | 4.325x | 5.875x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1899.753 | 109.852 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 895.415 | 1913.009 | 113.211 | 2.136x | 0.126x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.799 | 1.372 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.519 | 0.802 | 1.361 | 1.545x | 2.621x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1980.137 | 131.398 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1003.467 | 1903.221 | 127.690 | 1.897x | 0.127x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.915 | 1.211 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.212 | 0.915 | 1.193 | 4.324x | 5.640x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1894.134 | 119.785 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 897.057 | 1896.672 | 106.273 | 2.114x | 0.118x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.798 | 1.361 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.520 | 0.801 | 1.367 | 1.539x | 2.627x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2175.483 | 356.382 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1299.394 | 2176.476 | 356.278 | 1.675x | 0.274x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.920 | 1.551 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.215 | 0.912 | 1.674 | 4.249x | 7.797x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2099.469 | 348.804 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1193.903 | 2097.488 | 350.200 | 1.757x | 0.293x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.185 | 8.433 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.524 | 6.162 | 8.594 | 11.771x | 16.416x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1257.921 | 1343.288 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 234.963 | 1241.333 | 1373.848 | 5.283x | 5.847x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.155 | 6.994 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.212 | 6.160 | 8.487 | 29.101x | 40.096x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1170.740 | 1321.712 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 169.344 | 1168.854 | 1337.014 | 6.902x | 7.895x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.149 | 8.816 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.513 | 6.160 | 9.420 | 12.016x | 18.377x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1229.856 | 1338.033 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 235.739 | 1252.761 | 1373.911 | 5.314x | 5.828x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.170 | 7.190 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.210 | 6.168 | 7.507 | 29.361x | 35.738x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1179.393 | 1359.077 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 171.805 | 1171.322 | 1323.411 | 6.818x | 7.703x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.142 | 8.303 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.514 | 6.161 | 7.880 | 11.982x | 15.324x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1316.151 | 1284.894 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 239.234 | 1368.858 | 1287.768 | 5.722x | 5.383x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.160 | 6.308 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.213 | 6.192 | 6.367 | 29.077x | 29.901x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1274.418 | 1284.436 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 174.446 | 1284.336 | 1275.620 | 7.362x | 7.312x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.827 | 9.959 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.515 | 6.865 | 9.894 | 13.333x | 19.217x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3404.890 | 2635.994 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 607.272 | 3482.098 | 2639.594 | 5.734x | 4.347x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.851 | 9.106 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.212 | 6.852 | 7.916 | 32.270x | 37.282x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3453.944 | 2624.722 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 494.452 | 3425.427 | 2592.762 | 6.928x | 5.244x |

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
