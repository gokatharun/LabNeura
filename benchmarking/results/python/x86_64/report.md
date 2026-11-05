# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-05T20:26:00.396824+00:00
- Commit: `b91df83`
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.327 | 0.788 | 1.377 | 2.410x | 4.214x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.351 | 0.783 | 1.366 | 2.230x | 3.893x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 369.920 | 648.269 | 712.765 | 1.752x | 1.927x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 392.351 | 237.352 | 296.901 | 0.605x | 0.757x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.150 | 0.903 | 1.260 | 6.013x | 8.387x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.152 | 0.894 | 1.258 | 5.869x | 8.266x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 276.213 | 178.377 | 237.212 | 0.646x | 0.859x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 327.920 | 185.370 | 263.324 | 0.565x | 0.803x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.336 | 0.785 | 1.382 | 2.336x | 4.114x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.353 | 0.789 | 1.495 | 2.237x | 4.238x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 383.774 | 292.445 | 308.899 | 0.762x | 0.805x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 390.791 | 253.386 | 321.052 | 0.648x | 0.822x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.145 | 0.900 | 1.297 | 6.189x | 8.921x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.151 | 0.911 | 1.295 | 6.018x | 8.552x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 243.822 | 172.192 | 240.529 | 0.706x | 0.986x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 315.902 | 177.675 | 252.387 | 0.562x | 0.799x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.328 | 0.783 | 1.360 | 2.384x | 4.143x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.345 | 0.783 | 1.349 | 2.266x | 3.906x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 373.613 | 237.925 | 316.495 | 0.637x | 0.847x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 419.589 | 288.085 | 311.934 | 0.687x | 0.743x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.151 | 0.915 | 1.208 | 6.052x | 7.988x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.155 | 0.921 | 1.216 | 5.946x | 7.853x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 256.310 | 179.663 | 252.367 | 0.701x | 0.985x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 328.899 | 178.034 | 239.498 | 0.541x | 0.728x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.361 | 0.767 | 1.356 | 2.125x | 3.758x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.335 | 0.772 | 1.372 | 2.303x | 4.091x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 413.088 | 241.015 | 333.524 | 0.583x | 0.807x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 416.873 | 235.459 | 285.822 | 0.565x | 0.686x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.148 | 0.902 | 1.226 | 6.102x | 8.295x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.150 | 0.896 | 1.243 | 5.969x | 8.288x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 249.123 | 187.185 | 254.368 | 0.751x | 1.021x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 336.460 | 183.119 | 244.172 | 0.544x | 0.726x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.368 | 5.979 | 6.197 | 16.250x | 16.841x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.348 | 6.167 | 6.357 | 17.736x | 18.283x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 2685.231 | 321.290 | 308.517 | 0.120x | 0.115x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 78.401 | 324.192 | 308.975 | 4.135x | 3.941x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.158 | 6.038 | 4.991 | 38.207x | 31.581x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.156 | 6.120 | 5.004 | 39.276x | 32.114x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 2653.222 | 325.228 | 313.672 | 0.123x | 0.118x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 62.568 | 332.193 | 306.331 | 5.309x | 4.896x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.346 | 6.057 | 6.238 | 17.501x | 18.025x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.346 | 6.214 | 6.330 | 17.973x | 18.310x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 2851.259 | 331.243 | 301.880 | 0.116x | 0.106x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 80.470 | 326.264 | 306.692 | 4.054x | 3.811x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.157 | 6.110 | 5.169 | 38.811x | 32.834x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.160 | 6.073 | 5.218 | 37.950x | 32.603x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 2845.585 | 337.330 | 311.055 | 0.119x | 0.109x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 59.560 | 339.249 | 299.013 | 5.696x | 5.020x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.335 | 6.028 | 6.174 | 18.011x | 18.447x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.349 | 5.994 | 6.148 | 17.198x | 17.641x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4344.290 | 363.930 | 323.465 | 0.084x | 0.074x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 87.922 | 330.414 | 306.819 | 3.758x | 3.490x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.152 | 6.138 | 5.024 | 40.319x | 32.999x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.156 | 6.070 | 4.992 | 38.901x | 31.991x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4304.009 | 330.465 | 291.231 | 0.077x | 0.068x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 62.662 | 327.837 | 305.539 | 5.232x | 4.876x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.337 | 6.799 | 6.140 | 20.192x | 18.235x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.357 | 6.767 | 6.308 | 18.937x | 17.654x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1684.606 | 1631.381 | 1859.465 | 0.968x | 1.104x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 586.052 | 1522.416 | 1876.143 | 2.598x | 3.201x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.157 | 6.778 | 5.010 | 43.109x | 31.866x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.163 | 6.844 | 5.064 | 41.874x | 30.982x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1645.481 | 1625.587 | 1851.799 | 0.988x | 1.125x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 547.784 | 1564.436 | 1851.441 | 2.856x | 3.380x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.809 | 1.365 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.365 | 0.813 | 1.375 | 2.229x | 3.768x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 5567.088 | 281.504 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 191.297 | 5481.317 | 175.773 | 28.653x | 0.919x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.926 | 1.256 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.154 | 0.930 | 1.264 | 6.024x | 8.185x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 5482.054 | 150.148 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 123.049 | 5475.049 | 142.915 | 44.495x | 1.161x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.813 | 1.389 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.350 | 0.809 | 1.379 | 2.311x | 3.940x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 5833.802 | 279.964 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 203.221 | 5760.543 | 176.660 | 28.346x | 0.869x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.923 | 1.288 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.153 | 0.917 | 1.296 | 6.002x | 8.480x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 5752.784 | 148.890 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 124.119 | 5767.066 | 142.815 | 46.464x | 1.151x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.809 | 1.377 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.359 | 0.811 | 1.347 | 2.260x | 3.753x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 5567.315 | 271.607 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 188.266 | 5485.574 | 163.122 | 29.137x | 0.866x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.928 | 1.215 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.154 | 0.930 | 1.218 | 6.031x | 7.896x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 5481.671 | 137.174 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 122.405 | 5470.648 | 129.204 | 44.693x | 1.056x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.795 | 1.362 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.355 | 0.795 | 1.362 | 2.239x | 3.836x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 5572.070 | 281.990 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 212.296 | 5477.351 | 168.971 | 25.801x | 0.796x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.912 | 1.224 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.156 | 0.910 | 1.218 | 5.849x | 7.828x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 5485.335 | 143.675 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 128.571 | 5472.648 | 136.018 | 42.565x | 1.058x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 5.990 | 6.289 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.361 | 5.967 | 6.252 | 16.541x | 17.332x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 952.085 | 1040.307 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 164.636 | 835.638 | 907.682 | 5.076x | 5.513x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.047 | 4.976 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.170 | 6.070 | 4.997 | 35.779x | 29.451x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 868.284 | 946.994 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 118.230 | 861.312 | 906.977 | 7.285x | 7.671x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 5.992 | 6.318 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.352 | 5.926 | 6.160 | 16.819x | 17.484x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 913.191 | 1051.631 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 165.325 | 869.349 | 977.965 | 5.258x | 5.915x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.033 | 5.077 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.155 | 6.032 | 5.059 | 38.810x | 32.550x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 854.381 | 920.729 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 121.298 | 824.658 | 1197.788 | 6.799x | 9.875x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 5.951 | 6.094 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.352 | 6.113 | 6.135 | 17.360x | 17.422x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1032.996 | 1116.538 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 187.295 | 869.590 | 958.798 | 4.643x | 5.119x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.041 | 4.962 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.156 | 6.142 | 4.994 | 39.269x | 31.929x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1000.229 | 954.906 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 130.078 | 883.710 | 937.592 | 6.794x | 7.208x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.695 | 6.091 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.354 | 6.736 | 6.208 | 19.044x | 17.550x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 2559.696 | 2648.043 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 596.917 | 2581.664 | 2416.501 | 4.325x | 4.048x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.770 | 4.939 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.159 | 6.742 | 4.889 | 42.446x | 30.776x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 2675.416 | 2404.368 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 513.069 | 2565.627 | 2401.236 | 5.001x | 4.680x |

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
