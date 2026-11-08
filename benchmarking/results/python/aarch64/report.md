# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-08T08:13:31.682413+00:00
- Commit: `55ab392`
- CPU: aarch64 (aarch64)
- OS: Linux-7.0.0-1017-azure-aarch64-with-glibc2.39
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.4.2
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.622 | 0.792 | 1.442 | 1.273x | 2.318x |
| FP32 | ADD | allocating | 7 | NEON | 0.616 | 0.791 | 1.386 | 1.285x | 2.250x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 470.031 | 498.695 | 272.032 | 1.061x | 0.579x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 392.585 | 559.904 | 290.021 | 1.426x | 0.739x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.235 | 0.918 | 1.284 | 3.899x | 5.456x |
| FP32 | ADD | inplace | 7 | NEON | 0.256 | 0.916 | 1.225 | 3.571x | 4.776x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 348.506 | 223.633 | 226.289 | 0.642x | 0.649x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 342.600 | 229.999 | 224.347 | 0.671x | 0.655x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.640 | 0.780 | 1.373 | 1.220x | 2.146x |
| FP32 | SUB | allocating | 7 | NEON | 0.606 | 0.779 | 1.374 | 1.287x | 2.269x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 458.925 | 497.541 | 267.503 | 1.084x | 0.583x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 435.736 | 480.496 | 260.037 | 1.103x | 0.597x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.229 | 0.916 | 1.540 | 4.005x | 6.737x |
| FP32 | SUB | inplace | 7 | NEON | 0.261 | 0.917 | 1.569 | 3.519x | 6.018x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 341.768 | 210.142 | 227.987 | 0.615x | 0.667x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 339.458 | 216.099 | 220.007 | 0.637x | 0.648x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.631 | 0.816 | 1.367 | 1.294x | 2.168x |
| FP32 | MUL | allocating | 7 | NEON | 0.603 | 0.815 | 1.363 | 1.351x | 2.260x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 461.821 | 504.544 | 276.474 | 1.093x | 0.599x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 395.393 | 498.377 | 253.309 | 1.260x | 0.641x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.236 | 0.955 | 1.212 | 4.045x | 5.137x |
| FP32 | MUL | inplace | 7 | NEON | 0.270 | 0.949 | 1.237 | 3.513x | 4.578x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 344.076 | 225.669 | 222.471 | 0.656x | 0.647x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 341.964 | 224.099 | 221.361 | 0.655x | 0.647x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.635 | 0.767 | 1.366 | 1.208x | 2.150x |
| FP32 | DIV | allocating | 7 | NEON | 0.613 | 0.770 | 1.386 | 1.255x | 2.260x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 687.412 | 753.035 | 487.532 | 1.095x | 0.709x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 475.593 | 703.919 | 490.636 | 1.480x | 1.032x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.240 | 0.911 | 1.603 | 3.787x | 6.666x |
| FP32 | DIV | inplace | 7 | NEON | 0.260 | 0.904 | 1.534 | 3.477x | 5.896x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 463.509 | 466.289 | 480.736 | 1.006x | 1.037x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 469.941 | 464.781 | 477.018 | 0.989x | 1.015x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.626 | 6.327 | 8.941 | 10.111x | 14.289x |
| INT8 | ADD | allocating | 7 | NEON | 0.693 | 6.368 | 9.200 | 9.192x | 13.280x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1018.710 | 556.580 | 597.124 | 0.546x | 0.586x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 118.625 | 552.015 | 592.958 | 4.653x | 4.999x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.236 | 6.452 | 6.931 | 27.325x | 29.350x |
| INT8 | ADD | inplace | 7 | NEON | 0.294 | 6.329 | 6.750 | 21.497x | 22.927x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 957.097 | 541.488 | 586.945 | 0.566x | 0.613x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 90.690 | 535.430 | 581.389 | 5.904x | 6.411x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.639 | 6.348 | 9.660 | 9.939x | 15.125x |
| INT8 | SUB | allocating | 7 | NEON | 0.692 | 6.327 | 11.335 | 9.146x | 16.384x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1018.175 | 555.341 | 598.434 | 0.545x | 0.588x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 118.037 | 553.337 | 596.416 | 4.688x | 5.053x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.248 | 6.449 | 8.977 | 25.956x | 36.129x |
| INT8 | SUB | inplace | 7 | NEON | 0.267 | 6.476 | 7.613 | 24.238x | 28.492x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 956.928 | 548.342 | 595.645 | 0.573x | 0.622x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 90.585 | 550.956 | 601.985 | 6.082x | 6.646x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.634 | 6.470 | 7.914 | 10.204x | 12.483x |
| INT8 | MUL | allocating | 7 | NEON | 0.687 | 6.338 | 7.849 | 9.229x | 11.430x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 971.488 | 563.552 | 595.208 | 0.580x | 0.613x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 212.558 | 562.657 | 594.593 | 2.647x | 2.797x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.233 | 6.327 | 6.791 | 27.195x | 29.192x |
| INT8 | MUL | inplace | 7 | NEON | 0.272 | 6.327 | 6.709 | 23.222x | 24.622x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 911.735 | 560.714 | 660.872 | 0.615x | 0.725x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 94.651 | 548.698 | 585.796 | 5.797x | 6.189x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.649 | 6.974 | 9.789 | 10.749x | 15.088x |
| INT8 | DIV | allocating | 7 | NEON | 0.687 | 6.986 | 9.250 | 10.171x | 13.467x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1679.353 | 2794.614 | 2071.343 | 1.664x | 1.233x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 569.809 | 2782.964 | 2063.365 | 4.884x | 3.621x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.244 | 7.021 | 9.112 | 28.740x | 37.300x |
| INT8 | DIV | inplace | 7 | NEON | 0.266 | 7.000 | 7.198 | 26.311x | 27.056x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1617.857 | 2784.690 | 2052.078 | 1.721x | 1.268x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 502.711 | 2737.437 | 2059.836 | 5.445x | 4.097x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.803 | 1.416 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.682 | 0.799 | 1.360 | 1.173x | 1.995x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1949.645 | 143.378 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1008.323 | 1956.380 | 142.888 | 1.940x | 0.142x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.930 | 1.232 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.267 | 0.929 | 1.224 | 3.472x | 4.577x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1885.578 | 118.977 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 897.008 | 1882.514 | 117.857 | 2.099x | 0.131x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.809 | 1.372 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.729 | 0.803 | 1.429 | 1.102x | 1.961x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1978.486 | 145.822 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1012.204 | 1984.461 | 143.076 | 1.961x | 0.141x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.930 | 1.560 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.264 | 0.933 | 1.584 | 3.528x | 5.992x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1901.627 | 122.889 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 897.887 | 1900.728 | 120.072 | 2.117x | 0.134x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.800 | 1.361 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.689 | 0.796 | 1.414 | 1.156x | 2.053x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1971.875 | 137.902 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1012.455 | 1981.359 | 139.679 | 1.957x | 0.138x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.925 | 1.206 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.263 | 0.927 | 1.218 | 3.519x | 4.626x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1899.725 | 119.077 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 897.620 | 1901.157 | 118.354 | 2.118x | 0.132x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.788 | 1.370 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.689 | 0.791 | 1.365 | 1.148x | 1.980x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2178.724 | 361.838 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1309.247 | 2178.818 | 358.824 | 1.664x | 0.274x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.920 | 1.622 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.260 | 0.914 | 1.542 | 3.514x | 5.933x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2103.442 | 359.037 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1195.861 | 2099.625 | 351.620 | 1.756x | 0.294x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.259 | 8.348 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.685 | 6.242 | 8.191 | 9.116x | 11.962x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1273.373 | 1362.268 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 249.396 | 1311.301 | 1374.865 | 5.258x | 5.513x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.228 | 6.901 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.287 | 6.258 | 6.553 | 21.803x | 22.832x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1218.281 | 1376.884 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 184.650 | 1234.759 | 1388.757 | 6.687x | 7.521x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.208 | 8.870 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.684 | 6.257 | 10.046 | 9.144x | 14.681x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1303.513 | 1374.928 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 256.964 | 1325.054 | 1412.321 | 5.157x | 5.496x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.242 | 7.204 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.261 | 6.266 | 7.364 | 24.041x | 28.253x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1245.895 | 1416.105 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 183.048 | 1255.875 | 1373.645 | 6.861x | 7.504x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.207 | 8.120 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.684 | 6.258 | 8.154 | 9.154x | 11.927x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1392.638 | 1333.837 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 259.231 | 1379.961 | 1347.487 | 5.323x | 5.198x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.244 | 6.350 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.263 | 6.260 | 6.312 | 23.799x | 23.997x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1318.226 | 1361.210 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 184.139 | 1315.628 | 1318.238 | 7.145x | 7.159x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.870 | 10.324 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.680 | 6.892 | 9.300 | 10.136x | 13.677x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3529.968 | 2655.943 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 616.369 | 3635.710 | 2644.701 | 5.899x | 4.291x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.919 | 8.179 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.268 | 6.903 | 8.864 | 25.760x | 33.076x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3671.089 | 2652.673 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 496.333 | 3616.264 | 2656.123 | 7.286x | 5.351x |

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
