# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-20T19:22:08.042103+00:00
- Commit: `32726d8`
- CPU: Intel(R) Xeon(R) 6975P-C (x86_64)
- OS: Linux-7.0.0-1016-aws-x86_64-with-glibc2.43
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.366 | 0.562 | 1.133 | 1.538x | 3.100x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.317 | 0.564 | 1.147 | 1.779x | 3.619x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 784.153 | 745.418 | 695.177 | 0.951x | 0.887x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 547.528 | 745.681 | 730.298 | 1.362x | 1.334x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.124 | 0.693 | 1.063 | 5.607x | 8.600x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.128 | 0.685 | 1.072 | 5.340x | 8.353x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 419.834 | 430.794 | 425.783 | 1.026x | 1.014x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 417.118 | 429.860 | 436.474 | 1.031x | 1.046x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.349 | 0.563 | 1.064 | 1.613x | 3.048x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.318 | 0.559 | 1.074 | 1.759x | 3.377x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 779.059 | 749.636 | 707.084 | 0.962x | 0.908x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 547.475 | 693.077 | 695.510 | 1.266x | 1.270x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.125 | 0.660 | 1.041 | 5.261x | 8.299x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.125 | 0.661 | 1.053 | 5.297x | 8.438x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 419.214 | 416.541 | 424.596 | 0.994x | 1.013x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 427.895 | 421.308 | 425.091 | 0.985x | 0.993x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.354 | 0.581 | 1.010 | 1.642x | 2.853x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.318 | 0.571 | 1.003 | 1.795x | 3.152x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 773.173 | 772.698 | 708.626 | 0.999x | 0.917x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 545.412 | 771.719 | 770.237 | 1.415x | 1.412x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.124 | 0.671 | 0.967 | 5.421x | 7.811x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.127 | 0.675 | 0.967 | 5.308x | 7.603x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 432.720 | 425.992 | 437.160 | 0.984x | 1.010x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 437.254 | 436.838 | 460.647 | 0.999x | 1.053x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.353 | 0.553 | 1.052 | 1.568x | 2.982x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.319 | 0.560 | 1.046 | 1.757x | 3.286x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 758.956 | 769.707 | 712.717 | 1.014x | 0.939x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 569.476 | 785.629 | 754.301 | 1.380x | 1.325x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.124 | 0.650 | 0.990 | 5.251x | 7.989x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.128 | 0.653 | 0.988 | 5.090x | 7.698x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 426.885 | 452.954 | 454.134 | 1.061x | 1.064x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 426.096 | 441.351 | 438.577 | 1.036x | 1.029x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.370 | 5.290 | 5.274 | 14.288x | 14.244x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.349 | 5.081 | 5.315 | 14.565x | 15.237x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 2602.972 | 823.912 | 835.262 | 0.317x | 0.321x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 169.711 | 815.159 | 818.012 | 4.803x | 4.820x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.132 | 5.247 | 4.229 | 39.648x | 31.952x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.131 | 5.174 | 4.190 | 39.413x | 31.917x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 2495.037 | 811.603 | 797.488 | 0.325x | 0.320x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 108.806 | 833.578 | 819.910 | 7.661x | 7.536x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.366 | 5.130 | 5.232 | 14.002x | 14.280x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.461 | 5.323 | 5.244 | 11.543x | 11.371x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 2652.617 | 849.783 | 840.925 | 0.320x | 0.317x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 168.202 | 860.485 | 861.968 | 5.116x | 5.125x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.193 | 6.472 | 6.301 | 33.554x | 32.668x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.131 | 5.248 | 4.193 | 40.083x | 32.024x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 2539.685 | 852.872 | 840.397 | 0.336x | 0.331x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 112.785 | 839.068 | 824.246 | 7.440x | 7.308x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.358 | 5.149 | 5.150 | 14.364x | 14.368x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.348 | 5.108 | 5.175 | 14.659x | 14.851x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4979.390 | 852.368 | 840.183 | 0.171x | 0.169x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 223.970 | 809.379 | 815.919 | 3.614x | 3.643x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.128 | 5.218 | 4.218 | 40.702x | 32.900x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.128 | 5.352 | 4.282 | 41.653x | 33.332x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4876.512 | 834.381 | 818.815 | 0.171x | 0.168x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 119.687 | 812.566 | 808.602 | 6.789x | 6.756x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.365 | 5.788 | 5.359 | 15.848x | 14.675x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.351 | 5.764 | 5.137 | 16.423x | 14.637x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1740.472 | 2195.987 | 2159.748 | 1.262x | 1.241x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 773.307 | 2214.592 | 2209.657 | 2.864x | 2.857x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.134 | 5.772 | 4.209 | 42.928x | 31.303x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.132 | 5.805 | 4.218 | 43.972x | 31.952x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1614.806 | 2206.101 | 2165.154 | 1.366x | 1.341x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 784.233 | 2265.242 | 2167.230 | 2.888x | 2.764x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.624 | 1.070 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.362 | 0.607 | 1.064 | 1.679x | 2.942x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 3576.706 | 402.739 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 465.102 | 3488.614 | 390.960 | 7.501x | 0.841x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.704 | 0.975 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.136 | 0.711 | 0.967 | 5.242x | 7.135x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 3410.746 | 432.723 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 273.406 | 3409.067 | 437.948 | 12.469x | 1.602x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.609 | 1.008 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.357 | 0.604 | 1.017 | 1.692x | 2.848x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 3503.133 | 386.101 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 470.347 | 3488.521 | 390.039 | 7.417x | 0.829x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.709 | 0.951 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.132 | 0.708 | 0.964 | 5.369x | 7.313x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 3421.048 | 444.898 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 269.675 | 3431.542 | 452.689 | 12.725x | 1.679x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.615 | 1.032 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.356 | 0.613 | 1.011 | 1.722x | 2.841x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 3493.231 | 380.613 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 481.627 | 3505.022 | 415.483 | 7.277x | 0.863x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.708 | 0.979 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.134 | 0.703 | 0.957 | 5.261x | 7.169x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 3431.559 | 346.572 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 265.521 | 3431.721 | 339.777 | 12.924x | 1.280x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.595 | 1.027 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.357 | 0.597 | 1.044 | 1.670x | 2.921x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 3542.915 | 407.436 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 453.023 | 3479.046 | 399.923 | 7.680x | 0.883x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.703 | 0.985 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.132 | 0.711 | 0.972 | 5.376x | 7.347x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 3547.198 | 403.707 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 262.882 | 3543.711 | 392.626 | 13.480x | 1.494x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 4.999 | 5.250 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.369 | 5.128 | 5.317 | 13.890x | 14.404x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1684.807 | 1855.680 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 399.558 | 1717.165 | 1831.799 | 4.298x | 4.585x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 5.122 | 4.213 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.132 | 5.002 | 4.236 | 37.888x | 32.090x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1662.065 | 1785.688 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 219.143 | 1629.537 | 1786.816 | 7.436x | 8.154x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 4.978 | 5.273 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.354 | 5.162 | 5.287 | 14.588x | 14.942x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1677.637 | 1894.051 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 395.113 | 1722.448 | 1859.031 | 4.359x | 4.705x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 5.099 | 4.237 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.136 | 5.153 | 4.448 | 37.796x | 32.623x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1659.840 | 1864.367 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 229.366 | 1714.892 | 1914.707 | 7.477x | 8.348x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 5.013 | 5.145 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.355 | 5.011 | 5.208 | 14.124x | 14.677x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1795.767 | 1928.923 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 473.657 | 1797.696 | 1936.727 | 3.795x | 4.089x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 5.128 | 4.219 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.134 | 5.106 | 4.334 | 38.055x | 32.296x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1716.416 | 1938.190 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 270.733 | 1755.650 | 1906.886 | 6.485x | 7.043x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 5.656 | 5.126 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.363 | 5.853 | 5.208 | 16.140x | 14.359x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3363.784 | 4045.694 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 1468.582 | 3421.268 | 3940.049 | 2.330x | 2.683x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 5.727 | 4.195 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.135 | 5.687 | 4.147 | 42.169x | 30.747x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3617.722 | 3900.059 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 1319.350 | 3405.012 | 3876.463 | 2.581x | 2.938x |

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
