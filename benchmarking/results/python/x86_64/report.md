# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-14T11:43:57.179359+00:00
- Commit: `039c9d5`
- CPU: Intel(R) Xeon(R) 6975P-C (x86_64)
- OS: Linux-7.0.0-1016-aws-x86_64-with-glibc2.43
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.4.3
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.354 | 0.576 | 1.042 | 1.627x | 2.945x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.318 | 0.567 | 1.052 | 1.784x | 3.309x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 794.871 | 725.340 | 707.869 | 0.913x | 0.891x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 595.684 | 712.647 | 699.725 | 1.196x | 1.175x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.123 | 0.682 | 0.964 | 5.543x | 7.833x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.132 | 0.695 | 0.940 | 5.268x | 7.129x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 423.018 | 446.769 | 448.678 | 1.056x | 1.061x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 427.489 | 438.956 | 421.520 | 1.027x | 0.986x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.355 | 0.571 | 1.008 | 1.612x | 2.843x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.318 | 0.556 | 1.015 | 1.750x | 3.191x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 792.737 | 741.480 | 718.486 | 0.935x | 0.906x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 576.328 | 770.220 | 756.544 | 1.336x | 1.313x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.124 | 0.671 | 0.956 | 5.412x | 7.706x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.124 | 0.668 | 0.951 | 5.366x | 7.647x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 440.998 | 441.579 | 459.276 | 1.001x | 1.041x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 444.659 | 439.910 | 452.385 | 0.989x | 1.017x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.356 | 0.571 | 0.999 | 1.601x | 2.801x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.325 | 0.571 | 0.990 | 1.756x | 3.043x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 788.785 | 698.458 | 543.374 | 0.885x | 0.689x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 598.368 | 715.860 | 734.547 | 1.196x | 1.228x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.123 | 0.668 | 0.915 | 5.417x | 7.419x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.126 | 0.665 | 0.944 | 5.301x | 7.523x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 430.789 | 438.050 | 443.923 | 1.017x | 1.030x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 443.647 | 425.569 | 454.591 | 0.959x | 1.025x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.344 | 0.550 | 1.053 | 1.596x | 3.058x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.315 | 0.550 | 1.064 | 1.749x | 3.380x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 792.963 | 718.442 | 724.234 | 0.906x | 0.913x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 596.247 | 731.148 | 752.772 | 1.226x | 1.263x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.125 | 0.646 | 1.022 | 5.150x | 8.145x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.128 | 0.645 | 0.963 | 5.025x | 7.501x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 427.903 | 441.655 | 435.639 | 1.032x | 1.018x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 433.909 | 449.231 | 451.518 | 1.035x | 1.041x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.385 | 5.328 | 5.162 | 13.825x | 13.393x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.347 | 5.360 | 5.258 | 15.429x | 15.136x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 2618.724 | 818.843 | 831.508 | 0.313x | 0.318x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 170.332 | 817.584 | 834.099 | 4.800x | 4.897x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.132 | 5.341 | 4.158 | 40.407x | 31.458x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.134 | 5.433 | 4.172 | 40.447x | 31.058x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 2509.256 | 805.700 | 799.262 | 0.321x | 0.319x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 108.105 | 809.917 | 809.229 | 7.492x | 7.486x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.373 | 5.276 | 5.210 | 14.150x | 13.972x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.348 | 5.164 | 5.077 | 14.843x | 14.593x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 2660.290 | 846.951 | 850.923 | 0.318x | 0.320x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 171.949 | 856.300 | 850.966 | 4.980x | 4.949x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.128 | 5.301 | 4.200 | 41.430x | 32.825x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.128 | 5.278 | 4.137 | 41.121x | 32.226x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 2567.154 | 834.508 | 829.862 | 0.325x | 0.323x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 114.425 | 841.596 | 837.339 | 7.355x | 7.318x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.357 | 5.168 | 5.147 | 14.461x | 14.402x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.363 | 5.200 | 5.257 | 14.310x | 14.469x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 5019.505 | 856.239 | 845.633 | 0.171x | 0.168x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 225.417 | 870.445 | 853.488 | 3.861x | 3.786x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.129 | 5.407 | 4.270 | 41.795x | 33.009x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.130 | 5.299 | 4.265 | 40.771x | 32.815x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4948.939 | 843.392 | 836.666 | 0.170x | 0.169x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 126.491 | 839.325 | 825.315 | 6.635x | 6.525x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.366 | 6.075 | 5.204 | 16.582x | 14.205x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.353 | 5.785 | 5.150 | 16.396x | 14.598x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1754.902 | 2284.885 | 2227.563 | 1.302x | 1.269x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 779.365 | 2331.614 | 2247.303 | 2.992x | 2.884x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.135 | 5.916 | 4.238 | 43.700x | 31.305x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.133 | 5.897 | 4.186 | 44.338x | 31.474x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1615.969 | 2307.176 | 2209.895 | 1.428x | 1.368x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 786.910 | 2273.857 | 2203.760 | 2.890x | 2.801x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.602 | 1.038 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.354 | 0.606 | 1.038 | 1.713x | 2.932x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 3499.002 | 393.431 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 448.133 | 3513.439 | 380.295 | 7.840x | 0.849x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.711 | 0.947 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.137 | 0.727 | 0.986 | 5.312x | 7.204x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 3447.437 | 405.969 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 252.671 | 3452.741 | 411.582 | 13.665x | 1.629x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.604 | 0.994 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.354 | 0.619 | 1.005 | 1.747x | 2.835x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 3527.951 | 390.118 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 456.192 | 3516.483 | 380.425 | 7.708x | 0.834x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.699 | 0.954 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.133 | 0.704 | 0.944 | 5.299x | 7.109x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 3452.399 | 407.520 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 252.449 | 3446.379 | 447.459 | 13.652x | 1.772x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.616 | 1.005 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.365 | 0.609 | 1.012 | 1.671x | 2.774x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 3552.341 | 375.687 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 450.404 | 3496.473 | 378.787 | 7.763x | 0.841x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.700 | 0.935 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.143 | 0.700 | 0.930 | 4.908x | 6.514x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 3443.719 | 320.887 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 250.497 | 3430.455 | 323.950 | 13.695x | 1.293x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.596 | 1.031 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.358 | 0.599 | 1.034 | 1.673x | 2.889x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 3504.425 | 375.612 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 442.631 | 3559.702 | 379.678 | 8.042x | 0.858x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.701 | 0.999 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.134 | 0.688 | 0.996 | 5.126x | 7.423x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 3424.754 | 367.600 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 249.201 | 3470.060 | 369.189 | 13.925x | 1.481x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 5.035 | 5.171 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.361 | 5.170 | 5.131 | 14.338x | 14.229x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1645.411 | 1769.139 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 380.874 | 1634.669 | 1736.095 | 4.292x | 4.558x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 5.242 | 4.201 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.135 | 5.015 | 4.111 | 37.269x | 30.549x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1585.859 | 1707.332 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 210.195 | 1626.156 | 1615.243 | 7.736x | 7.685x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 5.165 | 5.222 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.359 | 5.250 | 5.275 | 14.621x | 14.693x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1635.879 | 1761.521 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 380.080 | 1634.458 | 1761.649 | 4.300x | 4.635x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 5.169 | 4.135 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.133 | 5.129 | 4.242 | 38.457x | 31.806x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1589.483 | 1687.127 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 214.990 | 1674.062 | 1734.787 | 7.787x | 8.069x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 4.970 | 5.142 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.362 | 5.266 | 5.150 | 14.538x | 14.219x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1671.956 | 1791.690 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 449.131 | 1677.579 | 1799.632 | 3.735x | 4.007x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 5.203 | 4.189 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.136 | 5.209 | 4.099 | 38.315x | 30.156x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1631.290 | 1745.272 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 243.372 | 1600.275 | 1692.161 | 6.575x | 6.953x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 5.655 | 5.116 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.364 | 5.763 | 5.281 | 15.837x | 14.514x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3232.387 | 3784.960 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 1309.511 | 3230.051 | 3804.285 | 2.467x | 2.905x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 5.929 | 4.143 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.141 | 5.903 | 4.168 | 41.832x | 29.537x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3122.661 | 3800.134 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 1195.648 | 3208.561 | 3725.234 | 2.684x | 3.116x |

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
