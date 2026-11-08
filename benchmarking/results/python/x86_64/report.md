# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-08T08:14:28.541459+00:00
- Commit: `55ab392`
- CPU: INTEL(R) XEON(R) PLATINUM 8573C (x86_64)
- OS: Linux-7.0.0-1017-azure-x86_64-with-glibc2.39
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.472 | 0.682 | 1.198 | 1.446x | 2.540x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.444 | 0.664 | 1.210 | 1.496x | 2.727x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 840.058 | 782.526 | 673.364 | 0.932x | 0.802x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 613.182 | 826.958 | 644.396 | 1.349x | 1.051x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.160 | 1.214 | 1.115 | 7.571x | 6.953x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.194 | 0.804 | 1.118 | 4.143x | 5.760x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 533.946 | 459.156 | 398.954 | 0.860x | 0.747x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 480.756 | 498.614 | 352.845 | 1.037x | 0.734x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.452 | 0.664 | 1.201 | 1.468x | 2.654x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.441 | 0.667 | 1.435 | 1.511x | 3.252x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 844.886 | 827.841 | 668.754 | 0.980x | 0.792x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 639.516 | 526.670 | 457.020 | 0.824x | 0.715x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.164 | 0.790 | 1.109 | 4.817x | 6.759x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.192 | 0.792 | 1.107 | 4.118x | 5.751x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 538.143 | 482.817 | 390.962 | 0.897x | 0.727x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 476.342 | 477.772 | 378.381 | 1.003x | 0.794x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.454 | 0.664 | 1.197 | 1.463x | 2.637x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.440 | 0.665 | 1.185 | 1.511x | 2.693x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 816.348 | 808.949 | 682.587 | 0.991x | 0.836x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 594.285 | 832.056 | 687.750 | 1.400x | 1.157x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.160 | 0.790 | 1.054 | 4.930x | 6.582x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.195 | 0.787 | 1.061 | 4.045x | 5.454x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 536.497 | 471.465 | 326.123 | 0.879x | 0.608x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 475.566 | 489.548 | 328.775 | 1.029x | 0.691x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.547 | 0.655 | 1.179 | 1.196x | 2.155x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.441 | 0.659 | 1.235 | 1.494x | 2.799x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 822.921 | 872.178 | 698.559 | 1.060x | 0.849x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 595.582 | 881.544 | 650.212 | 1.480x | 1.092x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.163 | 0.793 | 1.096 | 4.861x | 6.718x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.191 | 0.787 | 1.097 | 4.116x | 5.740x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 534.858 | 455.472 | 347.996 | 0.852x | 0.651x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 477.316 | 485.763 | 339.017 | 1.018x | 0.710x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.475 | 5.473 | 5.719 | 11.512x | 12.030x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.515 | 5.410 | 6.263 | 10.495x | 12.149x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 3255.223 | 824.577 | 809.590 | 0.253x | 0.249x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 160.280 | 817.723 | 855.960 | 5.102x | 5.340x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.166 | 5.815 | 4.750 | 35.120x | 28.685x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.194 | 5.552 | 4.712 | 28.681x | 24.344x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 3147.970 | 808.697 | 788.411 | 0.257x | 0.250x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 122.125 | 802.596 | 783.745 | 6.572x | 6.418x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.463 | 5.525 | 5.895 | 11.922x | 12.722x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.510 | 5.577 | 5.976 | 10.941x | 11.723x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 3361.341 | 815.455 | 810.746 | 0.243x | 0.241x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 168.081 | 847.213 | 822.176 | 5.041x | 4.892x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.166 | 5.515 | 4.794 | 33.289x | 28.939x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.196 | 5.492 | 4.810 | 28.030x | 24.550x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 3272.585 | 825.529 | 789.012 | 0.252x | 0.241x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 122.162 | 815.573 | 792.545 | 6.676x | 6.488x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.453 | 5.566 | 5.907 | 12.291x | 13.045x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.518 | 5.561 | 5.757 | 10.743x | 11.122x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 5068.341 | 846.589 | 802.235 | 0.167x | 0.158x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 207.623 | 814.425 | 808.048 | 3.923x | 3.892x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.166 | 5.618 | 4.635 | 33.762x | 27.855x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.194 | 5.592 | 4.644 | 28.858x | 23.969x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 5012.577 | 829.886 | 792.905 | 0.166x | 0.158x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 162.176 | 811.762 | 791.949 | 5.005x | 4.883x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.458 | 6.135 | 5.895 | 13.400x | 12.875x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.517 | 6.223 | 6.069 | 12.030x | 11.732x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1826.112 | 2237.527 | 2292.711 | 1.225x | 1.256x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 721.822 | 2233.953 | 2302.638 | 3.095x | 3.190x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.163 | 6.265 | 4.682 | 38.321x | 28.638x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.194 | 6.339 | 4.661 | 32.619x | 23.984x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1724.829 | 2204.874 | 2282.318 | 1.278x | 1.323x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 639.117 | 2206.906 | 2237.397 | 3.453x | 3.501x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.728 | 1.198 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.517 | 0.727 | 1.199 | 1.406x | 2.320x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 5841.566 | 362.927 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 452.866 | 5893.228 | 391.717 | 13.013x | 0.865x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.857 | 1.132 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.195 | 0.855 | 1.141 | 4.374x | 5.837x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 5797.573 | 286.415 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 325.328 | 5695.587 | 267.168 | 17.507x | 0.821x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.722 | 1.209 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.546 | 0.745 | 1.262 | 1.366x | 2.312x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 5274.658 | 300.609 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 448.290 | 5291.397 | 410.411 | 11.804x | 0.916x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.853 | 1.135 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.194 | 0.857 | 1.129 | 4.424x | 5.832x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 5212.536 | 318.968 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 324.195 | 5518.379 | 331.958 | 17.022x | 1.024x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.717 | 1.226 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.518 | 0.722 | 1.220 | 1.392x | 2.354x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 5861.693 | 329.555 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 454.391 | 5972.635 | 391.723 | 13.144x | 0.862x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.849 | 1.091 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.196 | 0.850 | 1.069 | 4.348x | 5.465x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 5930.546 | 272.085 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 330.543 | 5955.154 | 259.867 | 18.016x | 0.786x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.719 | 1.215 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.519 | 0.722 | 1.192 | 1.390x | 2.294x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 5963.836 | 355.260 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 455.627 | 5904.064 | 372.251 | 12.958x | 0.817x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.852 | 1.094 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.197 | 0.860 | 1.105 | 4.357x | 5.598x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 5957.039 | 330.504 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 331.283 | 5920.062 | 305.035 | 17.870x | 0.921x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 5.445 | 5.801 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.508 | 5.483 | 5.857 | 10.783x | 11.518x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1754.002 | 1721.554 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 390.644 | 1721.417 | 1754.231 | 4.407x | 4.491x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 5.487 | 4.719 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.195 | 5.548 | 4.739 | 28.441x | 24.294x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1624.794 | 1730.555 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 248.714 | 1669.712 | 1675.626 | 6.713x | 6.737x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 5.350 | 6.103 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.513 | 5.451 | 5.908 | 10.636x | 11.527x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1671.642 | 1734.545 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 382.317 | 1717.709 | 1761.740 | 4.493x | 4.608x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 5.478 | 4.775 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.197 | 5.576 | 4.757 | 28.246x | 24.096x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1727.227 | 1764.068 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 249.352 | 1668.328 | 1686.761 | 6.691x | 6.765x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 5.370 | 5.798 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.517 | 5.539 | 5.986 | 10.723x | 11.587x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1707.191 | 1772.307 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 457.732 | 1694.194 | 1789.180 | 3.701x | 3.909x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 5.488 | 4.692 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.197 | 5.644 | 4.716 | 28.700x | 23.985x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1627.678 | 1750.452 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 332.313 | 1650.186 | 1708.863 | 4.966x | 5.142x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.130 | 5.813 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.515 | 6.216 | 5.774 | 12.059x | 11.201x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3200.323 | 3943.298 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 818.784 | 3309.815 | 3997.503 | 4.042x | 4.882x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.265 | 4.740 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.200 | 6.297 | 4.759 | 31.424x | 23.748x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3237.089 | 3976.524 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 646.001 | 3244.049 | 3925.827 | 5.022x | 6.077x |

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
