# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-19T12:19:50.906200+00:00
- Commit: `ffa2ef3`
- CPU: INTEL(R) XEON(R) PLATINUM 8573C (x86_64)
- OS: Linux-7.0.0-1019-azure-x86_64-with-glibc2.43
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.4.9
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.525 | 0.784 | 1.511 | 1.492x | 2.877x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.456 | 0.789 | 1.596 | 1.729x | 3.498x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 827.714 | 938.182 | 901.966 | 1.133x | 1.090x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 736.476 | 956.006 | 929.685 | 1.298x | 1.262x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.181 | 0.960 | 1.495 | 5.313x | 8.273x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.181 | 0.966 | 1.592 | 5.334x | 8.791x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 487.628 | 522.680 | 538.982 | 1.072x | 1.105x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 498.664 | 517.898 | 546.279 | 1.039x | 1.095x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.504 | 0.788 | 1.430 | 1.563x | 2.838x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.453 | 0.787 | 1.550 | 1.736x | 3.422x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 797.984 | 935.841 | 883.518 | 1.173x | 1.107x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 666.415 | 952.479 | 908.999 | 1.429x | 1.364x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.181 | 1.002 | 1.333 | 5.544x | 7.373x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.178 | 0.938 | 1.402 | 5.272x | 7.885x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 483.001 | 508.166 | 529.047 | 1.052x | 1.095x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 482.046 | 504.201 | 513.188 | 1.046x | 1.065x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.534 | 0.805 | 1.679 | 1.509x | 3.147x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.448 | 0.863 | 1.437 | 1.926x | 3.206x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 815.238 | 967.463 | 884.474 | 1.187x | 1.085x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 708.534 | 934.381 | 875.827 | 1.319x | 1.236x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.179 | 0.929 | 1.364 | 5.184x | 7.610x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.182 | 0.944 | 1.387 | 5.182x | 7.613x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 485.220 | 509.492 | 515.447 | 1.050x | 1.062x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 475.915 | 519.010 | 511.682 | 1.091x | 1.075x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.495 | 0.784 | 1.469 | 1.584x | 2.969x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.455 | 0.869 | 1.470 | 1.911x | 3.232x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 790.351 | 952.149 | 908.528 | 1.205x | 1.150x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 679.230 | 969.098 | 915.675 | 1.427x | 1.348x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.181 | 0.921 | 1.385 | 5.090x | 7.657x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.179 | 0.928 | 1.353 | 5.171x | 7.538x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 487.102 | 504.299 | 522.270 | 1.035x | 1.072x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 483.645 | 499.830 | 522.855 | 1.033x | 1.081x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.479 | 6.696 | 7.023 | 13.976x | 14.657x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.489 | 6.991 | 6.979 | 14.309x | 14.285x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 3491.842 | 855.958 | 882.320 | 0.245x | 0.253x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 171.380 | 870.462 | 892.987 | 5.079x | 5.211x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.176 | 6.998 | 5.731 | 39.677x | 32.499x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.182 | 6.954 | 5.615 | 38.210x | 30.850x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 3402.676 | 878.840 | 891.292 | 0.258x | 0.262x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 122.940 | 874.876 | 896.574 | 7.116x | 7.293x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.498 | 6.868 | 6.911 | 13.789x | 13.875x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.488 | 6.696 | 7.066 | 13.724x | 14.482x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 3300.280 | 913.797 | 952.081 | 0.277x | 0.288x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 175.904 | 900.554 | 911.238 | 5.120x | 5.180x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.180 | 6.887 | 5.462 | 38.208x | 30.302x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.178 | 6.895 | 5.774 | 38.665x | 32.378x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 3210.285 | 894.513 | 915.043 | 0.279x | 0.285x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 125.056 | 886.258 | 900.660 | 7.087x | 7.202x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.500 | 6.826 | 6.814 | 13.658x | 13.634x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.493 | 6.914 | 6.743 | 14.019x | 13.672x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 6683.506 | 890.704 | 904.767 | 0.133x | 0.135x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 217.353 | 865.792 | 879.958 | 3.983x | 4.049x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.186 | 6.887 | 5.590 | 37.099x | 30.114x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.180 | 6.533 | 5.770 | 36.264x | 32.027x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 6671.620 | 898.396 | 909.064 | 0.135x | 0.136x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 157.946 | 890.538 | 906.720 | 5.638x | 5.741x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.494 | 7.530 | 6.957 | 15.235x | 14.075x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.489 | 7.802 | 6.918 | 15.946x | 14.138x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 2170.401 | 2563.673 | 2636.137 | 1.181x | 1.215x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 853.269 | 2516.019 | 2664.247 | 2.949x | 3.122x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.183 | 7.778 | 5.764 | 42.553x | 31.537x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.188 | 7.708 | 5.635 | 41.107x | 30.053x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 2055.569 | 2532.855 | 2631.900 | 1.232x | 1.280x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 762.504 | 2548.978 | 2649.846 | 3.343x | 3.475x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.857 | 1.433 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.498 | 0.847 | 1.424 | 1.699x | 2.858x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 6947.152 | 471.207 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 462.522 | 7117.158 | 441.153 | 15.388x | 0.954x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 1.026 | 1.453 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.187 | 1.016 | 1.445 | 5.448x | 7.746x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 7047.646 | 392.761 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 320.362 | 6873.055 | 376.296 | 21.454x | 1.175x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.845 | 1.398 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.482 | 0.848 | 1.407 | 1.758x | 2.917x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 6192.441 | 479.582 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 463.188 | 6129.932 | 451.887 | 13.234x | 0.976x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.974 | 1.266 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.181 | 1.014 | 1.323 | 5.609x | 7.315x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 6120.368 | 375.588 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 320.826 | 6104.863 | 376.798 | 19.029x | 1.174x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.853 | 1.415 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.505 | 0.845 | 1.402 | 1.674x | 2.777x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 7056.335 | 460.805 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 466.492 | 7035.636 | 446.503 | 15.082x | 0.957x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 1.014 | 1.290 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.182 | 1.020 | 1.286 | 5.613x | 7.077x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 7034.818 | 368.349 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 318.382 | 6903.112 | 369.312 | 21.682x | 1.160x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.834 | 1.415 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.503 | 0.845 | 1.422 | 1.681x | 2.827x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 7058.848 | 444.333 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 481.888 | 7053.753 | 463.038 | 14.638x | 0.961x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 1.015 | 1.325 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.175 | 0.986 | 1.283 | 5.637x | 7.336x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 6964.812 | 373.665 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 326.758 | 7003.057 | 380.829 | 21.432x | 1.165x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.875 | 7.038 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.494 | 6.638 | 7.040 | 13.448x | 14.264x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1996.737 | 2177.236 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 418.842 | 2003.906 | 2161.439 | 4.784x | 5.161x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.770 | 5.736 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.181 | 6.904 | 5.813 | 38.037x | 32.029x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1983.462 | 2105.609 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 250.684 | 1915.508 | 2070.730 | 7.641x | 8.260x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.680 | 6.942 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.505 | 6.635 | 6.981 | 13.135x | 13.821x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1914.140 | 2134.875 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 404.486 | 2018.000 | 2150.325 | 4.989x | 5.316x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.656 | 5.605 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.184 | 6.724 | 5.608 | 36.512x | 30.450x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1850.166 | 2026.710 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 261.310 | 1977.427 | 2140.858 | 7.567x | 8.193x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.584 | 6.812 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.498 | 6.549 | 6.482 | 13.165x | 13.029x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1999.570 | 2239.626 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 490.148 | 1991.389 | 2254.009 | 4.063x | 4.599x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.692 | 5.732 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.178 | 6.490 | 5.364 | 36.465x | 30.136x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 2019.829 | 2268.312 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 329.901 | 2004.337 | 2260.592 | 6.076x | 6.852x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.986 | 6.742 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.506 | 7.352 | 6.923 | 14.528x | 13.680x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 4196.846 | 4832.381 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 974.286 | 4195.837 | 4838.454 | 4.307x | 4.966x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 7.516 | 5.307 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.191 | 7.478 | 5.490 | 39.159x | 28.752x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 4253.540 | 4805.673 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 775.220 | 4194.828 | 4788.963 | 5.411x | 6.178x |

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
