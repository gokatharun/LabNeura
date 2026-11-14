# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-14T10:22:47.804712+00:00
- Commit: `849e1ea`
- CPU: AMD EPYC 7763 64-Core Processor (x86_64)
- OS: Linux-7.0.0-1019-azure-x86_64-with-glibc2.43
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.576 | 1.036 | 1.843 | 1.799x | 3.200x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.537 | 1.045 | 1.856 | 1.948x | 3.459x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 450.578 | 683.474 | 728.696 | 1.517x | 1.617x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 365.747 | 743.135 | 729.858 | 2.032x | 1.996x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.204 | 1.180 | 1.709 | 5.786x | 8.379x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.208 | 1.189 | 1.709 | 5.705x | 8.200x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 264.463 | 260.258 | 277.633 | 0.984x | 1.050x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 325.467 | 266.875 | 287.917 | 0.820x | 0.885x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.582 | 1.049 | 1.873 | 1.803x | 3.219x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.537 | 1.039 | 1.899 | 1.935x | 3.538x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 480.506 | 679.268 | 706.439 | 1.414x | 1.470x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 409.316 | 701.492 | 680.797 | 1.714x | 1.663x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.204 | 1.172 | 1.708 | 5.752x | 8.380x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.205 | 1.175 | 1.704 | 5.745x | 8.329x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 264.766 | 286.399 | 275.604 | 1.082x | 1.041x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 333.334 | 278.319 | 287.884 | 0.835x | 0.864x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.581 | 1.042 | 1.822 | 1.792x | 3.136x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.554 | 1.034 | 1.824 | 1.868x | 3.294x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 421.290 | 701.032 | 685.159 | 1.664x | 1.626x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 371.084 | 690.126 | 678.279 | 1.860x | 1.828x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.207 | 1.171 | 1.654 | 5.669x | 8.009x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.215 | 1.170 | 1.645 | 5.433x | 7.637x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 262.226 | 255.829 | 261.091 | 0.976x | 0.996x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 288.841 | 270.554 | 260.300 | 0.937x | 0.901x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.588 | 1.028 | 1.814 | 1.749x | 3.086x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.537 | 1.028 | 1.805 | 1.914x | 3.359x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 449.513 | 672.328 | 683.083 | 1.496x | 1.520x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 333.587 | 681.573 | 693.778 | 2.043x | 2.080x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.207 | 1.167 | 1.651 | 5.639x | 7.977x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.214 | 1.163 | 1.662 | 5.431x | 7.757x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 259.877 | 267.692 | 274.001 | 1.030x | 1.054x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 314.737 | 262.569 | 257.440 | 0.834x | 0.818x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.585 | 8.042 | 8.622 | 13.740x | 14.731x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.597 | 7.998 | 8.510 | 13.399x | 14.257x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 2898.131 | 485.416 | 406.095 | 0.167x | 0.140x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 101.819 | 488.038 | 414.432 | 4.793x | 4.070x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.220 | 8.079 | 6.751 | 36.655x | 30.628x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.220 | 8.238 | 6.728 | 37.408x | 30.549x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 2847.251 | 460.577 | 404.605 | 0.162x | 0.142x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 70.676 | 462.789 | 393.494 | 6.548x | 5.568x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.595 | 8.042 | 8.401 | 13.513x | 14.117x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.584 | 8.096 | 8.530 | 13.866x | 14.608x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 2905.256 | 487.738 | 411.180 | 0.168x | 0.142x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 103.438 | 494.415 | 406.101 | 4.780x | 3.926x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.218 | 8.235 | 6.681 | 37.701x | 30.590x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.223 | 8.307 | 6.677 | 37.213x | 29.911x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 2849.867 | 470.682 | 390.671 | 0.165x | 0.137x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 68.654 | 465.688 | 404.244 | 6.783x | 5.888x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.581 | 8.220 | 8.281 | 14.156x | 14.260x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.583 | 8.008 | 8.313 | 13.744x | 14.265x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4898.095 | 487.859 | 416.478 | 0.100x | 0.085x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 146.344 | 487.613 | 415.241 | 3.332x | 2.837x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.225 | 8.109 | 6.549 | 35.972x | 29.052x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.221 | 8.244 | 6.563 | 37.231x | 29.640x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4835.094 | 456.197 | 391.886 | 0.094x | 0.081x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 102.781 | 453.591 | 396.816 | 4.413x | 3.861x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.600 | 9.220 | 8.597 | 15.368x | 14.330x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.592 | 9.266 | 9.413 | 15.660x | 15.908x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1929.818 | 1463.975 | 2204.808 | 0.759x | 1.142x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 757.384 | 1490.252 | 2194.821 | 1.968x | 2.898x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.224 | 9.344 | 6.647 | 41.635x | 29.619x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.233 | 9.290 | 6.700 | 39.897x | 28.776x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1874.444 | 1432.180 | 2172.879 | 0.764x | 1.159x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 714.337 | 1511.973 | 2173.642 | 2.117x | 3.043x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 1.093 | 1.886 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.582 | 1.093 | 1.866 | 1.879x | 3.208x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 6664.848 | 345.423 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 243.009 | 6671.869 | 347.256 | 27.455x | 1.429x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 1.243 | 1.719 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.218 | 1.241 | 1.717 | 5.680x | 7.860x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 6582.071 | 238.665 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 149.341 | 6571.857 | 239.623 | 44.006x | 1.605x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 1.096 | 1.887 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.592 | 1.087 | 1.879 | 1.836x | 3.173x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 7013.146 | 344.929 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 251.609 | 6973.921 | 350.084 | 27.717x | 1.391x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 1.256 | 1.722 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.232 | 1.236 | 1.711 | 5.333x | 7.381x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 6880.522 | 243.105 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 148.334 | 6874.050 | 248.238 | 46.342x | 1.674x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 1.105 | 1.823 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.601 | 1.101 | 1.807 | 1.834x | 3.009x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 6653.192 | 316.066 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 236.760 | 6675.431 | 317.496 | 28.195x | 1.341x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 1.256 | 1.653 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.227 | 1.261 | 1.642 | 5.562x | 7.248x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 6635.759 | 207.953 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 149.622 | 6576.483 | 212.470 | 43.954x | 1.420x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 1.078 | 1.800 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.598 | 1.079 | 1.804 | 1.805x | 3.017x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 6658.311 | 314.201 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 269.479 | 6654.650 | 318.832 | 24.695x | 1.183x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 1.218 | 1.649 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.231 | 1.221 | 1.665 | 5.281x | 7.200x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 6563.708 | 206.971 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 157.381 | 6568.256 | 218.619 | 41.735x | 1.389x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 7.892 | 8.628 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.606 | 7.944 | 8.422 | 13.107x | 13.894x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1111.106 | 1354.023 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 217.742 | 1152.655 | 1391.732 | 5.294x | 6.392x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 7.952 | 6.665 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.223 | 7.992 | 6.885 | 35.805x | 30.845x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1038.646 | 1269.654 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 133.954 | 1089.753 | 1261.747 | 8.135x | 9.419x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 8.156 | 8.657 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.602 | 7.983 | 8.400 | 13.262x | 13.956x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1055.702 | 1366.529 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 218.981 | 1314.431 | 1376.121 | 6.002x | 6.284x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 8.082 | 6.728 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.224 | 8.126 | 6.669 | 36.306x | 29.796x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1056.987 | 1259.137 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 133.642 | 1052.696 | 1256.494 | 7.877x | 9.402x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 8.033 | 8.400 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.607 | 7.998 | 8.348 | 13.174x | 13.751x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1154.486 | 1405.174 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 307.640 | 1159.920 | 1440.145 | 3.770x | 4.681x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 8.179 | 6.531 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.226 | 8.261 | 6.622 | 36.616x | 29.352x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1082.379 | 1329.406 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 192.658 | 1115.072 | 1327.660 | 5.788x | 6.891x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 9.194 | 8.391 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.615 | 9.113 | 8.332 | 14.815x | 13.545x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 2749.730 | 2895.885 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 758.741 | 2763.929 | 2895.864 | 3.643x | 3.817x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 9.234 | 6.536 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.228 | 9.075 | 6.742 | 39.834x | 29.595x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 2643.506 | 2816.253 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 646.865 | 2755.658 | 2832.113 | 4.260x | 4.378x |

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
