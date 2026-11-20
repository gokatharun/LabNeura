# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-20T07:39:37.760814+00:00
- Commit: `70c975a`
- CPU: AMD EPYC 7763 64-Core Processor (x86_64)
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.544 | 1.033 | 1.821 | 1.899x | 3.348x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.519 | 1.036 | 1.820 | 1.997x | 3.508x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 472.695 | 770.106 | 732.402 | 1.629x | 1.549x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 363.199 | 735.349 | 838.061 | 2.025x | 2.307x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.200 | 1.167 | 1.624 | 5.849x | 8.139x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.200 | 1.169 | 1.626 | 5.855x | 8.149x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 270.386 | 261.755 | 276.455 | 0.968x | 1.022x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 323.998 | 252.069 | 262.541 | 0.778x | 0.810x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.546 | 1.023 | 1.796 | 1.874x | 3.290x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.513 | 1.019 | 1.806 | 1.986x | 3.519x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 494.383 | 791.842 | 757.834 | 1.602x | 1.533x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 354.047 | 712.564 | 354.192 | 2.013x | 1.000x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.198 | 1.148 | 1.616 | 5.804x | 8.173x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.201 | 1.148 | 1.628 | 5.707x | 8.093x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 262.381 | 251.776 | 265.041 | 0.960x | 1.010x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 315.540 | 249.189 | 269.417 | 0.790x | 0.854x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.544 | 1.015 | 1.833 | 1.867x | 3.371x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.518 | 1.015 | 1.825 | 1.961x | 3.524x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 467.874 | 745.947 | 722.125 | 1.594x | 1.543x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 350.290 | 689.257 | 667.534 | 1.968x | 1.906x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.196 | 1.156 | 1.587 | 5.907x | 8.107x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.204 | 1.153 | 1.620 | 5.659x | 7.949x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 270.171 | 264.773 | 245.267 | 0.980x | 0.908x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 287.766 | 255.288 | 254.850 | 0.887x | 0.886x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.555 | 1.006 | 1.801 | 1.813x | 3.246x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.529 | 0.998 | 1.785 | 1.887x | 3.376x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 456.610 | 705.502 | 694.722 | 1.545x | 1.521x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 407.825 | 671.000 | 762.608 | 1.645x | 1.870x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.195 | 1.131 | 1.608 | 5.802x | 8.246x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.202 | 1.131 | 1.611 | 5.592x | 7.970x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 271.859 | 251.100 | 253.159 | 0.924x | 0.931x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 308.663 | 247.256 | 273.132 | 0.801x | 0.885x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.563 | 7.969 | 8.293 | 14.143x | 14.719x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.578 | 7.845 | 8.196 | 13.571x | 14.177x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 2908.280 | 480.836 | 407.879 | 0.165x | 0.140x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 108.594 | 491.529 | 418.631 | 4.526x | 3.855x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.216 | 7.960 | 6.631 | 36.819x | 30.670x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.222 | 8.079 | 6.551 | 36.355x | 29.481x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 2844.094 | 474.287 | 394.833 | 0.167x | 0.139x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 69.625 | 493.088 | 401.150 | 7.082x | 5.762x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.555 | 7.879 | 8.193 | 14.190x | 14.756x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.574 | 7.811 | 8.159 | 13.616x | 14.223x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 2907.227 | 500.519 | 416.581 | 0.172x | 0.143x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 104.736 | 484.921 | 408.089 | 4.630x | 3.896x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.207 | 8.105 | 6.693 | 39.158x | 32.336x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.211 | 8.139 | 6.591 | 38.614x | 31.269x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 2848.615 | 470.222 | 390.930 | 0.165x | 0.137x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 70.935 | 478.281 | 388.749 | 6.743x | 5.480x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.549 | 7.868 | 8.039 | 14.341x | 14.653x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.583 | 7.860 | 8.384 | 13.490x | 14.389x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4894.369 | 531.163 | 422.940 | 0.109x | 0.086x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 149.877 | 494.164 | 449.878 | 3.297x | 3.002x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.208 | 8.076 | 6.362 | 38.795x | 30.562x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.215 | 7.890 | 6.451 | 36.730x | 30.032x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4832.792 | 480.744 | 389.431 | 0.099x | 0.081x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 101.090 | 469.277 | 393.065 | 4.642x | 3.888x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.560 | 8.949 | 8.435 | 15.985x | 15.067x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.569 | 8.894 | 8.037 | 15.624x | 14.118x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1926.763 | 1476.519 | 2202.842 | 0.766x | 1.143x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 763.585 | 1448.081 | 2204.101 | 1.896x | 2.887x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.409 | 11.354 | 6.641 | 27.737x | 16.224x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.217 | 9.054 | 6.507 | 41.764x | 30.012x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1872.241 | 1490.475 | 2186.505 | 0.796x | 1.168x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 712.720 | 1478.637 | 2193.089 | 2.075x | 3.077x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 1.079 | 1.823 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.586 | 1.083 | 1.822 | 1.850x | 3.111x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 6570.886 | 339.778 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 242.883 | 6910.280 | 342.157 | 28.451x | 1.409x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 1.188 | 1.644 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.219 | 1.208 | 1.631 | 5.504x | 7.434x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 6581.866 | 242.148 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 159.302 | 6581.243 | 244.017 | 41.313x | 1.532x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 1.077 | 1.805 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.573 | 1.071 | 1.813 | 1.868x | 3.162x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 6980.351 | 341.942 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 259.834 | 6977.160 | 352.225 | 26.852x | 1.356x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 1.207 | 1.648 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.210 | 1.193 | 1.643 | 5.670x | 7.811x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 6891.417 | 239.737 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 154.851 | 6873.581 | 239.964 | 44.388x | 1.550x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 1.073 | 1.820 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.590 | 1.071 | 1.808 | 1.815x | 3.065x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 6674.058 | 316.015 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 264.194 | 6661.373 | 317.753 | 25.214x | 1.203x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 1.195 | 1.608 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.218 | 1.202 | 1.591 | 5.519x | 7.305x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 6595.556 | 211.052 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 150.934 | 6821.055 | 215.042 | 45.192x | 1.425x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 1.065 | 1.795 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.572 | 1.060 | 1.830 | 1.855x | 3.202x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 6662.595 | 306.795 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 284.994 | 6663.493 | 310.010 | 23.381x | 1.088x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 1.186 | 1.617 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.213 | 1.188 | 1.628 | 5.590x | 7.658x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 6607.186 | 203.010 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 166.159 | 6580.605 | 205.201 | 39.604x | 1.235x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 7.756 | 8.376 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.568 | 7.794 | 8.343 | 13.716x | 14.682x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1113.581 | 1406.828 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 204.574 | 1218.028 | 1401.231 | 5.954x | 6.850x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 7.772 | 6.537 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.213 | 7.803 | 6.550 | 36.703x | 30.811x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1040.881 | 1336.893 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 133.871 | 1078.430 | 1322.769 | 8.056x | 9.881x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 7.767 | 8.483 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.605 | 7.763 | 8.151 | 12.829x | 13.469x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1172.086 | 1407.550 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 234.767 | 1174.825 | 1392.512 | 5.004x | 5.931x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 7.751 | 6.690 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.222 | 7.797 | 6.518 | 35.155x | 29.387x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1166.235 | 1319.298 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 134.971 | 1066.770 | 1351.042 | 7.904x | 10.010x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 7.701 | 8.204 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.584 | 7.790 | 8.060 | 13.337x | 13.799x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1263.693 | 1431.830 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 293.022 | 1218.016 | 1456.301 | 4.157x | 4.970x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 7.781 | 6.433 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.217 | 7.848 | 6.407 | 36.098x | 29.470x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1051.151 | 1351.365 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 193.364 | 1153.013 | 1365.537 | 5.963x | 7.062x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 8.665 | 8.217 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.591 | 8.609 | 8.274 | 14.579x | 14.012x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 2724.250 | 2948.678 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 772.112 | 2716.915 | 2927.467 | 3.519x | 3.792x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 8.768 | 6.431 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.219 | 8.797 | 6.463 | 40.132x | 29.486x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 2695.762 | 2833.217 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 644.472 | 2720.533 | 2868.274 | 4.221x | 4.451x |

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
