# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-19T12:18:09.146153+00:00
- Commit: `ffa2ef3`
- CPU: aarch64 (aarch64)
- OS: Linux-7.0.0-1017-azure-aarch64-with-glibc2.39
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.4.9
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.596 | 0.790 | 1.475 | 1.326x | 2.476x |
| FP32 | ADD | allocating | 7 | NEON | 0.583 | 0.799 | 1.396 | 1.370x | 2.394x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 479.425 | 552.639 | 322.348 | 1.153x | 0.672x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 551.485 | 553.275 | 321.021 | 1.003x | 0.582x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.237 | 0.926 | 1.235 | 3.907x | 5.207x |
| FP32 | ADD | inplace | 7 | NEON | 0.256 | 0.925 | 1.253 | 3.610x | 4.891x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 352.136 | 266.884 | 297.568 | 0.758x | 0.845x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 357.141 | 292.226 | 282.686 | 0.818x | 0.792x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.609 | 0.786 | 1.401 | 1.291x | 2.301x |
| FP32 | SUB | allocating | 7 | NEON | 0.596 | 0.797 | 1.430 | 1.337x | 2.399x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 489.281 | 574.122 | 353.133 | 1.173x | 0.722x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 429.935 | 638.180 | 343.534 | 1.484x | 0.799x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.235 | 0.941 | 1.329 | 4.005x | 5.660x |
| FP32 | SUB | inplace | 7 | NEON | 0.266 | 0.940 | 1.268 | 3.530x | 4.763x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 355.531 | 323.257 | 314.711 | 0.909x | 0.885x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 353.406 | 270.261 | 281.707 | 0.765x | 0.797x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.597 | 0.794 | 1.390 | 1.330x | 2.329x |
| FP32 | MUL | allocating | 7 | NEON | 0.585 | 0.807 | 1.405 | 1.381x | 2.403x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 479.169 | 681.380 | 369.092 | 1.422x | 0.770x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 421.080 | 615.036 | 419.077 | 1.461x | 0.995x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.230 | 0.952 | 1.211 | 4.138x | 5.264x |
| FP32 | MUL | inplace | 7 | NEON | 0.259 | 0.939 | 1.222 | 3.629x | 4.721x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 357.778 | 300.450 | 311.197 | 0.840x | 0.870x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 400.805 | 298.002 | 368.608 | 0.744x | 0.920x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.592 | 0.778 | 1.392 | 1.314x | 2.352x |
| FP32 | DIV | allocating | 7 | NEON | 0.576 | 0.792 | 1.397 | 1.375x | 2.425x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 695.084 | 729.646 | 508.233 | 1.050x | 0.731x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 707.191 | 754.057 | 504.436 | 1.066x | 0.713x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.236 | 0.931 | 1.620 | 3.939x | 6.855x |
| FP32 | DIV | inplace | 7 | NEON | 0.256 | 0.931 | 1.577 | 3.632x | 6.153x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 478.551 | 473.735 | 487.915 | 0.990x | 1.020x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 475.597 | 469.351 | 487.668 | 0.987x | 1.025x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.619 | 6.503 | 9.630 | 10.498x | 15.545x |
| INT8 | ADD | allocating | 7 | NEON | 0.646 | 6.336 | 9.164 | 9.813x | 14.192x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1020.361 | 713.037 | 793.975 | 0.699x | 0.778x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 121.058 | 599.515 | 673.044 | 4.952x | 5.560x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.241 | 6.399 | 7.254 | 26.539x | 30.083x |
| INT8 | ADD | inplace | 7 | NEON | 0.270 | 6.390 | 7.085 | 23.691x | 26.264x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 958.473 | 572.941 | 657.479 | 0.598x | 0.686x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 93.875 | 571.649 | 635.491 | 6.089x | 6.770x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.610 | 6.337 | 9.078 | 10.383x | 14.873x |
| INT8 | SUB | allocating | 7 | NEON | 0.661 | 6.356 | 10.302 | 9.619x | 15.591x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1019.090 | 592.851 | 652.972 | 0.582x | 0.641x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 122.729 | 590.238 | 627.098 | 4.809x | 5.110x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.238 | 6.375 | 8.857 | 26.834x | 37.278x |
| INT8 | SUB | inplace | 7 | NEON | 0.276 | 6.351 | 7.781 | 23.013x | 28.193x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 957.476 | 588.772 | 642.308 | 0.615x | 0.671x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 98.330 | 559.992 | 614.273 | 5.695x | 6.247x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.601 | 6.328 | 7.674 | 10.528x | 12.766x |
| INT8 | MUL | allocating | 7 | NEON | 0.659 | 6.404 | 8.334 | 9.725x | 12.656x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 970.693 | 564.445 | 611.412 | 0.581x | 0.630x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 120.999 | 563.695 | 604.903 | 4.659x | 4.999x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.236 | 6.344 | 6.355 | 26.885x | 26.931x |
| INT8 | MUL | inplace | 7 | NEON | 0.265 | 6.384 | 6.542 | 24.081x | 24.676x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 912.158 | 564.082 | 583.865 | 0.618x | 0.640x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 92.843 | 555.691 | 601.807 | 5.985x | 6.482x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.622 | 7.032 | 10.272 | 11.314x | 16.526x |
| INT8 | DIV | allocating | 7 | NEON | 0.653 | 7.002 | 10.599 | 10.722x | 16.230x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1678.781 | 2903.597 | 2076.326 | 1.730x | 1.237x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 577.294 | 2831.464 | 2077.073 | 4.905x | 3.598x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.237 | 7.050 | 8.059 | 29.794x | 34.060x |
| INT8 | DIV | inplace | 7 | NEON | 0.273 | 7.053 | 9.399 | 25.824x | 34.415x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1614.432 | 2837.114 | 2062.219 | 1.757x | 1.277x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 518.119 | 2833.051 | 2056.893 | 5.468x | 3.970x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.813 | 1.387 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.650 | 0.817 | 1.413 | 1.256x | 2.173x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1953.273 | 173.942 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1011.050 | 1960.347 | 159.557 | 1.939x | 0.158x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.961 | 1.248 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.261 | 0.958 | 1.230 | 3.674x | 4.720x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1885.071 | 155.650 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 898.297 | 1888.598 | 146.861 | 2.102x | 0.163x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.806 | 1.382 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.650 | 0.817 | 1.438 | 1.257x | 2.212x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 2010.101 | 177.731 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1011.035 | 1982.960 | 192.635 | 1.961x | 0.191x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.969 | 1.363 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.264 | 0.955 | 1.349 | 3.623x | 5.119x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1907.877 | 135.638 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 901.403 | 1902.950 | 154.077 | 2.111x | 0.171x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.801 | 1.382 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.648 | 0.813 | 1.402 | 1.254x | 2.164x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1988.118 | 182.262 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1012.332 | 1973.805 | 167.547 | 1.950x | 0.166x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.950 | 1.219 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.259 | 0.954 | 1.216 | 3.688x | 4.702x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1899.630 | 147.801 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 900.790 | 1898.384 | 159.479 | 2.107x | 0.177x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.807 | 1.371 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.664 | 0.811 | 1.409 | 1.220x | 2.121x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2185.521 | 362.221 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1304.857 | 2180.491 | 368.801 | 1.671x | 0.283x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.952 | 1.567 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.272 | 0.954 | 1.598 | 3.509x | 5.878x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2104.910 | 354.661 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1196.515 | 2105.002 | 354.751 | 1.759x | 0.296x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.336 | 9.292 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.649 | 6.331 | 8.996 | 9.753x | 13.859x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1364.899 | 1483.691 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 275.572 | 1338.161 | 1467.506 | 4.856x | 5.325x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.314 | 6.599 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.271 | 6.297 | 6.784 | 23.275x | 25.077x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1308.109 | 1470.653 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 179.292 | 1310.310 | 1466.850 | 7.308x | 8.181x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.246 | 10.323 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.657 | 6.224 | 10.338 | 9.467x | 15.726x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1362.695 | 1461.617 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 242.656 | 1370.512 | 1487.699 | 5.648x | 6.131x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.258 | 7.049 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.267 | 6.292 | 6.709 | 23.605x | 25.172x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1312.139 | 1427.672 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 183.544 | 1308.349 | 1474.673 | 7.128x | 8.034x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.287 | 8.242 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.655 | 6.247 | 9.249 | 9.535x | 14.116x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1459.153 | 1444.118 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 252.819 | 1450.176 | 1408.376 | 5.736x | 5.571x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.293 | 6.397 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.268 | 6.271 | 7.340 | 23.400x | 27.389x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1402.065 | 1432.597 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 177.666 | 1395.042 | 1415.338 | 7.852x | 7.966x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.893 | 8.977 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.664 | 6.884 | 9.832 | 10.373x | 14.816x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3868.583 | 2705.920 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 618.452 | 4038.202 | 2726.899 | 6.530x | 4.409x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.878 | 7.838 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.268 | 6.907 | 8.239 | 25.760x | 30.728x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3892.379 | 2677.882 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 521.780 | 3943.309 | 2704.282 | 7.557x | 5.183x |

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
