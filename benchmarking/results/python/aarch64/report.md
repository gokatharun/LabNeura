# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-20T16:25:58.542422+00:00
- Commit: `e807bf9`
- CPU: aarch64 (aarch64)
- OS: Linux-7.0.0-1017-azure-aarch64-with-glibc2.39
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.5.2
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.598 | 0.792 | 1.393 | 1.324x | 2.330x |
| FP32 | ADD | allocating | 7 | NEON | 0.582 | 0.786 | 1.388 | 1.351x | 2.385x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 490.743 | 525.451 | 336.607 | 1.071x | 0.686x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 413.014 | 486.203 | 290.533 | 1.177x | 0.703x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.223 | 0.914 | 1.297 | 4.098x | 5.817x |
| FP32 | ADD | inplace | 7 | NEON | 0.257 | 0.913 | 1.279 | 3.548x | 4.968x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 356.116 | 237.703 | 254.108 | 0.667x | 0.714x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 354.494 | 239.305 | 248.543 | 0.675x | 0.701x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.589 | 0.776 | 1.378 | 1.317x | 2.341x |
| FP32 | SUB | allocating | 7 | NEON | 0.577 | 0.772 | 1.402 | 1.337x | 2.429x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 488.224 | 563.506 | 288.594 | 1.154x | 0.591x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 488.916 | 531.761 | 286.698 | 1.088x | 0.586x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.220 | 0.920 | 1.584 | 4.189x | 7.211x |
| FP32 | SUB | inplace | 7 | NEON | 0.255 | 0.909 | 1.564 | 3.572x | 6.142x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 355.947 | 250.083 | 252.679 | 0.703x | 0.710x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 382.747 | 245.897 | 257.389 | 0.642x | 0.672x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.594 | 0.805 | 1.389 | 1.355x | 2.338x |
| FP32 | MUL | allocating | 7 | NEON | 0.590 | 0.819 | 1.386 | 1.388x | 2.350x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 484.241 | 281.105 | 305.607 | 0.581x | 0.631x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 424.443 | 538.828 | 306.384 | 1.269x | 0.722x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.217 | 0.943 | 1.216 | 4.339x | 5.594x |
| FP32 | MUL | inplace | 7 | NEON | 0.250 | 0.947 | 1.225 | 3.794x | 4.909x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 360.829 | 240.585 | 250.115 | 0.667x | 0.693x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 368.291 | 296.453 | 278.295 | 0.805x | 0.756x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.602 | 0.763 | 1.407 | 1.267x | 2.339x |
| FP32 | DIV | allocating | 7 | NEON | 0.575 | 0.763 | 1.391 | 1.328x | 2.421x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 692.109 | 715.470 | 491.253 | 1.034x | 0.710x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 480.454 | 710.704 | 488.843 | 1.479x | 1.017x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.231 | 0.900 | 1.638 | 3.901x | 7.100x |
| FP32 | DIV | inplace | 7 | NEON | 0.260 | 0.901 | 1.578 | 3.472x | 6.082x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 464.576 | 467.480 | 479.728 | 1.006x | 1.033x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 472.011 | 473.113 | 477.317 | 1.002x | 1.011x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.614 | 6.406 | 8.609 | 10.440x | 14.030x |
| INT8 | ADD | allocating | 7 | NEON | 0.646 | 6.304 | 8.165 | 9.759x | 12.640x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1018.234 | 575.115 | 617.103 | 0.565x | 0.606x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 119.588 | 567.773 | 611.546 | 4.748x | 5.114x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.241 | 6.407 | 6.740 | 26.590x | 27.971x |
| INT8 | ADD | inplace | 7 | NEON | 0.265 | 6.471 | 7.242 | 24.419x | 27.331x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 956.836 | 556.481 | 600.196 | 0.582x | 0.627x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 92.445 | 571.851 | 610.291 | 6.186x | 6.602x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.616 | 6.325 | 9.006 | 10.273x | 14.628x |
| INT8 | SUB | allocating | 7 | NEON | 0.662 | 6.360 | 9.138 | 9.608x | 13.805x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1019.122 | 573.246 | 620.350 | 0.562x | 0.609x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 119.454 | 571.499 | 621.288 | 4.784x | 5.201x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.242 | 6.345 | 7.801 | 26.225x | 32.242x |
| INT8 | SUB | inplace | 7 | NEON | 0.266 | 6.324 | 7.414 | 23.767x | 27.864x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 958.472 | 554.504 | 615.229 | 0.579x | 0.642x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 93.809 | 559.721 | 617.221 | 5.967x | 6.580x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.624 | 6.335 | 7.976 | 10.149x | 12.778x |
| INT8 | MUL | allocating | 7 | NEON | 0.669 | 6.334 | 7.970 | 9.473x | 11.919x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 973.893 | 589.309 | 622.873 | 0.605x | 0.640x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 128.398 | 581.475 | 618.287 | 4.529x | 4.815x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.230 | 6.340 | 6.580 | 27.513x | 28.555x |
| INT8 | MUL | inplace | 7 | NEON | 0.265 | 6.432 | 6.947 | 24.261x | 26.205x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 912.123 | 573.095 | 614.276 | 0.628x | 0.673x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 95.564 | 566.425 | 609.078 | 5.927x | 6.373x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.605 | 7.087 | 8.946 | 11.706x | 14.776x |
| INT8 | DIV | allocating | 7 | NEON | 0.656 | 7.054 | 9.288 | 10.761x | 14.169x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1683.146 | 2900.521 | 2075.264 | 1.723x | 1.233x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 572.759 | 2814.045 | 2080.322 | 4.913x | 3.632x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.227 | 7.077 | 7.764 | 31.238x | 34.270x |
| INT8 | DIV | inplace | 7 | NEON | 0.258 | 6.996 | 7.996 | 27.104x | 30.981x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1620.223 | 2814.673 | 2061.121 | 1.737x | 1.272x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 513.833 | 2852.784 | 2062.142 | 5.552x | 4.013x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.790 | 1.391 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.663 | 0.786 | 1.404 | 1.185x | 2.118x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1978.812 | 168.755 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1013.293 | 1978.445 | 158.244 | 1.952x | 0.156x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.923 | 1.259 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.268 | 0.926 | 1.290 | 3.455x | 4.810x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1886.987 | 138.529 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 896.927 | 1887.120 | 135.191 | 2.104x | 0.151x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.812 | 1.438 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.663 | 0.796 | 1.414 | 1.201x | 2.134x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1996.152 | 172.669 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1016.751 | 1982.601 | 165.239 | 1.950x | 0.163x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.933 | 1.586 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.264 | 0.947 | 1.594 | 3.585x | 6.036x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1906.275 | 145.482 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 899.361 | 1902.351 | 140.955 | 2.115x | 0.157x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.788 | 1.395 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.682 | 0.790 | 1.428 | 1.158x | 2.094x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1984.791 | 158.488 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1014.083 | 1979.385 | 163.944 | 1.952x | 0.162x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.932 | 1.233 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.274 | 0.925 | 1.224 | 3.381x | 4.474x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1900.072 | 147.846 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 897.227 | 1929.645 | 124.828 | 2.151x | 0.139x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.775 | 1.393 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.674 | 0.788 | 1.385 | 1.170x | 2.055x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2188.656 | 358.720 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1310.206 | 2194.250 | 361.330 | 1.675x | 0.276x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.908 | 1.612 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.272 | 0.913 | 1.586 | 3.362x | 5.842x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2100.914 | 350.622 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1192.704 | 2101.282 | 351.370 | 1.762x | 0.295x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.265 | 8.615 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.668 | 6.220 | 8.269 | 9.304x | 12.369x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1356.616 | 1450.043 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 249.260 | 1347.671 | 1442.947 | 5.407x | 5.789x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.254 | 6.924 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.265 | 6.220 | 6.988 | 23.448x | 26.342x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1312.407 | 1427.986 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 188.294 | 1305.373 | 1413.652 | 6.933x | 7.508x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.242 | 8.895 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.666 | 6.255 | 8.992 | 9.396x | 13.507x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1338.236 | 1436.208 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 250.679 | 1378.835 | 1536.681 | 5.500x | 6.130x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.260 | 7.174 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.276 | 6.260 | 8.227 | 22.641x | 29.755x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1276.515 | 1431.468 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 186.309 | 1300.521 | 1493.779 | 6.980x | 8.018x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.202 | 8.548 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.654 | 6.231 | 8.235 | 9.524x | 12.586x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1490.323 | 1454.226 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 259.652 | 1470.501 | 1408.457 | 5.663x | 5.424x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.235 | 6.192 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.259 | 6.220 | 6.409 | 24.027x | 24.758x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1412.435 | 1443.217 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 183.660 | 1370.250 | 1388.902 | 7.461x | 7.562x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.887 | 11.218 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.678 | 6.881 | 8.679 | 10.154x | 12.809x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3803.881 | 2698.850 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 617.023 | 3903.323 | 2694.205 | 6.326x | 4.366x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.874 | 9.229 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.289 | 6.937 | 8.474 | 24.007x | 29.324x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3776.699 | 2657.567 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 521.690 | 6593.505 | 3375.023 | 12.639x | 6.469x |

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
