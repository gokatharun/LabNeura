# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-07T15:39:02.183910+00:00
- Commit: `47369a6`
- CPU: AMD EPYC 9V45 96-Core Processor (x86_64)
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.297 | 0.563 | 1.021 | 1.898x | 3.441x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.293 | 0.564 | 1.019 | 1.923x | 3.476x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 283.434 | 475.387 | 557.696 | 1.677x | 1.968x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 254.430 | 231.796 | 236.570 | 0.911x | 0.930x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.105 | 0.660 | 0.925 | 6.310x | 8.848x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.118 | 0.650 | 0.923 | 5.498x | 7.808x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 202.312 | 184.207 | 192.357 | 0.911x | 0.951x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 205.669 | 175.256 | 185.143 | 0.852x | 0.900x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.304 | 0.556 | 1.026 | 1.828x | 3.373x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.285 | 0.557 | 1.027 | 1.957x | 3.609x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 297.703 | 490.532 | 512.250 | 1.648x | 1.721x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 248.036 | 256.806 | 230.079 | 1.035x | 0.928x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.110 | 0.644 | 0.928 | 5.880x | 8.466x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.124 | 0.643 | 0.924 | 5.204x | 7.474x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 200.744 | 185.536 | 189.503 | 0.924x | 0.944x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 199.683 | 179.152 | 183.519 | 0.897x | 0.919x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.307 | 0.558 | 0.992 | 1.820x | 3.236x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.301 | 0.561 | 0.993 | 1.865x | 3.300x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 297.410 | 490.888 | 511.339 | 1.651x | 1.719x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 235.096 | 508.828 | 519.631 | 2.164x | 2.210x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.114 | 0.645 | 0.907 | 5.646x | 7.943x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.124 | 0.645 | 0.905 | 5.202x | 7.302x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 198.241 | 217.963 | 197.204 | 1.099x | 0.995x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 204.713 | 188.128 | 187.760 | 0.919x | 0.917x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.299 | 0.563 | 0.999 | 1.885x | 3.344x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.285 | 0.561 | 1.002 | 1.967x | 3.517x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 302.508 | 496.137 | 551.306 | 1.640x | 1.822x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 237.199 | 494.864 | 488.116 | 2.086x | 2.058x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.109 | 0.637 | 0.907 | 5.827x | 8.290x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.120 | 0.640 | 0.911 | 5.348x | 7.622x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 198.182 | 189.897 | 194.492 | 0.958x | 0.981x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 208.699 | 175.281 | 185.914 | 0.840x | 0.891x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.305 | 4.120 | 3.900 | 13.513x | 12.794x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.341 | 4.117 | 3.875 | 12.076x | 11.368x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 2284.222 | 287.313 | 289.294 | 0.126x | 0.127x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 62.767 | 280.713 | 240.473 | 4.472x | 3.831x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.108 | 4.166 | 3.374 | 38.591x | 31.251x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.126 | 4.185 | 3.369 | 33.165x | 26.698x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 2250.190 | 264.494 | 310.579 | 0.118x | 0.138x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 51.356 | 308.458 | 264.851 | 6.006x | 5.157x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.307 | 4.059 | 3.867 | 13.211x | 12.585x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.342 | 4.093 | 3.874 | 11.957x | 11.319x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 2411.950 | 292.565 | 251.223 | 0.121x | 0.104x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 63.724 | 278.514 | 237.726 | 4.371x | 3.731x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.113 | 4.168 | 3.361 | 36.829x | 29.700x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.130 | 4.186 | 3.360 | 32.249x | 25.885x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 2448.812 | 275.396 | 233.490 | 0.112x | 0.095x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 51.971 | 270.312 | 234.791 | 5.201x | 4.518x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.305 | 4.105 | 3.881 | 13.464x | 12.731x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.339 | 4.108 | 3.938 | 12.113x | 11.613x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4071.184 | 271.393 | 244.202 | 0.067x | 0.060x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 70.888 | 292.181 | 246.996 | 4.122x | 3.484x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.113 | 4.209 | 3.342 | 37.389x | 29.693x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.128 | 4.243 | 3.347 | 33.050x | 26.071x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4080.490 | 256.339 | 219.994 | 0.063x | 0.054x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 52.568 | 270.603 | 242.535 | 5.148x | 4.614x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.308 | 4.748 | 3.833 | 15.391x | 12.425x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.341 | 4.614 | 3.838 | 13.518x | 11.245x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1367.525 | 884.957 | 1495.378 | 0.647x | 1.093x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 444.909 | 900.852 | 1511.503 | 2.025x | 3.397x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.109 | 4.691 | 3.345 | 43.045x | 30.690x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.129 | 5.207 | 3.338 | 40.240x | 25.793x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1341.821 | 912.387 | 1618.043 | 0.680x | 1.206x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 410.403 | 934.262 | 1503.428 | 2.276x | 3.663x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.584 | 1.016 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.343 | 0.577 | 1.019 | 1.679x | 2.967x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 3427.062 | 216.550 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 145.832 | 3422.555 | 209.639 | 23.469x | 1.438x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.675 | 0.924 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.124 | 0.670 | 0.922 | 5.420x | 7.457x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 3424.191 | 109.872 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 100.942 | 3356.286 | 106.549 | 33.250x | 1.056x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.579 | 1.024 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.336 | 0.568 | 1.020 | 1.692x | 3.040x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 3436.793 | 205.928 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 152.271 | 3626.616 | 207.887 | 23.817x | 1.365x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.660 | 0.927 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.126 | 0.668 | 0.930 | 5.292x | 7.369x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 3352.484 | 112.871 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 101.506 | 3622.607 | 106.793 | 35.689x | 1.052x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.575 | 0.983 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.343 | 0.581 | 0.983 | 1.694x | 2.868x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 3617.426 | 214.146 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 146.456 | 3457.991 | 214.968 | 23.611x | 1.468x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.679 | 0.902 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.126 | 0.667 | 0.900 | 5.312x | 7.162x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 3383.444 | 99.936 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 105.800 | 3365.053 | 104.965 | 31.806x | 0.992x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.568 | 1.001 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.340 | 0.565 | 0.994 | 1.662x | 2.922x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 3611.940 | 218.233 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 161.174 | 3652.657 | 223.876 | 22.663x | 1.389x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.653 | 0.905 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.126 | 0.654 | 0.913 | 5.204x | 7.271x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 3546.241 | 150.994 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 106.260 | 3715.099 | 141.543 | 34.962x | 1.332x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 4.217 | 4.169 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.353 | 4.248 | 4.026 | 12.029x | 11.401x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 779.941 | 823.184 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 137.704 | 728.485 | 728.252 | 5.290x | 5.289x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 4.302 | 3.507 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.128 | 4.269 | 3.494 | 33.405x | 27.337x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 808.551 | 757.178 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 100.914 | 649.024 | 635.898 | 6.431x | 6.301x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 4.164 | 4.087 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.352 | 4.158 | 3.991 | 11.814x | 11.340x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 725.201 | 703.757 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 139.016 | 666.768 | 714.624 | 4.796x | 5.141x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 4.267 | 3.480 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.131 | 4.277 | 3.472 | 32.607x | 26.469x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 730.028 | 659.019 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 98.915 | 671.037 | 643.389 | 6.784x | 6.504x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 4.216 | 3.984 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.351 | 4.147 | 4.006 | 11.824x | 11.420x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 706.652 | 732.142 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 152.011 | 692.136 | 751.612 | 4.553x | 4.944x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 4.263 | 3.413 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.127 | 4.223 | 3.401 | 33.144x | 26.692x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 748.496 | 684.028 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 103.121 | 734.067 | 672.449 | 7.118x | 6.521x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 4.629 | 4.013 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.342 | 4.651 | 3.866 | 13.612x | 11.315x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 1879.763 | 1971.522 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 466.132 | 1845.688 | 1962.851 | 3.960x | 4.211x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 4.724 | 3.336 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.126 | 4.707 | 3.328 | 37.415x | 26.452x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 1781.513 | 1883.271 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 390.937 | 1841.569 | 1884.128 | 4.711x | 4.820x |

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
