# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-08T05:14:44.272924+00:00
- Commit: `665991b`
- CPU: AMD EPYC 9V74 80-Core Processor (x86_64)
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.565 | 0.816 | 1.411 | 1.443x | 2.496x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.459 | 0.773 | 1.404 | 1.684x | 3.060x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 404.958 | 658.902 | 648.837 | 1.627x | 1.602x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 338.704 | 650.807 | 635.729 | 1.921x | 1.877x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.154 | 0.890 | 1.264 | 5.760x | 8.182x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.183 | 0.883 | 1.264 | 4.814x | 6.886x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 233.450 | 223.863 | 225.221 | 0.959x | 0.965x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 320.606 | 231.275 | 225.957 | 0.721x | 0.705x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.444 | 0.769 | 1.413 | 1.731x | 3.183x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.466 | 0.771 | 1.415 | 1.653x | 3.035x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 383.218 | 628.348 | 643.343 | 1.640x | 1.679x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 346.480 | 610.289 | 636.262 | 1.761x | 1.836x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.155 | 0.886 | 1.286 | 5.712x | 8.292x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.181 | 0.906 | 1.281 | 5.005x | 7.073x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 237.316 | 229.755 | 228.077 | 0.968x | 0.961x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 327.820 | 223.157 | 226.280 | 0.681x | 0.690x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.447 | 0.771 | 1.402 | 1.724x | 3.134x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.462 | 0.771 | 1.384 | 1.668x | 2.994x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 369.192 | 669.985 | 633.701 | 1.815x | 1.716x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 332.659 | 291.365 | 272.815 | 0.876x | 0.820x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.158 | 0.884 | 1.225 | 5.598x | 7.759x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.183 | 0.895 | 1.239 | 4.894x | 6.776x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 238.776 | 229.730 | 223.914 | 0.962x | 0.938x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 314.875 | 220.103 | 235.639 | 0.699x | 0.748x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.453 | 0.764 | 1.428 | 1.685x | 3.149x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.469 | 0.761 | 1.408 | 1.623x | 3.003x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 412.029 | 638.689 | 642.642 | 1.550x | 1.560x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 309.029 | 598.654 | 609.122 | 1.937x | 1.971x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.156 | 0.870 | 1.249 | 5.561x | 7.984x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.186 | 0.872 | 1.245 | 4.695x | 6.704x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 240.546 | 217.379 | 223.613 | 0.904x | 0.930x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 330.278 | 231.137 | 225.095 | 0.700x | 0.682x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.473 | 6.069 | 6.320 | 12.833x | 13.364x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.511 | 6.090 | 6.252 | 11.924x | 12.241x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 2681.874 | 350.140 | 329.773 | 0.131x | 0.123x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 78.393 | 369.423 | 359.612 | 4.712x | 4.587x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.160 | 6.183 | 4.993 | 38.584x | 31.161x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.184 | 6.131 | 4.985 | 33.271x | 27.053x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 2659.385 | 336.758 | 307.144 | 0.127x | 0.115x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 58.471 | 335.710 | 308.771 | 5.741x | 5.281x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.464 | 6.009 | 6.230 | 12.958x | 13.434x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.515 | 6.035 | 6.268 | 11.710x | 12.161x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 2881.284 | 367.599 | 319.614 | 0.128x | 0.111x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 79.551 | 370.232 | 320.574 | 4.654x | 4.030x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.161 | 6.190 | 5.061 | 38.531x | 31.505x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.183 | 6.171 | 5.123 | 33.707x | 27.982x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 2854.216 | 393.954 | 346.225 | 0.138x | 0.121x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 58.821 | 407.233 | 342.326 | 6.923x | 5.820x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.459 | 6.047 | 6.261 | 13.161x | 13.625x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.540 | 6.037 | 6.227 | 11.179x | 11.531x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4348.043 | 357.814 | 314.605 | 0.082x | 0.072x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 87.716 | 353.059 | 312.505 | 4.025x | 3.563x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.165 | 6.192 | 4.960 | 37.472x | 30.016x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.202 | 6.151 | 4.986 | 30.406x | 24.649x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4309.220 | 344.661 | 327.163 | 0.080x | 0.076x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 60.453 | 335.739 | 319.953 | 5.554x | 5.293x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.472 | 6.800 | 6.204 | 14.405x | 13.142x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.513 | 6.863 | 6.279 | 13.379x | 12.240x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1686.043 | 1263.528 | 1872.736 | 0.749x | 1.111x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 586.575 | 1244.914 | 1871.545 | 2.122x | 3.191x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.162 | 6.912 | 5.026 | 42.761x | 31.095x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.186 | 6.923 | 5.013 | 37.242x | 26.970x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1645.714 | 1236.044 | 1857.199 | 0.751x | 1.129x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 547.843 | 1251.569 | 1861.055 | 2.285x | 3.397x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.804 | 1.409 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.506 | 0.813 | 1.417 | 1.606x | 2.798x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 5617.590 | 264.796 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 185.979 | 5572.648 | 276.201 | 29.964x | 1.485x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.936 | 1.259 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.192 | 0.927 | 1.260 | 4.835x | 6.575x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 5503.153 | 145.642 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 119.956 | 5492.187 | 147.021 | 45.785x | 1.226x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.809 | 1.424 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.508 | 0.811 | 1.411 | 1.595x | 2.777x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 5845.599 | 265.108 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 182.168 | 5852.897 | 276.013 | 32.129x | 1.515x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.926 | 1.277 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.183 | 0.934 | 1.290 | 5.092x | 7.030x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 5802.494 | 151.924 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 119.179 | 5788.935 | 164.729 | 48.573x | 1.382x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.810 | 1.391 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.514 | 0.812 | 1.391 | 1.580x | 2.706x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 5753.621 | 538.468 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 181.307 | 5576.543 | 263.718 | 30.757x | 1.455x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.931 | 1.224 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.186 | 0.939 | 1.217 | 5.044x | 6.541x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 5536.706 | 132.568 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 118.951 | 5497.999 | 127.304 | 46.221x | 1.070x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.790 | 1.392 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.529 | 0.794 | 1.408 | 1.502x | 2.663x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 5572.990 | 264.489 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 210.420 | 5584.442 | 272.410 | 26.540x | 1.295x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.922 | 1.237 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.187 | 0.919 | 1.236 | 4.912x | 6.605x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 5494.922 | 142.903 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 125.886 | 5491.595 | 149.482 | 43.623x | 1.187x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 5.973 | 6.190 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.510 | 5.969 | 6.198 | 11.710x | 12.159x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1056.413 | 989.897 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 168.296 | 927.856 | 1010.209 | 5.513x | 6.003x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.076 | 4.995 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.183 | 6.072 | 5.021 | 33.091x | 27.365x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 891.368 | 886.529 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 122.785 | 860.463 | 901.238 | 7.008x | 7.340x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.014 | 6.306 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.511 | 5.929 | 6.331 | 11.614x | 12.400x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 890.635 | 989.225 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 167.772 | 915.640 | 990.773 | 5.458x | 5.905x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.101 | 5.046 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.185 | 6.112 | 5.061 | 32.953x | 27.287x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 844.583 | 892.193 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 122.083 | 860.557 | 895.521 | 7.049x | 7.335x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 5.949 | 6.271 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.519 | 5.967 | 6.326 | 11.498x | 12.189x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 959.779 | 1001.948 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 184.398 | 963.040 | 1012.465 | 5.223x | 5.491x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.094 | 4.941 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.188 | 6.087 | 4.964 | 32.396x | 26.424x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 876.944 | 984.380 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 123.066 | 863.308 | 944.485 | 7.015x | 7.675x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.792 | 6.188 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.524 | 6.827 | 6.190 | 13.018x | 11.805x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 2444.576 | 2417.712 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 600.642 | 2439.878 | 2452.280 | 4.062x | 4.083x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.817 | 4.958 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.190 | 6.901 | 4.980 | 36.381x | 26.253x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 2562.636 | 2362.790 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 513.025 | 2387.622 | 2334.575 | 4.654x | 4.551x |

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
