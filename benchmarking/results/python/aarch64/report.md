# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-14T10:20:55.358751+00:00
- Commit: `849e1ea`
- CPU: aarch64 (aarch64)
- OS: Linux-7.0.0-1017-azure-aarch64-with-glibc2.39
- Python / NumPy / PyTorch: 3.11.18 / 2.4.6 / 2.15.0+cu132
- LabNeura: 0.4.2
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.623 | 0.793 | 1.394 | 1.272x | 2.238x |
| FP32 | ADD | allocating | 7 | NEON | 0.621 | 0.795 | 1.386 | 1.279x | 2.230x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 462.016 | 227.124 | 276.007 | 0.492x | 0.597x |
| FP32 | ADD | allocating | 1,000,000 | NEON | 407.015 | 500.622 | 291.452 | 1.230x | 0.716x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.238 | 0.926 | 1.264 | 3.893x | 5.317x |
| FP32 | ADD | inplace | 7 | NEON | 0.266 | 0.911 | 1.272 | 3.432x | 4.789x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 339.759 | 229.848 | 252.449 | 0.677x | 0.743x |
| FP32 | ADD | inplace | 1,000,000 | NEON | 342.791 | 243.831 | 253.178 | 0.711x | 0.739x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.620 | 0.801 | 1.392 | 1.293x | 2.246x |
| FP32 | SUB | allocating | 7 | NEON | 0.621 | 0.789 | 1.404 | 1.271x | 2.262x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 461.580 | 551.872 | 300.826 | 1.196x | 0.652x |
| FP32 | SUB | allocating | 1,000,000 | NEON | 376.183 | 501.785 | 271.771 | 1.334x | 0.722x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.244 | 0.933 | 1.328 | 3.829x | 5.451x |
| FP32 | SUB | inplace | 7 | NEON | 0.260 | 0.931 | 1.299 | 3.574x | 4.986x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 337.853 | 236.731 | 261.684 | 0.701x | 0.775x |
| FP32 | SUB | inplace | 1,000,000 | NEON | 337.287 | 267.272 | 252.694 | 0.792x | 0.749x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.612 | 0.809 | 1.393 | 1.322x | 2.277x |
| FP32 | MUL | allocating | 7 | NEON | 0.613 | 0.793 | 1.398 | 1.294x | 2.280x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 477.117 | 513.240 | 294.262 | 1.076x | 0.617x |
| FP32 | MUL | allocating | 1,000,000 | NEON | 372.285 | 498.819 | 284.128 | 1.340x | 0.763x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.236 | 0.946 | 1.247 | 4.006x | 5.279x |
| FP32 | MUL | inplace | 7 | NEON | 0.266 | 0.942 | 1.240 | 3.546x | 4.665x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 336.461 | 219.166 | 242.991 | 0.651x | 0.722x |
| FP32 | MUL | inplace | 1,000,000 | NEON | 339.175 | 309.273 | 233.908 | 0.912x | 0.690x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.613 | 0.784 | 1.387 | 1.279x | 2.263x |
| FP32 | DIV | allocating | 7 | NEON | 0.614 | 0.787 | 1.377 | 1.282x | 2.242x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 675.964 | 740.406 | 492.188 | 1.095x | 0.728x |
| FP32 | DIV | allocating | 1,000,000 | NEON | 496.454 | 721.750 | 479.339 | 1.454x | 0.966x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.234 | 0.922 | 1.584 | 3.935x | 6.761x |
| FP32 | DIV | inplace | 7 | NEON | 0.264 | 0.915 | 1.521 | 3.469x | 5.763x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 466.953 | 459.856 | 478.695 | 0.985x | 1.025x |
| FP32 | DIV | inplace | 1,000,000 | NEON | 467.513 | 461.621 | 478.627 | 0.987x | 1.024x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.631 | 6.385 | 8.481 | 10.113x | 13.433x |
| INT8 | ADD | allocating | 7 | NEON | 0.676 | 6.328 | 8.339 | 9.367x | 12.343x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 1014.709 | 568.880 | 608.694 | 0.561x | 0.600x |
| INT8 | ADD | allocating | 1,000,000 | NEON | 113.317 | 566.466 | 625.541 | 4.999x | 5.520x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.242 | 6.440 | 6.915 | 26.624x | 28.587x |
| INT8 | ADD | inplace | 7 | NEON | 0.273 | 6.423 | 6.742 | 23.561x | 24.731x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 957.357 | 560.424 | 610.264 | 0.585x | 0.637x |
| INT8 | ADD | inplace | 1,000,000 | NEON | 87.942 | 548.766 | 601.140 | 6.240x | 6.836x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.622 | 6.355 | 10.706 | 10.221x | 17.220x |
| INT8 | SUB | allocating | 7 | NEON | 0.676 | 6.326 | 8.745 | 9.358x | 12.937x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 1016.190 | 566.046 | 626.822 | 0.557x | 0.617x |
| INT8 | SUB | allocating | 1,000,000 | NEON | 114.421 | 562.828 | 625.260 | 4.919x | 5.465x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.234 | 6.382 | 7.921 | 27.231x | 33.796x |
| INT8 | SUB | inplace | 7 | NEON | 0.279 | 6.383 | 7.210 | 22.862x | 25.827x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 956.282 | 552.545 | 633.496 | 0.578x | 0.662x |
| INT8 | SUB | inplace | 1,000,000 | NEON | 91.964 | 546.938 | 600.202 | 5.947x | 6.526x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.639 | 6.406 | 7.835 | 10.028x | 12.264x |
| INT8 | MUL | allocating | 7 | NEON | 0.690 | 6.337 | 8.976 | 9.179x | 13.001x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 973.040 | 551.632 | 599.599 | 0.567x | 0.616x |
| INT8 | MUL | allocating | 1,000,000 | NEON | 122.581 | 576.585 | 599.263 | 4.704x | 4.889x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.240 | 6.605 | 6.140 | 27.556x | 25.617x |
| INT8 | MUL | inplace | 7 | NEON | 0.274 | 6.395 | 6.416 | 23.377x | 23.452x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 911.953 | 543.508 | 588.300 | 0.596x | 0.645x |
| INT8 | MUL | inplace | 1,000,000 | NEON | 89.081 | 538.979 | 581.891 | 6.050x | 6.532x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.630 | 7.007 | 9.248 | 11.115x | 14.671x |
| INT8 | DIV | allocating | 7 | NEON | 0.679 | 7.018 | 8.820 | 10.333x | 12.987x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1676.025 | 2877.462 | 2068.372 | 1.717x | 1.234x |
| INT8 | DIV | allocating | 1,000,000 | NEON | 573.035 | 2873.076 | 2076.667 | 5.014x | 3.624x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.240 | 7.053 | 7.418 | 29.350x | 30.869x |
| INT8 | DIV | inplace | 7 | NEON | 0.276 | 7.026 | 7.581 | 25.459x | 27.469x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1615.787 | 2898.957 | 2060.466 | 1.794x | 1.275x |
| INT8 | DIV | inplace | 1,000,000 | NEON | 513.080 | 2838.708 | 2052.548 | 5.533x | 4.000x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 0.806 | 1.410 | —x | —x |
| FP16 | ADD | allocating | 7 | NEON | 0.670 | 0.807 | 1.421 | 1.204x | 2.120x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 1961.930 | 150.087 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | NEON | 1009.987 | 1959.956 | 154.622 | 1.941x | 0.153x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 0.924 | 1.252 | —x | —x |
| FP16 | ADD | inplace | 7 | NEON | 0.273 | 0.932 | 1.245 | 3.412x | 4.556x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 1890.401 | 135.982 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | NEON | 901.359 | 1884.474 | 138.730 | 2.091x | 0.154x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 0.817 | 1.398 | —x | —x |
| FP16 | SUB | allocating | 7 | NEON | 0.670 | 0.807 | 1.400 | 1.206x | 2.091x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 1988.685 | 150.100 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | NEON | 1008.101 | 1987.082 | 151.176 | 1.971x | 0.150x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 0.936 | 1.309 | —x | —x |
| FP16 | SUB | inplace | 7 | NEON | 0.271 | 0.939 | 1.347 | 3.467x | 4.975x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 1900.332 | 131.923 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | NEON | 900.072 | 1902.980 | 217.231 | 2.114x | 0.241x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 0.809 | 1.379 | —x | —x |
| FP16 | MUL | allocating | 7 | NEON | 0.681 | 0.800 | 1.378 | 1.174x | 2.023x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 1976.928 | 184.979 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | NEON | 1008.851 | 1972.915 | 169.566 | 1.956x | 0.168x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 0.930 | 1.233 | —x | —x |
| FP16 | MUL | inplace | 7 | NEON | 0.275 | 0.931 | 1.259 | 3.390x | 4.586x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 1899.986 | 141.638 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | NEON | 899.562 | 1899.753 | 140.121 | 2.112x | 0.156x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 0.804 | 1.381 | —x | —x |
| FP16 | DIV | allocating | 7 | NEON | 0.673 | 0.797 | 1.375 | 1.183x | 2.043x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 2185.901 | 361.330 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | NEON | 1306.088 | 2176.762 | 366.138 | 1.667x | 0.280x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 0.929 | 1.528 | —x | —x |
| FP16 | DIV | inplace | 7 | NEON | 0.271 | 0.935 | 1.543 | 3.451x | 5.698x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 2103.112 | 349.668 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | NEON | 1194.629 | 2101.509 | 357.230 | 1.759x | 0.299x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 6.247 | 8.089 | —x | —x |
| INT16 | ADD | allocating | 7 | NEON | 0.676 | 6.250 | 8.330 | 9.249x | 12.325x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1380.378 | 1472.713 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | NEON | 235.658 | 1327.095 | 1478.391 | 5.631x | 6.273x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 6.282 | 6.925 | —x | —x |
| INT16 | ADD | inplace | 7 | NEON | 0.269 | 6.298 | 6.825 | 23.404x | 25.364x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1329.509 | 1480.934 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | NEON | 187.302 | 1277.461 | 1404.215 | 6.820x | 7.497x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 6.221 | 9.801 | —x | —x |
| INT16 | SUB | allocating | 7 | NEON | 0.667 | 6.275 | 8.791 | 9.412x | 13.186x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1300.224 | 1476.153 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | NEON | 229.156 | 1358.213 | 1484.392 | 5.927x | 6.478x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 6.319 | 6.922 | —x | —x |
| INT16 | SUB | inplace | 7 | NEON | 0.268 | 6.308 | 7.387 | 23.566x | 27.598x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1297.608 | 1474.419 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | NEON | 184.611 | 1308.821 | 1471.573 | 7.090x | 7.971x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 6.252 | 7.803 | —x | —x |
| INT16 | MUL | allocating | 7 | NEON | 0.681 | 6.282 | 8.028 | 9.226x | 11.790x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1431.486 | 1445.236 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | NEON | 259.003 | 1407.524 | 1454.556 | 5.434x | 5.616x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 6.296 | 6.255 | —x | —x |
| INT16 | MUL | inplace | 7 | NEON | 0.267 | 6.288 | 6.349 | 23.577x | 23.805x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1393.941 | 1385.607 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | NEON | 186.080 | 1392.029 | 1494.869 | 7.481x | 8.033x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 6.890 | 9.030 | —x | —x |
| INT16 | DIV | allocating | 7 | NEON | 0.681 | 6.874 | 9.368 | 10.095x | 13.758x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 3929.257 | 2726.902 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | NEON | 616.335 | 4119.776 | 2755.815 | 6.684x | 4.471x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 6.914 | 7.162 | —x | —x |
| INT16 | DIV | inplace | 7 | NEON | 0.270 | 6.942 | 7.684 | 25.751x | 28.502x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 3971.183 | 2689.972 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | NEON | 505.363 | 3877.739 | 2693.588 | 7.673x | 5.330x |

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
