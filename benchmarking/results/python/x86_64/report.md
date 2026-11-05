# LabNeura Python/API Benchmark

- Timestamp (UTC): 2026-11-05T19:33:23.193014+00:00
- Commit: `e8dca86`
- CPU: AMD EPYC 7763 64-Core Processor (x86_64)
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
| FP32 | ADD | allocating | 7 | GENERIC | 0.478 | 1.027 | 1.908 | 2.150x | 3.995x |
| FP32 | ADD | allocating | 7 | AVX2 | 0.511 | 1.034 | 1.864 | 2.022x | 3.647x |
| FP32 | ADD | allocating | 1,000,000 | GENERIC | 489.542 | 266.542 | 333.235 | 0.544x | 0.681x |
| FP32 | ADD | allocating | 1,000,000 | AVX2 | 476.239 | 268.200 | 829.243 | 0.563x | 1.741x |
| FP32 | ADD | inplace | 7 | GENERIC | 0.218 | 1.149 | 1.684 | 5.270x | 7.724x |
| FP32 | ADD | inplace | 7 | AVX2 | 0.219 | 1.152 | 1.666 | 5.264x | 7.613x |
| FP32 | ADD | inplace | 1,000,000 | GENERIC | 315.629 | 207.606 | 354.928 | 0.658x | 1.125x |
| FP32 | ADD | inplace | 1,000,000 | AVX2 | 316.794 | 192.710 | 320.250 | 0.608x | 1.011x |
| FP32 | SUB | allocating | 7 | GENERIC | 0.487 | 1.020 | 1.854 | 2.096x | 3.811x |
| FP32 | SUB | allocating | 7 | AVX2 | 0.490 | 1.022 | 1.850 | 2.088x | 3.780x |
| FP32 | SUB | allocating | 1,000,000 | GENERIC | 530.223 | 257.637 | 368.669 | 0.486x | 0.695x |
| FP32 | SUB | allocating | 1,000,000 | AVX2 | 474.481 | 276.984 | 358.838 | 0.584x | 0.756x |
| FP32 | SUB | inplace | 7 | GENERIC | 0.216 | 1.142 | 1.672 | 5.295x | 7.753x |
| FP32 | SUB | inplace | 7 | AVX2 | 0.221 | 1.132 | 1.677 | 5.117x | 7.581x |
| FP32 | SUB | inplace | 1,000,000 | GENERIC | 308.365 | 190.070 | 264.110 | 0.616x | 0.856x |
| FP32 | SUB | inplace | 1,000,000 | AVX2 | 318.780 | 192.976 | 280.921 | 0.605x | 0.881x |
| FP32 | MUL | allocating | 7 | GENERIC | 0.498 | 1.019 | 1.849 | 2.045x | 3.713x |
| FP32 | MUL | allocating | 7 | AVX2 | 0.502 | 1.027 | 1.840 | 2.045x | 3.667x |
| FP32 | MUL | allocating | 1,000,000 | GENERIC | 499.115 | 227.744 | 343.807 | 0.456x | 0.689x |
| FP32 | MUL | allocating | 1,000,000 | AVX2 | 517.254 | 266.059 | 378.540 | 0.514x | 0.732x |
| FP32 | MUL | inplace | 7 | GENERIC | 0.220 | 1.127 | 1.629 | 5.124x | 7.407x |
| FP32 | MUL | inplace | 7 | AVX2 | 0.224 | 1.145 | 1.633 | 5.109x | 7.286x |
| FP32 | MUL | inplace | 1,000,000 | GENERIC | 311.296 | 206.677 | 290.194 | 0.664x | 0.932x |
| FP32 | MUL | inplace | 1,000,000 | AVX2 | 328.083 | 197.051 | 266.727 | 0.601x | 0.813x |
| FP32 | DIV | allocating | 7 | GENERIC | 0.492 | 1.012 | 1.834 | 2.057x | 3.725x |
| FP32 | DIV | allocating | 7 | AVX2 | 0.493 | 1.010 | 1.829 | 2.047x | 3.708x |
| FP32 | DIV | allocating | 1,000,000 | GENERIC | 532.142 | 223.208 | 402.626 | 0.419x | 0.757x |
| FP32 | DIV | allocating | 1,000,000 | AVX2 | 559.889 | 242.356 | 337.683 | 0.433x | 0.603x |
| FP32 | DIV | inplace | 7 | GENERIC | 0.216 | 1.131 | 1.651 | 5.228x | 7.631x |
| FP32 | DIV | inplace | 7 | AVX2 | 0.220 | 1.138 | 1.625 | 5.183x | 7.400x |
| FP32 | DIV | inplace | 1,000,000 | GENERIC | 343.204 | 217.439 | 295.088 | 0.634x | 0.860x |
| FP32 | DIV | inplace | 1,000,000 | AVX2 | 361.195 | 210.311 | 288.015 | 0.582x | 0.797x |
| INT8 | ADD | allocating | 7 | GENERIC | 0.496 | 8.023 | 8.508 | 16.178x | 17.155x |
| INT8 | ADD | allocating | 7 | AVX2 | 0.507 | 7.999 | 8.714 | 15.772x | 17.183x |
| INT8 | ADD | allocating | 1,000,000 | GENERIC | 3320.271 | 466.634 | 455.387 | 0.141x | 0.137x |
| INT8 | ADD | allocating | 1,000,000 | AVX2 | 110.774 | 454.725 | 449.298 | 4.105x | 4.056x |
| INT8 | ADD | inplace | 7 | GENERIC | 0.218 | 8.076 | 6.723 | 37.114x | 30.897x |
| INT8 | ADD | inplace | 7 | AVX2 | 0.230 | 8.021 | 6.819 | 34.836x | 29.615x |
| INT8 | ADD | inplace | 1,000,000 | GENERIC | 3254.786 | 461.163 | 440.143 | 0.142x | 0.135x |
| INT8 | ADD | inplace | 1,000,000 | AVX2 | 75.248 | 458.818 | 450.624 | 6.097x | 5.989x |
| INT8 | SUB | allocating | 7 | GENERIC | 0.499 | 8.025 | 8.494 | 16.091x | 17.032x |
| INT8 | SUB | allocating | 7 | AVX2 | 0.503 | 8.087 | 8.571 | 16.066x | 17.027x |
| INT8 | SUB | allocating | 1,000,000 | GENERIC | 3237.130 | 464.704 | 453.901 | 0.144x | 0.140x |
| INT8 | SUB | allocating | 1,000,000 | AVX2 | 113.844 | 470.064 | 457.866 | 4.129x | 4.022x |
| INT8 | SUB | inplace | 7 | GENERIC | 0.236 | 8.033 | 6.835 | 34.034x | 28.956x |
| INT8 | SUB | inplace | 7 | AVX2 | 0.233 | 8.192 | 6.754 | 35.185x | 29.008x |
| INT8 | SUB | inplace | 1,000,000 | GENERIC | 3181.826 | 469.347 | 456.280 | 0.148x | 0.143x |
| INT8 | SUB | inplace | 1,000,000 | AVX2 | 75.162 | 477.644 | 452.597 | 6.355x | 6.022x |
| INT8 | MUL | allocating | 7 | GENERIC | 0.497 | 8.036 | 8.782 | 16.177x | 17.678x |
| INT8 | MUL | allocating | 7 | AVX2 | 0.503 | 8.380 | 8.376 | 16.649x | 16.640x |
| INT8 | MUL | allocating | 1,000,000 | GENERIC | 4828.235 | 461.497 | 450.428 | 0.096x | 0.093x |
| INT8 | MUL | allocating | 1,000,000 | AVX2 | 154.115 | 474.530 | 445.969 | 3.079x | 2.894x |
| INT8 | MUL | inplace | 7 | GENERIC | 0.228 | 7.962 | 6.695 | 34.977x | 29.410x |
| INT8 | MUL | inplace | 7 | AVX2 | 0.235 | 8.021 | 6.804 | 34.154x | 28.975x |
| INT8 | MUL | inplace | 1,000,000 | GENERIC | 4761.461 | 467.362 | 455.235 | 0.098x | 0.096x |
| INT8 | MUL | inplace | 1,000,000 | AVX2 | 108.853 | 490.076 | 427.418 | 4.502x | 3.927x |
| INT8 | DIV | allocating | 7 | GENERIC | 0.504 | 9.202 | 8.180 | 18.268x | 16.239x |
| INT8 | DIV | allocating | 7 | AVX2 | 0.508 | 9.245 | 8.369 | 18.214x | 16.488x |
| INT8 | DIV | allocating | 1,000,000 | GENERIC | 1937.823 | 1429.240 | 2191.931 | 0.738x | 1.131x |
| INT8 | DIV | allocating | 1,000,000 | AVX2 | 765.164 | 1482.034 | 2199.737 | 1.937x | 2.875x |
| INT8 | DIV | inplace | 7 | GENERIC | 0.229 | 9.081 | 6.658 | 39.580x | 29.020x |
| INT8 | DIV | inplace | 7 | AVX2 | 0.241 | 9.140 | 6.568 | 37.981x | 27.291x |
| INT8 | DIV | inplace | 1,000,000 | GENERIC | 1877.073 | 1482.862 | 2201.304 | 0.790x | 1.173x |
| INT8 | DIV | inplace | 1,000,000 | AVX2 | 705.803 | 1448.832 | 2193.615 | 2.053x | 3.108x |
| FP16 | ADD | allocating | 7 | GENERIC | — | 1.058 | 1.899 | —x | —x |
| FP16 | ADD | allocating | 7 | AVX2 | 0.554 | 1.066 | 1.868 | 1.924x | 3.371x |
| FP16 | ADD | allocating | 1,000,000 | GENERIC | — | 6710.412 | 370.198 | —x | —x |
| FP16 | ADD | allocating | 1,000,000 | AVX2 | 9545.347 | 6600.586 | 246.298 | 0.691x | 0.026x |
| FP16 | ADD | inplace | 7 | GENERIC | — | 1.184 | 1.662 | —x | —x |
| FP16 | ADD | inplace | 7 | AVX2 | 0.245 | 1.189 | 1.681 | 4.846x | 6.854x |
| FP16 | ADD | inplace | 1,000,000 | GENERIC | — | 6620.118 | 255.274 | —x | —x |
| FP16 | ADD | inplace | 1,000,000 | AVX2 | 9389.785 | 6582.685 | 242.942 | 0.701x | 0.026x |
| FP16 | SUB | allocating | 7 | GENERIC | — | 1.070 | 1.851 | —x | —x |
| FP16 | SUB | allocating | 7 | AVX2 | 0.530 | 1.071 | 1.872 | 2.019x | 3.531x |
| FP16 | SUB | allocating | 1,000,000 | GENERIC | — | 7027.479 | 365.893 | —x | —x |
| FP16 | SUB | allocating | 1,000,000 | AVX2 | 10117.415 | 6898.060 | 245.049 | 0.682x | 0.024x |
| FP16 | SUB | inplace | 7 | GENERIC | — | 1.194 | 1.671 | —x | —x |
| FP16 | SUB | inplace | 7 | AVX2 | 0.244 | 1.192 | 1.668 | 4.875x | 6.823x |
| FP16 | SUB | inplace | 1,000,000 | GENERIC | — | 6938.012 | 258.331 | —x | —x |
| FP16 | SUB | inplace | 1,000,000 | AVX2 | 9955.616 | 6899.703 | 241.834 | 0.693x | 0.024x |
| FP16 | MUL | allocating | 7 | GENERIC | — | 1.074 | 1.845 | —x | —x |
| FP16 | MUL | allocating | 7 | AVX2 | 0.533 | 1.074 | 1.848 | 2.017x | 3.471x |
| FP16 | MUL | allocating | 1,000,000 | GENERIC | — | 6712.173 | 333.432 | —x | —x |
| FP16 | MUL | allocating | 1,000,000 | AVX2 | 9939.808 | 6699.208 | 341.165 | 0.674x | 0.034x |
| FP16 | MUL | inplace | 7 | GENERIC | — | 1.188 | 1.637 | —x | —x |
| FP16 | MUL | inplace | 7 | AVX2 | 0.242 | 1.180 | 1.617 | 4.877x | 6.686x |
| FP16 | MUL | inplace | 1,000,000 | GENERIC | — | 6599.934 | 227.590 | —x | —x |
| FP16 | MUL | inplace | 1,000,000 | AVX2 | 9819.561 | 6589.464 | 215.421 | 0.671x | 0.022x |
| FP16 | DIV | allocating | 7 | GENERIC | — | 1.055 | 1.856 | —x | —x |
| FP16 | DIV | allocating | 7 | AVX2 | 0.546 | 1.059 | 1.834 | 1.940x | 3.360x |
| FP16 | DIV | allocating | 1,000,000 | GENERIC | — | 6720.957 | 342.506 | —x | —x |
| FP16 | DIV | allocating | 1,000,000 | AVX2 | 10882.499 | 6698.479 | 335.340 | 0.616x | 0.031x |
| FP16 | DIV | inplace | 7 | GENERIC | — | 1.174 | 1.617 | —x | —x |
| FP16 | DIV | inplace | 7 | AVX2 | 0.249 | 1.172 | 1.671 | 4.700x | 6.703x |
| FP16 | DIV | inplace | 1,000,000 | GENERIC | — | 6620.427 | 212.515 | —x | —x |
| FP16 | DIV | inplace | 1,000,000 | AVX2 | 10724.448 | 6593.945 | 206.232 | 0.615x | 0.019x |
| INT16 | ADD | allocating | 7 | GENERIC | — | 8.012 | 8.430 | —x | —x |
| INT16 | ADD | allocating | 7 | AVX2 | 0.510 | 7.856 | 8.431 | 15.412x | 16.539x |
| INT16 | ADD | allocating | 1,000,000 | GENERIC | — | 1268.877 | 1547.103 | —x | —x |
| INT16 | ADD | allocating | 1,000,000 | AVX2 | 226.522 | 1113.309 | 1407.051 | 4.915x | 6.212x |
| INT16 | ADD | inplace | 7 | GENERIC | — | 7.868 | 6.731 | —x | —x |
| INT16 | ADD | inplace | 7 | AVX2 | 0.229 | 7.884 | 6.639 | 34.421x | 28.986x |
| INT16 | ADD | inplace | 1,000,000 | GENERIC | — | 1123.646 | 1395.296 | —x | —x |
| INT16 | ADD | inplace | 1,000,000 | AVX2 | 145.658 | 1183.097 | 1368.211 | 8.122x | 9.393x |
| INT16 | SUB | allocating | 7 | GENERIC | — | 7.949 | 8.403 | —x | —x |
| INT16 | SUB | allocating | 7 | AVX2 | 0.514 | 7.988 | 8.533 | 15.542x | 16.603x |
| INT16 | SUB | allocating | 1,000,000 | GENERIC | — | 1178.542 | 1491.496 | —x | —x |
| INT16 | SUB | allocating | 1,000,000 | AVX2 | 225.436 | 1145.571 | 1474.509 | 5.082x | 6.541x |
| INT16 | SUB | inplace | 7 | GENERIC | — | 7.839 | 6.649 | —x | —x |
| INT16 | SUB | inplace | 7 | AVX2 | 0.252 | 8.088 | 6.701 | 32.111x | 26.606x |
| INT16 | SUB | inplace | 1,000,000 | GENERIC | — | 1081.604 | 1378.619 | —x | —x |
| INT16 | SUB | inplace | 1,000,000 | AVX2 | 141.029 | 1055.271 | 1321.722 | 7.483x | 9.372x |
| INT16 | MUL | allocating | 7 | GENERIC | — | 8.119 | 8.424 | —x | —x |
| INT16 | MUL | allocating | 7 | AVX2 | 0.518 | 7.971 | 8.381 | 15.389x | 16.179x |
| INT16 | MUL | allocating | 1,000,000 | GENERIC | — | 1243.835 | 1503.720 | —x | —x |
| INT16 | MUL | allocating | 1,000,000 | AVX2 | 304.875 | 1310.123 | 1388.441 | 4.297x | 4.554x |
| INT16 | MUL | inplace | 7 | GENERIC | — | 8.072 | 6.592 | —x | —x |
| INT16 | MUL | inplace | 7 | AVX2 | 0.236 | 7.941 | 6.577 | 33.641x | 27.864x |
| INT16 | MUL | inplace | 1,000,000 | GENERIC | — | 1127.462 | 1445.116 | —x | —x |
| INT16 | MUL | inplace | 1,000,000 | AVX2 | 196.677 | 1082.537 | 1416.710 | 5.504x | 7.203x |
| INT16 | DIV | allocating | 7 | GENERIC | — | 9.177 | 8.306 | —x | —x |
| INT16 | DIV | allocating | 7 | AVX2 | 0.546 | 8.932 | 8.316 | 16.352x | 15.225x |
| INT16 | DIV | allocating | 1,000,000 | GENERIC | — | 2815.231 | 2993.438 | —x | —x |
| INT16 | DIV | allocating | 1,000,000 | AVX2 | 762.975 | 2718.698 | 2942.507 | 3.563x | 3.857x |
| INT16 | DIV | inplace | 7 | GENERIC | — | 9.021 | 6.507 | —x | —x |
| INT16 | DIV | inplace | 7 | AVX2 | 0.237 | 9.205 | 6.495 | 38.863x | 27.421x |
| INT16 | DIV | inplace | 1,000,000 | GENERIC | — | 2843.849 | 2975.564 | —x | —x |
| INT16 | DIV | inplace | 1,000,000 | AVX2 | 649.740 | 2872.161 | 2879.725 | 4.420x | 4.432x |

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
