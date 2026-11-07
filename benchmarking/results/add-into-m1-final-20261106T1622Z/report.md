# Reusable-output FP32 ADD benchmark

- Timestamp UTC: 2026-11-06T16:20:45.419431+00:00
- CPU: Apple M1; OS: macOS-27.0.1-arm64-arm-64bit
- Python: 3.9.6; NumPy: 2.0.2; PyTorch: 2.8.0
- LabNeura backend(s): GENERIC, NEON; execution threads: 1
- Compiler: Apple clang version 21.0.0 (clang-2100.3.34.2); build type: Release; flags: ` -O3 -DNDEBUG`
- Rotating pool target: 64 MiB; pool capped at 128 sets. The target includes input and reusable output pools; allocating mode has input pools and a transient output. This is not a claim of DRAM-cold execution.
- Allocating mode times result creation and release. Reusable-output mode reuses a preallocated output for every call.
- All inputs are constructed before timing. Reusable-output destinations are also preallocated; allocating results are created and released inside the timed calls. Each library cycles through the same deterministic operand-set indices. Inputs are not mutated during timed calls.
- NumPy native thread pools and PyTorch CPU threads are set to one. LabNeura execution thread count is explicitly set to one.

Median, IQR and p95 are per-call Python API latency in microseconds. Raw batch samples are in `benchmark.json`.

## Hot cache regime

### Allocating

| Elements | Library / backend | Median µs | IQR µs | p95 µs | vs NumPy | vs PyTorch | Generic / NEON | Iterations/batch | Pool working set MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | LabNeura / GENERIC | 0.570 | 0.021 | 0.634 | 0.85x | 1.47x | 1.17x | 2048 | 0.01 |
| 1024 | LabNeura / NEON | 0.486 | 0.018 | 0.653 | 1.00x | 1.72x | n/a | 2048 | 0.01 |
| 1024 | NumPy / CPU | 0.485 | 0.019 | 0.544 | 1.00x | 1.73x | n/a | 2048 | 0.01 |
| 1024 | PyTorch / CPU | 0.838 | 0.029 | 0.980 | 0.58x | 1.00x | n/a | 2048 | 0.01 |
| 4096 | LabNeura / GENERIC | 0.946 | 0.019 | 1.137 | 0.74x | 1.17x | 1.35x | 2048 | 0.03 |
| 4096 | LabNeura / NEON | 0.698 | 0.039 | 0.859 | 1.00x | 1.58x | n/a | 2048 | 0.03 |
| 4096 | NumPy / CPU | 0.698 | 0.014 | 0.872 | 1.00x | 1.59x | n/a | 2048 | 0.03 |
| 4096 | PyTorch / CPU | 1.106 | 0.025 | 1.195 | 0.63x | 1.00x | n/a | 2048 | 0.03 |
| 16384 | LabNeura / GENERIC | 3.813 | 0.074 | 3.877 | 0.58x | 0.75x | 1.71x | 2048 | 0.12 |
| 16384 | LabNeura / NEON | 2.236 | 0.079 | 2.318 | 0.99x | 1.28x | n/a | 2048 | 0.12 |
| 16384 | NumPy / CPU | 2.224 | 0.029 | 2.255 | 1.00x | 1.28x | n/a | 2048 | 0.12 |
| 16384 | PyTorch / CPU | 2.854 | 0.084 | 2.937 | 0.78x | 1.00x | n/a | 2048 | 0.12 |
| 65536 | LabNeura / GENERIC | 12.636 | 0.073 | 12.744 | 0.61x | 0.72x | 1.40x | 543 | 0.50 |
| 65536 | LabNeura / NEON | 9.008 | 0.709 | 9.158 | 0.86x | 1.00x | n/a | 543 | 0.50 |
| 65536 | NumPy / CPU | 7.749 | 0.787 | 8.485 | 1.00x | 1.17x | n/a | 543 | 0.50 |
| 65536 | PyTorch / CPU | 9.045 | 0.460 | 9.440 | 0.86x | 1.00x | n/a | 543 | 0.50 |
| 262144 | LabNeura / GENERIC | 50.615 | 1.130 | 52.745 | 0.67x | 0.69x | 1.48x | 144 | 2.00 |
| 262144 | LabNeura / NEON | 34.086 | 0.525 | 35.564 | 1.00x | 1.02x | n/a | 144 | 2.00 |
| 262144 | NumPy / CPU | 34.036 | 0.756 | 35.017 | 1.00x | 1.02x | n/a | 144 | 2.00 |
| 262144 | PyTorch / CPU | 34.697 | 0.357 | 35.483 | 0.98x | 1.00x | n/a | 144 | 2.00 |
| 1000000 | LabNeura / GENERIC | 211.431 | 23.715 | 236.818 | 0.65x | 0.71x | 1.44x | 34 | 7.63 |
| 1000000 | LabNeura / NEON | 146.420 | 16.460 | 163.174 | 0.94x | 1.03x | n/a | 34 | 7.63 |
| 1000000 | NumPy / CPU | 137.086 | 8.883 | 167.733 | 1.00x | 1.10x | n/a | 34 | 7.63 |
| 1000000 | PyTorch / CPU | 151.006 | 20.734 | 183.776 | 0.91x | 1.00x | n/a | 34 | 7.63 |

### Reusable Output

| Elements | Library / backend | Median µs | IQR µs | p95 µs | vs NumPy | vs PyTorch | Generic / NEON | Iterations/batch | Pool working set MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | LabNeura / GENERIC | 0.264 | 0.007 | 0.432 | 1.64x | 2.61x | 1.00x | 2048 | 0.01 |
| 1024 | LabNeura / NEON | 0.264 | 0.013 | 0.374 | 1.63x | 2.60x | n/a | 2048 | 0.01 |
| 1024 | NumPy / CPU | 0.432 | 0.008 | 0.535 | 1.00x | 1.59x | n/a | 2048 | 0.01 |
| 1024 | PyTorch / CPU | 0.688 | 0.031 | 0.727 | 0.63x | 1.00x | n/a | 2048 | 0.01 |
| 4096 | LabNeura / GENERIC | 0.450 | 0.024 | 0.559 | 1.37x | 2.03x | 0.98x | 2048 | 0.05 |
| 4096 | LabNeura / NEON | 0.459 | 0.079 | 0.683 | 1.34x | 1.99x | n/a | 2048 | 0.05 |
| 4096 | NumPy / CPU | 0.616 | 0.068 | 0.881 | 1.00x | 1.49x | n/a | 2048 | 0.05 |
| 4096 | PyTorch / CPU | 0.915 | 0.091 | 1.362 | 0.67x | 1.00x | n/a | 2048 | 0.05 |
| 16384 | LabNeura / GENERIC | 1.731 | 0.079 | 1.816 | 1.09x | 1.41x | 1.00x | 2048 | 0.19 |
| 16384 | LabNeura / NEON | 1.728 | 0.028 | 1.816 | 1.09x | 1.41x | n/a | 2048 | 0.19 |
| 16384 | NumPy / CPU | 1.883 | 0.051 | 1.963 | 1.00x | 1.29x | n/a | 2048 | 0.19 |
| 16384 | PyTorch / CPU | 2.434 | 0.072 | 2.538 | 0.77x | 1.00x | n/a | 2048 | 0.19 |
| 65536 | LabNeura / GENERIC | 7.540 | 0.742 | 8.721 | 1.03x | 1.14x | 1.00x | 565 | 0.75 |
| 65536 | LabNeura / NEON | 7.538 | 1.402 | 8.691 | 1.03x | 1.14x | n/a | 565 | 0.75 |
| 65536 | NumPy / CPU | 7.772 | 0.502 | 8.509 | 1.00x | 1.11x | n/a | 565 | 0.75 |
| 65536 | PyTorch / CPU | 8.602 | 0.385 | 9.134 | 0.90x | 1.00x | n/a | 565 | 0.75 |
| 262144 | LabNeura / GENERIC | 33.602 | 1.207 | 34.179 | 1.01x | 1.02x | 1.00x | 147 | 3.00 |
| 262144 | LabNeura / NEON | 33.713 | 0.968 | 34.787 | 1.00x | 1.02x | n/a | 147 | 3.00 |
| 262144 | NumPy / CPU | 33.797 | 0.940 | 34.333 | 1.00x | 1.02x | n/a | 147 | 3.00 |
| 262144 | PyTorch / CPU | 34.355 | 0.427 | 35.218 | 0.98x | 1.00x | n/a | 147 | 3.00 |
| 1000000 | LabNeura / GENERIC | 139.140 | 8.471 | 154.159 | 1.13x | 1.09x | 0.93x | 34 | 11.44 |
| 1000000 | LabNeura / NEON | 150.116 | 10.365 | 166.565 | 1.04x | 1.01x | n/a | 34 | 11.44 |
| 1000000 | NumPy / CPU | 156.825 | 10.463 | 168.590 | 1.00x | 0.96x | n/a | 34 | 11.44 |
| 1000000 | PyTorch / CPU | 151.039 | 11.326 | 177.333 | 1.04x | 1.00x | n/a | 34 | 11.44 |

## Rotating cache regime

### Allocating

| Elements | Library / backend | Median µs | IQR µs | p95 µs | vs NumPy | vs PyTorch | Generic / NEON | Iterations/batch | Pool working set MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | LabNeura / GENERIC | 0.616 | 0.057 | 0.826 | 0.86x | 1.46x | 1.19x | 2048 | 1.00 |
| 1024 | LabNeura / NEON | 0.517 | 0.023 | 0.626 | 1.02x | 1.74x | n/a | 2048 | 1.00 |
| 1024 | NumPy / CPU | 0.528 | 0.024 | 0.669 | 1.00x | 1.71x | n/a | 2048 | 1.00 |
| 1024 | PyTorch / CPU | 0.900 | 0.104 | 1.246 | 0.59x | 1.00x | n/a | 2048 | 1.00 |
| 4096 | LabNeura / GENERIC | 1.151 | 0.022 | 1.256 | 0.81x | 1.13x | 1.21x | 2048 | 4.00 |
| 4096 | LabNeura / NEON | 0.952 | 0.033 | 1.076 | 0.98x | 1.37x | n/a | 2048 | 4.00 |
| 4096 | NumPy / CPU | 0.932 | 0.058 | 1.096 | 1.00x | 1.40x | n/a | 2048 | 4.00 |
| 4096 | PyTorch / CPU | 1.303 | 0.049 | 1.429 | 0.71x | 1.00x | n/a | 2048 | 4.00 |
| 16384 | LabNeura / GENERIC | 4.401 | 0.141 | 4.549 | 0.80x | 0.89x | 1.19x | 1408 | 16.00 |
| 16384 | LabNeura / NEON | 3.713 | 0.257 | 3.972 | 0.94x | 1.05x | n/a | 1408 | 16.00 |
| 16384 | NumPy / CPU | 3.509 | 0.228 | 4.013 | 1.00x | 1.11x | n/a | 1408 | 16.00 |
| 16384 | PyTorch / CPU | 3.897 | 0.210 | 4.280 | 0.90x | 1.00x | n/a | 1408 | 16.00 |
| 65536 | LabNeura / GENERIC | 14.141 | 0.844 | 16.271 | 0.81x | 0.84x | 1.26x | 516 | 43.00 |
| 65536 | LabNeura / NEON | 11.250 | 0.873 | 13.840 | 1.01x | 1.05x | n/a | 516 | 43.00 |
| 65536 | NumPy / CPU | 11.405 | 1.346 | 14.016 | 1.00x | 1.04x | n/a | 516 | 43.00 |
| 65536 | PyTorch / CPU | 11.822 | 1.171 | 13.592 | 0.96x | 1.00x | n/a | 516 | 43.00 |
| 262144 | LabNeura / GENERIC | 56.623 | 4.007 | 64.476 | 0.86x | 0.85x | 1.17x | 132 | 44.00 |
| 262144 | LabNeura / NEON | 48.441 | 5.693 | 63.434 | 1.01x | 0.99x | n/a | 132 | 44.00 |
| 262144 | NumPy / CPU | 48.767 | 6.125 | 58.258 | 1.00x | 0.99x | n/a | 132 | 44.00 |
| 262144 | PyTorch / CPU | 48.128 | 5.920 | 64.633 | 1.01x | 1.00x | n/a | 132 | 44.00 |
| 1000000 | LabNeura / GENERIC | 210.569 | 19.515 | 226.671 | 0.87x | 0.97x | 1.11x | 36 | 45.78 |
| 1000000 | LabNeura / NEON | 189.220 | 17.733 | 217.653 | 0.97x | 1.08x | n/a | 36 | 45.78 |
| 1000000 | NumPy / CPU | 182.639 | 18.194 | 215.314 | 1.00x | 1.12x | n/a | 36 | 45.78 |
| 1000000 | PyTorch / CPU | 204.064 | 27.452 | 226.990 | 0.90x | 1.00x | n/a | 36 | 45.78 |

### Reusable Output

| Elements | Library / backend | Median µs | IQR µs | p95 µs | vs NumPy | vs PyTorch | Generic / NEON | Iterations/batch | Pool working set MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | LabNeura / GENERIC | 0.353 | 0.010 | 0.395 | 1.40x | 2.11x | 1.00x | 2048 | 1.50 |
| 1024 | LabNeura / NEON | 0.352 | 0.025 | 0.572 | 1.41x | 2.11x | n/a | 2048 | 1.50 |
| 1024 | NumPy / CPU | 0.495 | 0.009 | 0.519 | 1.00x | 1.50x | n/a | 2048 | 1.50 |
| 1024 | PyTorch / CPU | 0.743 | 0.065 | 1.052 | 0.67x | 1.00x | n/a | 2048 | 1.50 |
| 4096 | LabNeura / GENERIC | 0.786 | 0.030 | 0.918 | 1.18x | 1.54x | 0.99x | 2048 | 6.00 |
| 4096 | LabNeura / NEON | 0.793 | 0.018 | 0.900 | 1.17x | 1.53x | n/a | 2048 | 6.00 |
| 4096 | NumPy / CPU | 0.927 | 0.038 | 1.054 | 1.00x | 1.31x | n/a | 2048 | 6.00 |
| 4096 | PyTorch / CPU | 1.210 | 0.046 | 1.378 | 0.77x | 1.00x | n/a | 2048 | 6.00 |
| 16384 | LabNeura / GENERIC | 4.201 | 0.350 | 5.619 | 1.04x | 1.10x | 1.00x | 1152 | 24.00 |
| 16384 | LabNeura / NEON | 4.182 | 0.388 | 6.361 | 1.04x | 1.11x | n/a | 1152 | 24.00 |
| 16384 | NumPy / CPU | 4.349 | 0.431 | 7.106 | 1.00x | 1.06x | n/a | 1152 | 24.00 |
| 16384 | PyTorch / CPU | 4.630 | 0.372 | 12.732 | 0.94x | 1.00x | n/a | 1152 | 24.00 |
| 65536 | LabNeura / GENERIC | 15.610 | 0.478 | 16.437 | 1.03x | 1.04x | 0.99x | 344 | 64.50 |
| 65536 | LabNeura / NEON | 15.758 | 0.882 | 16.964 | 1.02x | 1.03x | n/a | 344 | 64.50 |
| 65536 | NumPy / CPU | 16.121 | 0.955 | 16.939 | 1.00x | 1.00x | n/a | 344 | 64.50 |
| 65536 | PyTorch / CPU | 16.169 | 0.693 | 16.823 | 1.00x | 1.00x | n/a | 344 | 64.50 |
| 262144 | LabNeura / GENERIC | 66.371 | 9.059 | 78.961 | 1.02x | 1.04x | 0.94x | 88 | 66.00 |
| 262144 | LabNeura / NEON | 70.702 | 8.528 | 82.355 | 0.96x | 0.97x | n/a | 88 | 66.00 |
| 262144 | NumPy / CPU | 67.889 | 7.685 | 80.637 | 1.00x | 1.02x | n/a | 88 | 66.00 |
| 262144 | PyTorch / CPU | 68.928 | 8.422 | 80.393 | 0.98x | 1.00x | n/a | 88 | 66.00 |
| 1000000 | LabNeura / GENERIC | 243.200 | 22.072 | 260.699 | 1.01x | 1.00x | 1.03x | 36 | 68.66 |
| 1000000 | LabNeura / NEON | 236.142 | 27.912 | 264.475 | 1.04x | 1.03x | n/a | 36 | 68.66 |
| 1000000 | NumPy / CPU | 245.131 | 22.289 | 265.155 | 1.00x | 0.99x | n/a | 36 | 68.66 |
| 1000000 | PyTorch / CPU | 242.616 | 16.902 | 271.394 | 1.01x | 1.00x | n/a | 36 | 68.66 |

## Interpretation

Compare reusable-output rows only with each other, and allocating rows only with each other. Results include Python binding and dispatch overhead; they are not isolated-kernel measurements. Backend selection is explicit and checked before input construction. Results are specific to the recorded machine and software versions. Speedup ratios are competitor median divided by the row median; values above 1.0x mean the row was faster. `Generic / NEON` is Generic median divided by NEON median.
