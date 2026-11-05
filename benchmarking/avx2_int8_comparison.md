# Linux x86_64 INT8 Multiplication: Generic vs AVX2

- Run time (UTC): 2026-11-05T14:33:34.090949+00:00
- Commit: `a3fc9795b9bc77a5b7cdea1e403faa103f5809c3`
- OS / architecture: Linux-7.0.0-1017-azure-x86_64-with-glibc2.39 / x86_64
- CPU: AMD EPYC 7763 64-Core Processor
- Runtime-selected backend: AVX2
- Operation: in-place INT8 elementwise multiply with multiplier values of 1
- Timing includes Python binding call overhead; tensors are allocated before timing.
- Samples: 7 per backend and size; median reported; 5 warmups per backend.
- Correctness was checked against independently clamped expected values and Generic results for sizes 1, 8, 31, 32, 33, 63, 64, 65, 96, 97, 256.

| Elements | Iterations/sample | Generic median (ns/op) | AVX2 median (ns/op) | AVX2 speedup | Generic M elements/s | AVX2 M elements/s |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 5,000 | 223.0 | 168.1 | 1.33x | 143.47 | 190.41 |
| 256 | 5,000 | 507.4 | 176.2 | 2.88x | 504.57 | 1,452.60 |
| 4,096 | 1,220 | 5,300.9 | 515.6 | 10.28x | 772.70 | 7,943.45 |
| 65,536 | 76 | 81,901.8 | 6,031.7 | 13.58x | 800.18 | 10,865.31 |
| 1,048,576 | 50 | 1,314,434.1 | 92,759.8 | 14.17x | 797.74 | 11,304.21 |

Speedup is Generic median latency divided by AVX2 median latency; values above 1 indicate AVX2 was faster. Results are specific to this GitHub-hosted runner and include Python-call overhead, so they are not a hardware-independent performance guarantee.
