# Performance

Current measured source revision: `32726d832fb8842eaca70ff4e0bb0c0f4ebdbd18`.
Immediate previous source revision: `cbc62824f8c29d50b16c990c21ed69ea75b5f5ab` (N/C until a compatible persisted measurement exists).
Measurement host: `macOS-27.2.1-arm64-arm-64bit`; uninstrumented Release binaries; benchmark-specific warmups/samples are retained in artifacts.

Higher throughput is better. `N/C` means no compatible baseline is available; it is not zero change.

| Workload | Previous | Current | Change |
|---|---:|---:|---:|
| Tensor GENERIC FP32 ADD | N/C | 4625.23 M elem/s | N/C |
| Tensor GENERIC INT8 ADD | N/C | 1117.84 M elem/s | N/C |
| Tensor NEON FP32 ADD | N/C | 8181.39 M elem/s | N/C |
| Tensor NEON INT8 ADD | N/C | 33026.02 M elem/s | N/C |
| EDA lexer scalar | N/C | 153.64 MB/s | N/C |
| EDA lexer neon | N/C | 216.04 MB/s | N/C |

Unsupported backends are reported by their benchmark artifacts as `N/A`; no fallback is labelled as SIMD.
