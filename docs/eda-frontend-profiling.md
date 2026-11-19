# EDA frontend comparative profiling

## M7.18.1 Apple M1 baseline (2026-11-19)

This is a measurement milestone. No production frontend or SIMD algorithm was
changed. Slang is pinned to `11.0.0+7ddf4059f`.

The Release baseline used Apple Clang 21 on arm64 macOS 27.2.1, two warmups,
nine measured repetitions, and `std::chrono::steady_clock`. It contains 33
matched workloads: eleven deterministic families at 1 KiB, 10 KiB, and
100 KiB. Every timed input passed both frontends before its ratio was accepted.
The complete raw report and generated sources are under
`benchmarking/results/m7181-m1-20261119/`.

### Measurement boundaries

- Syntax compares complete in-memory syntax-tree construction. Native includes
  its supported preprocessing; Slang uses `SyntaxTree::fromText`.
- Semantic compares syntax plus forced semantic/type/elaboration work. Slang
  forces `getRoot()`, `getAllDiagnostics()`, and member traversal so lazy work
  is not mistaken for completion.
- Source-to-netlist compares the existing public `parse_verilog(path)` with the
  native syntax, semantic, type, process-eligibility, and lowering pipeline.
  Slang's public path includes file I/O; native uses the preloaded source.
- The complete representations differ materially. Native expands supported RTL
  to generic gates, while `parse_verilog()` retains behavioral structure. The
  ratios below are boundary costs, not qualified equivalent-netlist speedups.

Across the 33 workloads, the median Slang/native ratio was 0.203 for syntax,
0.195 for semantic, and 0.207 for source-to-netlist. A value below one means
native was slower. At 1 KiB the median complete ratio was 1.18 (native slightly
faster); at 10 KiB it was 0.207; at 100 KiB it was 0.0277. The growing gap and
the structural expansion mean the complete result must not be presented as a
parser-only comparison.

For the 100 KiB Boolean workload (102,489 bytes, 60,217 native tokens, 1,003
continuous assignments), representative medians were:

| Boundary | Native scalar | Slang | Slang/native |
|---|---:|---:|---:|
| Syntax | 20.998 ms | 3.484 ms | 0.166x |
| Semantic | 88.911 ms | 7.067 ms | 0.0795x |
| Source-to-netlist | 1,025.755 ms | 22.070 ms | 0.0215x |

Native produced 12,039 nets and 10,030 generic instances; Slang's behavioral
module reported six nets and no instances. This is a structural difference,
not an equivalence failure.

### Backend and memory evidence

On the same 100 KiB Boolean source, native NEON syntax median was 20.159 ms
versus scalar 20.998 ms (about 4.2% faster). Semantic and complete medians were
within normal run variability. AVX2 is unavailable on this ARM64 host and is
recorded as unavailable rather than zero.

An unsandboxed `/usr/bin/time -l` run measured 84,295,680-byte peak RSS,
137,311,292,562 retired instructions, and 43,501,970,615 elapsed cycles for the
full native benchmark process. The matching Slang process measured 17,154,048
bytes, 5,222,185,600 instructions, and 1,243,619,013 cycles. These are
whole-process totals across all warmups and repetitions, not per-pass metrics.

### Sampled native hotspots

macOS `sample` collected 3,529 one-millisecond main-thread samples from the
100 KiB native workload. The leading top-of-stack entries were:

| Rank | Samples | Symbol |
|---:|---:|---|
| 1 | 1,253 | `_platform_memcmp` |
| 2 | 753 | native lowering constant-resolver callback |
| 3 | 417 | process dependency `Analyzer::expression` |
| 4 | 157 | `memcmp` stub |
| 5 | 142 | `VerilogNativeLoweringSession::resolve_symbol` |
| 6 | 121 | native `Scanner::run` |
| 7 | 49 | constant-value hash-table lookup |
| 8 | 44 | process dependency `Analyzer::destination` |
| 9 | 43 | deduplicated native symbol |
| 10 | 40 | allocator free path |
| 11 | 39 | expression-type hash-table insertion |
| 12 | 38 | `_platform_memchr` |
| 13 | 38 | constant integer parsing |
| 14 | 24 | tiny allocation path |
| 15 | 23 | `VerilogIntegralType` destruction |
| 16 | 18 | `_platform_memmove` |
| 17 | 17 | `TypeAnalyzer::infer_expression` |
| 18 | 16 | `mach_absolute_time` |
| 19 | 14 | `Parser::parse_postfix` |
| 20 | 12 | integral-type assignment |

The strongest measured bottleneck is repeated linear semantic-reference
searching. Native lowering scans `semantic.references` in `resolve_symbol()`
and its constant resolver, and process dependency analysis performs another
linear location lookup. The dominant `memcmp` samples occur beneath those
lookups. A future milestone should first prototype a source-location-to-symbol
index shared by analysis and lowering, then remeasure. Other evidence-backed
candidates are reducing repeated integral-type copies/destruction and avoiding
repeated constant evaluation. These are hypotheses only; M7.18.1 implements
none of them.

Slang's built-in hierarchical trace reported `parseFile` at 3.100 ms and
`AST Instance` at 2.921 ms for one semantic pass. `xctrace` is not installed,
so a Slang top-20 sampled-function list and allocation trace are unavailable.
No substitute list is fabricated.

### Limits and next milestone

The 1 MiB and 4 MiB complete tiers were not timed in the initial controlled
run: native gate expansion was already roughly one second per 100 KiB Boolean
pass and showed strongly super-linear scaling. Those tiers remain available
from the deterministic generator for syntax-only or separately budgeted runs.
Linux x86-64 AVX2, `perf`, and heap profiles remain pending until the separate
correctness baseline is green.

The recommended next implementation milestone is a narrowly scoped indexed
semantic-reference lookup experiment, guarded by all M7.17 equivalence tests,
followed by the identical 33-workload rerun. Do not optimize the lexer first:
NEON changed end-to-end syntax time only slightly on the measured workload.

## Reproduction

```sh
python3 benchmarks/eda/generate_frontend_workloads.py /tmp/frontend-workloads \
  --sizes 1024,10240,102400
cmake --build <release-build> --target \
  labneura_native_frontend_profile labneura_slang_frontend_profile
<release-build>/labneura_native_frontend_profile source.sv top scalar workload
<release-build>/labneura_slang_frontend_profile source.sv --trace slang-trace.json
python3 benchmarks/eda/profile_frontends.py --output-dir profile -- <command...>
python3 benchmarks/eda/aggregate_frontend_results.py raw/*.jsonl \
  --json comparison.json --markdown report.md
```
