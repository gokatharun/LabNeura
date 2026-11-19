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

## M7.18.2 indexed semantic references (2026-11-19)

The immutable performance checkpoint is commit `41254c6`. The optimized tree
is intentionally uncommitted pending review. The implementation retains
`semantic.references` as the ordered compatibility and diagnostic oracle and
adds one 24-byte ordinal entry per reference. Entries are stable-sorted by
source-buffer identity and byte offset once, after semantic resolution. A
binary search returns the complete duplicate range in original insertion
order; consumers then retain their established first-match, last-match,
declaration-filter, and null-result behavior. Symbols and scopes remain owned
by the semantic model, so moving the result does not invalidate indexed symbol
identity. External mutation of the public reference vector requires an
explicit `rebuild_reference_index()`.

The audited location lookups were in assignment coverage (symbol and constant
resolution), callable evaluation, lvalue resolution, native lowering (symbol
and constant resolution), process/event analysis, process dependencies,
sequential analysis, task execution, and type analysis. Each previously
searched the complete reference vector, making repeated resolution O(R) per
lookup and common lowering paths O(R x L). Name-only ambiguity fallbacks remain
linear because they implement a different global-name contract and were not a
confirmed location-lookup hotspot.

Standalone experiments included construction in total cost and tested misses,
duplicates, sequential/random access, and sets from 65 to 60,469 references.
At 60,469 references and 2,048 queries, representative total times were
74-84 ms linear, 1.24-1.25 ms sorted, 1.88 ms hash, and 1.26-1.29 ms dense.
Sorted storage was about 1.45 MB versus 2.42 MB estimated for the hash index.
Linear lookup remained competitive at 65 references. Dense storage was not
selected because the production key is a `(source identity, byte offset)` pair
without an existing bounded dense ID; hash storage used more memory and did
not preserve deterministic duplicate ordering without extra machinery.

### Apple M1 before and after

The exact 33 M7.18.1 sources, hashes, Release compiler configuration, scalar
backend, two warmups, and nine repetitions were reused. All hashes matched.
Speedup is baseline time divided by optimized time.

| Boundary | Geometric mean | Median workload | Range |
|---|---:|---:|---:|
| Syntax | 1.006x | 0.998x | 0.951-1.176x |
| Semantic | 1.203x | 1.050x | 0.921-2.038x |
| Source-to-netlist | 1.967x | 1.625x | 0.952-12.394x |

For the 100 KiB Boolean workload:

| Boundary | Baseline median | Indexed median | Indexed p95 | Speedup |
|---|---:|---:|---:|---:|
| Syntax | 20.998 ms | 20.581 ms | 20.763 ms | 1.020x |
| Semantic | 88.911 ms | 43.720 ms | 49.045 ms | 2.034x |
| Source-to-netlist | 1,025.755 ms | 82.761 ms | 85.074 ms | 12.394x |

That workload contained 9,027 semantic references. Its representative index
construction took 102,333 ns and reserved 216,648 bytes; no duplicate source
locations occurred in this generated input. Timing instrumentation is disabled
unless `LABNEURA_EDA_REFERENCE_INDEX_PROFILE` is set. The complete boundary
still compares structurally different products and is not an equivalent
gate-level native-versus-Slang claim.

An unsandboxed `/usr/bin/time -l` run measured 84,606,976-byte peak RSS for the
full optimized benchmark process, versus 84,295,680 bytes at baseline. A fresh
macOS `sample` profile no longer shows the old full-vector resolver or
`_platform_memcmp` as leading entries. `references_at()` had eight collapsed
samples; lexer classification and synthesis eligibility are now visible among
the sampled work. Hardware counters and xctrace remained unavailable.

Correctness evidence includes a fresh full Release CTest run (72 registered,
62 passed and 10 expected ARM64 AVX2 skips), plus the separate EDA-only run
(42 registered, 41 passed and one expected AVX2 skip), all Python
`test_eda*.py` regressions, the existing
Slang differential and M7.17 equivalence/formal/randomized tests, and focused
linear-oracle comparisons for real analysis plus synthetic duplicate, missing,
distinct-buffer, deterministic-rebuild, and move cases. `git diff --check`
passes. Linux x86-64 optimized profiling, AVX2 timing, `perf`, sanitizer
reruns, and hardware counters remain unavailable on this Apple M1 host; the
earlier Linux correctness workflow is not presented as M7.18.2 performance
evidence. Machine-readable optimized results are in
`benchmarking/results/m7182-m1-20261119/`.

The next evidence-backed optimization is to profile synthesis eligibility and
the remaining name-only constant fallbacks, then consider stable semantic IDs
only if their measured share justifies the added lifecycle complexity.

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
