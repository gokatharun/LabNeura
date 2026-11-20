# LabNeura Feature and Backend Test Matrix

Use this checklist for every feature and every backend implementation.

## Native-first production frontend (M7.19)

- [x] Production EDA builds without Slang, Tensor, or Acceleration Core.
- [x] Native parse/project/lowering paths are production defaults.
- [x] CI wheel/sdist artifact audit and clean install without Slang are
  implemented for Linux. CPython 3.11 Windows AMD64 core/EDA wheel generation,
  exact-wheel runtime qualification, PE inspection, strict MSVC CTest, immutable
  artifacts and release gates are implemented in M7.20.2; Windows execution
  remains pending. The EDA sdist is staged with native sources and must rebuild
  a wheel before release qualification.
- Unsupported SystemVerilog, AST JSON parity, instance arrays, and unsupported
  lowering forms diagnose explicitly; external Slang is validation-only.

## Feature contract
- [ ] API/signature and Python binding.
- [ ] Input/output shape and rank.
- [ ] Supported dtypes and promotion.
- [ ] Broadcasting and layout rules.
- [ ] Numerical semantics (overflow, saturation, rounding, NaN/Inf, divide-by-zero).
- [ ] Allocating/in-place behavior.
- [ ] Aliasing/view behavior.
- [ ] Invalid-input errors.
- [ ] Supported backends and sync behavior.

## Minimum correctness
- [ ] Typical, empty (where valid), singleton, odd length and large input.
- [ ] SIMD-width and parallel-grain boundary lengths.
- [ ] Invalid shape/dtype/argument.
- [ ] Floating special values where relevant.
- [ ] Integer overflow/saturation boundaries.
- [ ] Exact integer equality or documented floating tolerance.

### BOOL dtype foundation
- [x] Dedicated BOOL dtype, one `std::uint8_t` per logical element, canonical
  0/1 storage, and no INT8 aliasing or `std::vector<bool>` storage.
- [x] Dtype byte-size validation, allocation/deallocation, backend accessors,
  clone/deep-copy, views/lifetime, and zero-sized storage.
- [x] Generic and NEON explicitly selected construction/access tests; verify
  logical size independently of SIMD-padded capacity.
- [x] AVX2 construction/access runtime validation on an AVX2-capable host.
- [x] C++ byte/numeric and Python bool/numeric inputs normalize to 0/1; Python
  rejects nonnumeric data and returns actual Python bool values.
- [ ] BOOL arithmetic remains explicitly unsupported until separately
  implemented and tested.

### Tensor comparisons
- [x] Generic `eq`, `ne`, `lt`, `le`, `gt`, and `ge` return independent BOOL
  tensors.
- [x] Matching FP32, FP16, signed INT16, and signed INT8 inputs; mismatched
  dtypes and unsupported BOOL comparisons fail clearly.
- [x] Equal-shape, scalar, 2D/3D/nD, zero-sized, and offset-view cases reuse
  `BroadcastPlan`; output shape and contiguous layout are checked.
- [x] IEEE NaN/Inf/signed-zero cases and signed integer boundaries are covered;
  FP16 compares decoded binary16 values.
- [x] Generic and NEON tests explicitly select their backend; NEON parity covers
  all operations and numeric dtypes, IEEE special values, signed integer
  boundaries, broadcasting, scalar inputs, empty outputs, and offset views.
- [x] AVX2 kernels and explicit tests cover all operations/dtypes, Generic
  parity, special values, signed boundaries, broadcast/scalar/nested views,
  empty outputs, canonical BOOL bytes, and width-boundary tails.
- [x] Execute AVX2 native/Python comparison and regression tests on an
  AVX2-capable x86 host; confirm runtime detection and no fallback.

M3.3.5 runtime validation passed in GitHub Actions run `42390563490` on
implementation commit `895b7633e1cd6f48b1459397cd664ee1d4dd2c08`. The
`tensor_comparison_avx2` native CTest and AVX2 comparison Python tests passed.
The full native CTest run had 29 registered tests, 0 failures, and 7 expected
NEON-only skips; the full Python suite with AVX2 selected reported 1156 passed,
215 skipped, and 3 warnings. AVX2 arithmetic regression and FP16/F16C vector
and scalar fallback tests passed. ARM64 Generic and NEON comparison CTests
passed, with AVX2 skipped on ARM64. This validates correctness only; no
performance improvement is claimed.

### BOOL logical operations — M3.4.1–M3.4.5
- [x] Generic `logical_not` accepts BOOL only and returns an independent,
  canonical BOOL tensor with unchanged shape.
- [x] Mixed values, all-false/all-true, scalar, empty, offset-view, repeated
  calls, input immutability, and output-storage independence are tested.
- [x] Non-BOOL input is rejected; Generic backend selection is asserted.
- [x] NEON explicitly selected; vector chunks, all scalar-tail remainders,
  scalar/empty tensors, offset views, canonical output, immutability, and
  independent output storage are tested.
- [x] AVX2 kernel and explicit tests cover vector boundaries, all 32 scalar
  tail remainders, scalar/empty inputs, offset views, canonical output,
  immutability, repeated calls, and independent storage.
- [x] AVX2 `logical_not` passed x86_64 runtime validation in CI run
  `42449211301`, commit `665991bdf5f790297f29dc1423674d6278947d68`.
- [x] `logical_and` requires BOOL operands, uses shared NumPy-style
  broadcasting, returns an independent BOOL tensor, and rejects incompatible
  shapes and non-BOOL inputs.
- [x] Generic, NEON, and AVX2 `logical_and` tests cover the four truth-table
  combinations, same-shape and two-axis broadcasts, scalars, empty outputs,
  offset views, BOOL dtype/canonical values, input immutability, and SIMD
  vector boundaries/tails. Backend selection is explicitly checked.
- [x] Generic and NEON `logical_and` native tests and Python tests executed on
  ARM64. Full CTest: 21 passed, 8 AVX2 skips, 0 failures. Full Python suite:
  1164 passed, 215 skipped, 3 warnings.
- [x] `logical_and` AVX2 native and Python cases executed on x86_64 with AVX2
  selected in CI run `42455884336` at commit
  `1089815717ce37815210ec425ec295678c2d96ef`.
- [x] `logical_or` requires BOOL operands, uses shared NumPy-style
  broadcasting, returns an independent BOOL tensor, and rejects incompatible
  shapes and non-BOOL inputs.
- [x] Generic, NEON, and AVX2 `logical_or` native and Python tests cover the
  truth table, same-shape and two-axis broadcasts, scalars, empty outputs,
  offset views, BOOL dtype/canonical values, input immutability, and SIMD
  vector boundaries/tails. Backend selection is explicitly checked; the
  Python AVX2 case has a dedicated test.
- [x] Generic and NEON `logical_or` native and Python cases passed on the
  ARM64 development host.
- [ ] Execute AVX2 `logical_or` native and Python cases on an x86_64 host
  with AVX2 support; ARM skips and cross-compilation do not count as runtime
  validation.
- [ ] BOOL `where()`.

The current Tensor API creates contiguous layouts only; non-contiguous-view
coverage is not applicable yet. M3.4.1 provides the Generic `logical_not`
kernel and M3.4.2 adds NEON. M3.4.3 `logical_not` AVX2 runtime validation
passed in CI run `42449211301`. M3.4.4 `logical_and` passed x86_64 runtime
validation in CI run `42455884336` at commit
`1089815717ce37815210ec425ec295678c2d96ef`. M3.4.5 adds Generic, NEON, and
AVX2 `logical_or`; x86_64 runtime validation for `logical_or` remains pending.
Apple Silicon M3.4.2 results: focused CTests had 4 passes and 2
AVX2 skips; focused Python tests had 128 passes and 61 AVX2 skips. The full
native suite had 21 passes and 8 AVX2 skips; the full Python suite had 1162
passes, 214 skips, and 3 existing warnings. The NEON tests executed on ARM64.

## Broadcasting
- [ ] Same shape.
- [ ] Scalar with tensor.
- [ ] Singleton expansion.
- [ ] Missing leading dimensions.
- [ ] Multi-axis broadcast.
- [ ] Incompatible shapes.
- [ ] Zero-size dimensions if supported.
- [ ] In-place expansion of destination rejects.

Examples: `(3,1)+(1,4)->(3,4)`; `(5,3,1)+(1,4)->(5,3,4)`; `(2,3)+(4,3)` is incompatible.

M3.1 implements only a reusable shape/effective-stride planner. It accepts
element-based input strides, including non-contiguous layouts, and does not
allocate expanded storage or execute arithmetic itself. M3.2.2/M3.2.3 use this
planner and a shared typed iterator for Generic FP32, INT8, and INT16
elementwise arithmetic; FP16 and optimized-backend broadcast kernels remain
future work.
M3.1 is locally validated by the native Generic/NEON nD tests and Python
planner tests. The x86 CI run must still execute the `tensor_nd_avx2` planner
tests and Python suite on this change before AVX2 CI validation is complete.

M3.2.1 defines arithmetic semantics only; no arithmetic or backend behavior
has changed at that milestone. See
[arithmetic-semantics.md](./arithmetic-semantics.md) for the normative dtype,
overflow, division, broadcast, aliasing and API contract, plus the
operation-by-dtype semantic test matrix. M3.2.2 implements a subset:
Generic FP32 ADD/SUB/MUL/DIV with allocating and supported in-place
broadcasting. It covers scalar, rank extension, singleton and multi-axis
broadcasts, zero-sized output, incompatible shapes, destination expansion,
offset views, alias safety, IEEE special values, and NumPy comparisons. Focused
Python tests pass with Generic selected; full native/Python suite results are
recorded in `current-state.md`. M3.2.3 adds Generic INT8 ADD/SUB/MUL/DIV
broadcasting with widened-and-clipped NumPy references, saturation boundaries,
division edge cases, views, empty tensors, allocating independence and
in-place alias checks. M3.2.4 adds Generic INT16 allocating and in-place
broadcasting using widened-and-clipped references, including INT16 extremes,
division edge cases, equal-shape operations, offset views, empty tensors, and
alias rejection. FP16 and NEON/AVX2 broadcasting remain pending; the general
matrix below is not yet complete. Python tests that require FP16 storage or
arithmetic explicitly skip when Generic is selected; mixed-mode checks retain
coverage for supported FP32/INT8/INT16 combinations.

### Elementwise arithmetic semantic matrix

- [x] Generic FP32 ADD/SUB/MUL/DIV: allocating results, allowed in-place RHS
  broadcasting, destination-expansion rejection, scalar and nD shapes,
  zero-sized outputs, offset views, alias safety, and IEEE special values.
- [x] Generic INT8 ADD/SUB/MUL/DIV: allocating and in-place broadcasting,
  scalar/rank/multi-axis/empty shapes, offset views, saturation, zero-divisor
  and truncation semantics, minimum-value overflow, and alias rejection.
- [x] Generic INT16 ADD/SUB/MUL/DIV: allocating and in-place broadcasting,
  equal-shape and scalar/rank/multi-axis/empty shapes, offset views, saturation,
  zero-divisor and truncation semantics, minimum-value overflow, and alias
  rejection.
- [ ] ADD, SUB, MUL, DIV for FP32, FP16, INT16 and INT8, allocating/in-place.
- [ ] Same-dtype result preservation; every mixed-dtype pair rejected.
- [ ] Same shape, scalar, singleton, rank extension, multi-axis, zero-size,
  incompatible shape and output-shape checks.
- [ ] In-place RHS expansion allowed only when output shape equals destination;
  destination expansion rejected.
- [ ] INT8/INT16 saturation for ADD/SUB/MUL; truncation-toward-zero DIV;
  zero-divisor and minimum-signed-value divided by `-1` cases.
- [ ] FP32 IEEE signed zero, divide-by-zero, NaN/Inf, subnormal and overflow.
- [ ] FP16 ties-to-even rounding, binary16 subnormal/overflow and NaN/Inf.
- [ ] Exact and partial overlap, views/offsets, no expanded storage allocation.
- [ ] Explicit Generic/NEON/AVX2 operation × dtype tests, SIMD tails and Python
  API/error parity. AVX2 requires actual capable x86 runtime validation.

## Storage and views
- [ ] Shape/stride/offset invariants and overflow.
- [ ] Contiguous layout.
- [ ] First-axis half-open range slices and first-axis selection.
- [ ] Empty/full/first/last ranges and zero-sized dimensions.
- [ ] Nested slicing/selection and scalar selection from a 1D tensor.
- [ ] Bounds checking against logical storage extent, not padded capacity.
- [ ] Transpose view correctness.
- [ ] Reshape view only when legal.
- [ ] Clone/copy independence.
- [ ] View shares storage as documented.
- [ ] Mutation and lifetime semantics.
- [ ] Exact alias behavior and partial-overlap rejection.
- [ ] Offset views preserve backend and dtype; SIMD alignment and tails are safe.
- [ ] Sanitizer checks for lifetime/ownership changes.

## Backend-specific
- [ ] Explicitly force/directly invoke backend.
- [ ] Compare with Generic reference.
- [ ] Test each claimed dtype/operation.
- [ ] Aligned and unaligned inputs.
- [ ] SIMD tails and boundary values.
- [ ] Aliasing and unsupported combinations.
- [ ] Ensure backend is not silently skipped by auto-dispatch.

NEON requires ARM64 runtime validation; AVX2 requires x86_64 AVX2 runtime validation; GPU backends require actual supported hardware. Compile-only is not runtime validation.

## Operation-specific
Elementwise: tensor-tensor, tensor-scalar, scalar-tensor, broadcasting, allocating/in-place, aliasing.
Reductions: whole tensor, each axis, multiple axes, keepdims, empty policy, accumulation dtype.
Matmul: dot, matrix-vector, rectangular 2D, transposed operands if supported, batched, batch broadcast, tile tails, numerical reference.
ML: stable softmax, axis behavior, activation boundaries, loss shape/reduction, gradients once autograd exists.

## Python
- [ ] Construction/dtype conversion.
- [ ] Shape/metadata.
- [ ] Results and exceptions.
- [ ] In-place mutation.
- [ ] Copy/view semantics.
- [ ] C++/Python parity.

## Performance gate
- [ ] Correctness before timing.
- [ ] Identical inputs/schedules and semantics.
- [ ] Warmup and repeated batches.
- [ ] Median, IQR, p95.
- [ ] Backend/dtype/shape/mode/threads recorded.
- [ ] Hot and rotating regimes where relevant.
- [ ] Native and Python timings separated.
- [ ] Regressions reported; old result directories immutable.

## Definition of done
A feature is complete only when its semantics are documented, implementation exists, tests pass, each claimed backend has explicit tests, Python tests pass if exposed, performance is measured where relevant, unsupported cases are documented, and a completion report is supplied.

### Native Verilog lvalue and assignment subset (M7.13)
- [x] Identifier, packed bit, constant part, indexed `+:`/`-:`, and nested
  concatenated lvalues; ascending and descending declared ranges.
- [x] Runtime bit/indexed-base evaluation; unknown and unsupported indices;
  legal out-of-range read/write behavior; select width and arithmetic bounds.
- [x] Assignment resizing/truncation, signed and unsigned extension,
  two-state/four-state conversion, X/Z behavior, concatenation splitting,
  preservation of untouched bits, and transactional rollback.
- [x] Boundary tests for source and target widths 1 through 65 plus 128 bits.
- [x] Function/task local assignment integration, selected and concatenated
  task output actuals, selected inout copy-in/out, and explicit `ref` rejection.
- [x] Slang comparison against constant-function packed part-select assignment;
  task output effects are expected-value tests only.
- [ ] Unpacked arrays, multidimensional packed selects, global callable side
  effects, `ref` aliasing, nonblocking scheduling, and general simulation.

### Native Verilog procedural control flow (M7.14)
- [x] Structured analysis: loop nesting, return presence, unsupported nodes,
  and source-located break/continue placement validation.
- [x] If/else four-state procedural truth behavior; exact case, casez, casex,
  multiple expressions, first match, and default.
- [x] For, while, do-while, repeat, and bounded forever loops; initialization,
  condition, body, update order, nested loops, break, continue, and return.
- [x] Shared statement-step accounting, including empty forever loops and
  nested callable execution; explicit budget-exhaustion outcomes.
- [x] Slang constant-function differential coverage where public APIs provide
  a valid value oracle; task behavior checked with expected values.
- [ ] Nonblocking scheduling, wait, process simulation, global side effects,
  and complete SystemVerilog execution.

### Native process analysis and synthesis eligibility (M7.15)
- [x] Source-ordered discovery for continuous assignments, generic `always`,
  `always_comb`, `always_ff`, `always_latch`, and `initial` processes.
- [x] Semantic-identity read/write dependencies, selected ranges, dynamic
  selection uncertainty, function dependency summaries, and separate implicit
  sensitivity sets.
- [x] Path-sensitive packed-bit assignment coverage for branches, cases,
  concatenations, and statically justified loop execution; conservative
  unknown coverage and latch/storage indication.
- [x] Single-edge clock recognition, structurally matched synchronous and
  asynchronous reset candidates, blocking/NBA categories, and ambiguous
  multiple-edge detection.
- [x] Cross-process procedural writer diagnostics, explicit
  `Eligible`/`Unsupported`/`Indeterminate` results, expected-result tests, and
  Slang process-inventory comparison.
- [ ] Elaborated hierarchy-wide driver merging, RTL lowering, cell generation,
  process execution, and nonblocking assignment scheduling.

### Independent native AST-to-RTL lowering (M7.16)
- [x] Explicit opt-in API consumes the same native syntax/elaboration/type/
  eligibility identity and returns an existing Module only on full success.
- [x] Flat ports/signals, declared packed range ordering, continuous
  assignments, known constants, supported Boolean/arithmetic expressions,
  constant selects, concatenations, and diagnostics for unsupported hardware.
- [x] Ordered blocking combinational assignment lowering, if/case muxing,
  static lvalues, and incomplete-assignment rejection.
- [x] Canonical one-clock NBA process lowering with DFF, negative-edge clock
  inversion, hold/enable muxing, synchronous reset muxing, and supported
  constant asynchronous DFFR/DFFS reset forms.
- [x] Structural tests, failure atomicity, hierarchy/undriven diagnostics,
  native-vs-Slang AND truth-table differential, optional Yosys AND
  equivalence proof, and sequential structure tests.
- [x] Dynamic packed bit-select reads with a proven unsigned contiguous,
  power-of-two-aligned range inside a declared one-dimensional packed range;
  supports constants, identifiers, concatenation/zero extension, unambiguous
  continuous computed selectors, and overflow-safe constant offsets. MUX2-tree
  lowering covers ascending/descending nonzero ranges, exhaustive 2/4/8-way
  checks, sampled 16-way checks, and Yosys equivalence coverage.
- [ ] General arithmetic including multiply/divide, X/Z hardware semantics,
  dynamic writes/part-selects, local procedural storage, latches, hierarchy, inout,
  arbitrary resets, technology mapping, and general/sequential equivalence.

### Native RTL verification and differential validation (M7.17)
- [x] Two-state functional semantics and independently authored truth tables
  for AND2, OR2, XOR2, INV, MUX2, BUF, TIEHI, and TIELO.
- [x] Deterministic fanout/topological evaluation, declared port-bit order,
  constant propagation, cutpoints, and malformed/multiple/undriven/cyclic
  netlist diagnostics.
- [x] Bounded sequential functional evaluator with explicit initial state,
  positive/negative edge transitions, simultaneous old-state updates, hold,
  enable, synchronous reset, async active-high/low reset/set, frame-zero reset,
  unsupported-cell and resource-limit diagnostics.
- [x] Seven native combinational fixtures exhaustively checked against
  independent expected functions over 4,856 total vectors. Slang differential
  agrees on 160 vectors across two fixtures; five report explicit Unavailable
  because generic mapping produces unsupported cells or operations.
- [x] Deterministic seed 0x4d3731375eed: 24 depth-three trees and four
  procedural branch expressions checked over 512 vectors; negative,
  failure-atomicity, and deterministic-output tests.
- [x] Three Yosys combinational proofs and deliberate inequivalence detection
  for a wrong AND2 model.
- [ ] Sequential formal equivalence, unbounded sequential proof, general
  SystemVerilog event scheduling, four-state netlist evaluation, and generic
  cells outside the eight-cell two-state evaluator subset.

### Cross-platform stabilization (M7.17.1)
- [x] Local Apple Silicon M1 full native Release CTest: 59 passed; 10 AVX2
  tests skipped because the ARM64 CPU does not support AVX2.
- [x] Forced Generic and NEON backend verification: seven focused tests each,
  all passed without skips.
- [x] Local EDA-only Release build with Tensor, Acceleration Core, and
  compatibility disabled: 38 CTests passed; the AVX2 lexer test skipped.
- [x] Prescribed EDA Python regressions: 216 passed. Full Python suite:
  1,212 passed, with 225 skips classified as unavailable AVX2 cases,
  Generic-only broadcast coverage, or absent optional extension variables.
- [ ] Ubuntu x86-64 GCC and Clang full native and EDA-only workflow execution
  for the M7.20 native parser/mapper contract (configured; remote run pending).
- [ ] Native Ubuntu ARM64 build and strict NEON execution for the M7.20 native
  parser/mapper contract (configured on `ubuntu-24.04-arm`; remote run pending).
- [ ] Hosted macOS ARM64 native and EDA-only workflow execution for the M7.20
  native parser/mapper contract (configured; remote run pending).
- [ ] Windows x86_64 MSVC EDA-only build, Slang-free wheel linkage audit, clean
  install, and native contract execution (configured; remote run pending).
- [ ] Linux AVX2 job executes all ten AVX2 CTests and the portable native EDA
  contract without skips (configured; remote run pending).
- [ ] EDA AddressSanitizer/UndefinedBehaviorSanitizer job passes.

### Comparative frontend profiling (M7.18.1)
- [x] Opt-in native and Slang syntax, semantic, and complete-path Release benchmarks.
- [x] Deterministic 11-family workload generator with SHA-256 manifests and
  tested 1 KiB, 10 KiB, and 100 KiB tiers.
- [x] Machine-readable nullable metrics, comparison validation, Markdown/JSON
  aggregation, macOS/Linux profiler orchestration, and schema tests.
- [x] Apple M1 baseline across 33 workloads, native scalar/NEON comparison,
  Slang hierarchical trace, peak RSS/cycle/instruction measurements, and a
  native sampled profile.
- [ ] Linux x86-64 AVX2 timing/counters, 1 MiB/4 MiB controlled complete runs,
  Slang top-20 sampled functions, and allocation traces.

### Indexed semantic reference resolution (M7.18.2)

- [x] Sorted ordinal index preserves source identity, duplicate insertion
  order, missing references, null declarations, deterministic rebuilds, and
  move stability against the linear oracle.
- [x] Exact 33-workload Apple M1 before/after comparison with matching hashes,
  Release/scalar configuration, median and p95 timings, construction cost,
  storage estimate, peak RSS, and fresh sampled profile.
- [x] Full Release CTest: 62 passed and 10 expected ARM64 AVX2 skips. EDA-only
  Release: 41 passed and one expected AVX2 skip. Selected Python EDA suites
  passed.
- [ ] Linux x86-64 optimized scalar/AVX2 timing, `perf` counters and profile,
  plus sanitizer reruns on suitable CI hardware.
