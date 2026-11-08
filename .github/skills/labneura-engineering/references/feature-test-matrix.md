# LabNeura Feature and Backend Test Matrix

Use this checklist for every feature and every backend implementation.

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

### BOOL logical operations — M3.4.1/M3.4.2/M3.4.3 `logical_not`
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
- [ ] Execute AVX2 tests on an x86_64 host with AVX2 support; ARM skips and
  cross-compilation do not count as runtime validation.
- [ ] AVX2 `logical_and`, `logical_or`, and `where()`.

The current Tensor API creates contiguous layouts only; non-contiguous-view
coverage is not applicable yet. M3.4.1 provides the Generic kernel and M3.4.2
adds NEON. M3.4.3 adds AVX2 implementation and tests; x86_64 runtime validation
is pending. Apple Silicon M3.4.2 results: focused CTests had 4 passes and 2
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
