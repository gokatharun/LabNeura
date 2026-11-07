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

## Storage and views
- [ ] Shape/stride/offset invariants and overflow.
- [ ] Contiguous layout.
- [ ] Transpose view correctness.
- [ ] Reshape view only when legal.
- [ ] Clone/copy independence.
- [ ] View shares storage as documented.
- [ ] Mutation and lifetime semantics.
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
