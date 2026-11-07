---
name: labneura-engineering
description: Use for LabNeura tensor API, nD storage, numerical operations, broadcasting, CPU/GPU backends, runtime, testing, benchmarking, architecture and performance tasks.
---

# LabNeura Engineering Skill

You are an engineering agent working on LabNeura, a C++ numerical-computing and tensor library with Python bindings. Make small, correct, test-backed changes. Build broad functionality and excellent execution together.

Read the relevant references before substantial work:
- `references/roadmap.md`: product scope, architecture and milestone order.
- `references/feature-test-matrix.md`: required tests and definition of done.
- `templates/feature-task.md`: template for splitting work into bounded tasks.

## Mission

LabNeura has two equally important goals:

1. **Functionality:** nD tensors, shape manipulation, broadcasting, elementwise math/logical operations, reductions, linear algebra and ML-oriented operations.
2. **Execution:** storage/memory management, CPU threading, SIMD, backend dispatch, benchmarks and eventual GPU execution.

Do not build only a handful of fast kernels while neglecting the tensor API. Do not add a broad API surface without tested semantics and a backend plan. CPU performance is the immediate priority; GPU is a planned future capability.

## Mandatory workflow for every task

### 1. Inspect
- Check `git status --short` and relevant diffs.
- Locate the public API, implementation, dispatch, tests, CMake registration, Python bindings and benchmark code.
- Read existing conventions. Do not invent parallel abstractions without a concrete need.
- Identify dependencies and assumptions. Ask one focused question if required semantics are genuinely ambiguous.

### 2. Define the contract before coding
Specify:
- API and expected input/output shapes.
- Supported dtypes and backends.
- Broadcasting and shape rules.
- Numerical semantics: promotion, overflow/saturation, rounding, NaN/Inf and divide-by-zero as relevant.
- Allocation, mutation, aliasing and view behavior.
- Invalid-input errors.
- Synchronous/asynchronous behavior.
- Required tests and reference comparison.

Do not infer semantics from names alone. Follow the documented LabNeura contract and NumPy/PyTorch conventions where those are the intended reference.

### 3. Tests first or alongside implementation
- Add focused tests before or in the same change as implementation.
- Implement and test Generic/reference semantics first.
- When adding an optimized backend, add explicit tests for that backend in the same change.
- Add Python tests whenever public Python behavior changes.
- Include invalid-input and edge-case tests.
- Register native tests with the existing CMake/CTest setup.
- Automatic dispatch alone does not prove every backend was tested.

### 4. Implement one layer at a time
Preferred progression:
1. Shared semantic helper, only if reusable.
2. Generic implementation.
3. Generic tests.
4. One optimized backend.
5. That backend's explicit correctness/parity tests.
6. Benchmark.
7. Continue to the next backend only after the current stage is green.

Do not implement several backends in one speculative patch. Do not duplicate shape/broadcast logic inside every arithmetic kernel.

### 5. Validate
- Build the narrow target.
- Run focused native tests.
- Run focused Python tests if relevant.
- Run the complete relevant native and Python suites.
- Run existing formatting/static checks.
- Use sanitizers or concurrency tests when memory/thread behavior changes.
- Check `git diff --check` and inspect the final diff.
- Preserve unrelated work.

### 6. Benchmark honestly
For performance changes:
- Same operation, dtype, shape, inputs and numerical semantics.
- Same allocating/in-place mode.
- Record backend and thread count.
- Use warmups and repeated batches.
- Report median, IQR and p95.
- Include hot/rotating memory regimes when relevant.
- Separate native-kernel from Python API timing.
- Record CPU, OS, compiler, flags and competitor versions.
- Verify correctness before timing.
- If PyTorch/NumPy is unavailable, state that; never imply a comparison was made.
- Do not claim a universal 10% win from a selected tensor size. Report per operation/dtype/size and aggregate scores separately.

## Tensor model rules
- Use one nD Tensor abstraction. 1D/2D/3D are ranks/shapes of the same type, not separate tensor classes.
- Separate logical metadata (shape, strides, offset, dtype, layout) from physical storage and ownership.
- Centralize shape/stride/overflow invariants.
- Start with contiguous tensors; add non-contiguous views only with explicit alias/lifetime semantics.
- Never assume `numel == storage_size` for views.
- Check shape-product and byte-size overflow.
- Avoid accidental copies in reshape/transpose/view operations.
- Treat raw host pointers as CPU-only; never expose opaque GPU memory as a normal host pointer.
- Broadcasting is a shared shape/iteration facility, not repeated ad hoc logic in each kernel.

## Numerical/API rules
- Define integer overflow/saturation behavior for every dtype/operation.
- Plan a Boolean dtype for comparison/logical results; do not ambiguously encode booleans as INT8.
- Define mixed-dtype and scalar promotion rules.
- Define division-by-zero and floating special-value behavior.
- In-place operations must reject broadcasting that would expand the destination.
- Define exact alias and partial-overlap behavior.
- Keep C++ and Python semantics consistent.
- Prefer actionable exceptions over undefined behavior.

## Backend rules
- Generic is the semantic reference, not necessarily a scalar-performance path.
- NEON and AVX2 must match Generic's documented behavior.
- CPU thread pool/`parallel_for` and CPU allocators remain CPU-scoped.
- Track support by operation × dtype × backend × layout × mode.
- Unsupported combinations must fail clearly or use an explicitly documented dispatch fallback.
- Do not force CPU workers and GPU queues into one scheduler abstraction.
- Add Metal/CUDA/HIP only through a concrete vertical slice with actual hardware validation. Do not add speculative GPU frameworks now.

## Performance rules
- Correctness first, performance second.
- Profile before optimizing.
- Change one performance variable at a time.
- Do not assume more threads or SIMD always helps.
- Measure allocation, conversion, memory bandwidth, cache locality, dispatch and Python overhead separately where possible.
- Keep previous benchmark result directories immutable.

## Scope control
- One task normally changes one feature or optimization.
- Do not ask an agent to implement an entire roadmap phase in one prompt.
- No broad refactors, dependency installation, compiler changes or public API changes outside scope without approval.
- Do not commit or push unless requested.
- If a backend cannot run on the current machine, report that and identify the required CI/hardware validation.
- Stop after the requested scope and wait for approval before starting the next task.

## Required completion report
1. Exact change and semantic contract.
2. Files changed.
3. Generic and optimized backend coverage.
4. Tests added and exact commands/results.
5. Tests not run and why.
6. Benchmarks and regressions, if relevant.
7. Known limitations.
8. Confirmation unrelated worktree changes were preserved.
9. Confirmation no commit/push was made unless requested.

A feature is not complete merely because it compiles. Its semantics, tests, backend status and limitations must be explicit.
