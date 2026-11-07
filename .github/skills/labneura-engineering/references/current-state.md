# LabNeura Current Project State

Last verified: 2026-11-07. Treat this as a continuity reference, not a
substitute for inspecting the current source, tests, worktree, and benchmark
artifacts.

## Project identity

LabNeura is a C++ numerical-computing and Tensor library with Python bindings.
Its long-term direction is broad nD Tensor functionality, high-performance CPU
execution, and eventual GPU execution. CPU functionality and performance remain
the immediate priorities; GPU implementation is not active work.

## Verified completed work

### Tensor and CPU execution

- The library contains Generic, ARM NEON, and x86 AVX2 CPU backend
  implementations. Backend selection uses runtime CPU-feature detection and
  supports explicit backend selection through `LABNEURA_BACKEND`. A backend
  must be both compiled and supported by the current CPU before it can be
  selected.
- Tensor modes implemented at the library boundary are FP32, FP16, INT16, and
  INT8. FP64 and INT32 are explicitly rejected. Arithmetic operations are
  ADD, SUB, MUL, and DIV, but support is not uniform across every
  dtype/backend/operation combination: Generic arithmetic is implemented for
  FP32 and INT8, while NEON and AVX2 have additional FP16/INT16 code. Check the
  relevant backend and tests before claiming a particular combination.
- Tensor supports contiguous arbitrary-rank construction using
  `Tensor::from_flat` / `Tensor.from_flat(data, shape, mode="auto")`, including
  1D and higher-rank shapes. Existing vector, scalar, and 2D nested-row
  construction remain available. Shape product and Tensor byte-size overflow
  are checked at the Tensor construction boundary. Empty shape denotes a
  scalar; zero-sized dimensions produce zero elements.
- `reshape(shape)` and `flatten()` are implemented in C++ and Python.
  `reshape` requires an equal element count; `flatten` returns shape `[numel]`.
  Both preserve dtype and row-major element order and return independent deep
  copies. Scalar and zero-element tensors follow the existing shape rules.
- Elementwise Tensor operations require identical shapes. There are no
  implicit broadcasts.
- `add_into` is implemented for FP32 and exposed as the Python module function
  `labneura.add_into(a, b, out)`. It writes into caller-owned output storage,
  requires matching shape and backend, and supports exact input/output aliasing.
- Tensor execution configuration and `parallel_for` are implemented. The
  runtime has a persistent worker pool and defaults to one execution thread.
  Tensor arithmetic is not currently wired to `parallel_for`; do not imply that
  changing the execution thread count parallelizes Tensor operations.
- Native tests cover runtime/thread configuration, `parallel_for`, Tensor
  operations, `add_into`, and nD construction. Python tests cover the exposed
  APIs. The GitHub CI workflow includes x86_64 and aarch64 native SIMD test and
  benchmark jobs.

### Ownership, shape, and Python behavior

- A `Tensor` owns a `std::unique_ptr<TensorBackend>` and stores its shape
  metadata itself. Tensor and backend hold `std::shared_ptr<TensorStorage>`
  references to a non-virtual control block that owns the allocation and its
  matching deallocator. The control block records dtype, logical element
  count, physical allocation capacity, and alignment; backend pointers are
  non-owning aliases used by existing kernels.
- Tensor copy construction/assignment clone the backend, so copies have
  independent storage. Move operations transfer ownership.
- Backend `clone()` allocates fresh storage and copies only logical elements,
  retaining padded capacity and alignment. The shared control block can outlive
  one owner, but no Tensor view API currently shares it.
- M2.2.1 ownership validation passed on Apple Silicon: the full native CTest
  run reported 15 passed and 6 AVX2-specific skips; the full Python suite
  reported 991 passed, 27 hardware/backend skips, and 3 pytest return-value
  warnings. Python tests used the freshly rebuilt extension.
- Storage is a flat, row-major contiguous buffer. Each Tensor records
  row-major strides in elements and a zero storage offset. Non-contiguous
  layouts and Tensor views are not supported.
- Python exposes shape as a tuple. Existing `data_*` accessors return Python
  lists, which copy values out; they are not zero-copy views of Tensor storage.
- `reshape` and `flatten` clone the backend before changing shape metadata;
  neither operation creates a view or aliases the source.

### Allocation safety

- The shared aligned allocator checks byte-count multiplication, validates
  alignment, throws on allocation failure, and no longer falls back to
  potentially unaligned `malloc` storage. A zero-count request returns a small
  valid aligned allocation so empty-range pointer arithmetic is safe.
- AVX2 and NEON padded element counts use checked round-up while retaining the
  logical element count separately from padded capacity.
- The focused allocator-safety native test and full native test suite passed
  on the Apple Silicon development machine. AVX2-specific allocation tests
  skipped there because AVX2 is unavailable.
- Generic, NEON, and AVX2 allocations now use the same checked storage
  allocation path. Logical element count remains distinct from SIMD-padded
  capacity; zero-sized storage remains non-null and has zero logical capacity.

## Actual backend benchmark validation

These are observed results, not estimates. The native CI report is a historical
run and its runner hardware can vary; use it as validation evidence, not as a
portable performance guarantee.

| Benchmark | Hardware/backend | Case | Generic | SIMD | Observation |
|---|---|---|---:|---:|---|
| Native CI, in-place FP32 ADD | aarch64 / NEON | 1,048,576 elements | 175,725 ns | 173,228 ns | 1.01x |
| Native CI, in-place INT8 ADD | aarch64 / NEON | 1,048,576 elements | 1,001,228 ns | 41,593 ns | 24.07x |
| Native CI, in-place FP32 ADD | x86_64 / AVX2 | 1,048,576 elements | 135,852 ns | 106,548 ns | 1.28x |
| Native CI, in-place INT8 ADD | x86_64 / AVX2 | 1,048,576 elements | 1,313,742 ns | 28,954 ns | 45.37x |

The native benchmark checks values against an independent scalar reference and
only adds NEON/AVX2 when runtime feature detection reports support. Its archived
summary is [benchmarking/results/latest.md](../../../../benchmarking/results/latest.md),
which identifies commit `d7f5ee0d3724` and GitHub Actions run `42323991089`.
The summary reports medians and p95 for native in-place FP32 and INT8
ADD/SUB/MUL/DIV; raw timing is machine- and run-dependent.

On Apple M1, Python/API benchmarks explicitly selected NEON and checked
correctness before timing. The latest stored allocating/in-place comparison
includes Generic, NEON, NumPy, and PyTorch CPU; the reusable-output benchmark
separates allocation from preallocated `add_into` and compares against
`np.add(..., out=...)` and `torch.add(..., out=...)`. Results and methodology:
[current FP32 benchmark report](../../../../benchmarking/results/fp32-allocating-inplace-current-m1-20261107T0604/report.md)
and [reusable-output benchmark report](../../../../benchmarking/results/add-into-m1-final-20261106T1622Z/report.md).
For example, the reusable-output hot-cache 1M-element FP32 ADD medians were
150.116 us for NEON, 156.825 us for NumPy, and 151.039 us for PyTorch. The
separate rotating-cache case was 236.142 us, 245.131 us, and 242.616 us,
respectively. These are Python/API measurements, not isolated kernel timings.

## Current architecture

```text
Python API (pybind11)
  ├─ constructors / from_flat / shape / copied data lists
  ├─ arithmetic and add_into
  └─ CPU thread configuration / backend selection
          │
          ▼
Tensor
  ├─ shape, row-major strides, zero storage offset; contiguous-only
  ├─ shared TensorStorage owner + unique TensorBackend execution object
  └─ shape validation and elementwise operation entry points
          │
          ▼
TensorBackend interface
  ├─ raw typed data access, deep clone, shared TensorStorage reference
  ├─ allocating operation hook with clone-plus-in-place fallback
  └─ optional direct-output hook (used by supported operations)
          │
          ├─ Generic CPU backend
          ├─ ARM NEON CPU backend
          └─ x86 AVX2 CPU backend

TensorStorage: dtype + logical size + padded capacity + alignment
               + owned allocation + matching deleter

CPU runtime: thread configuration + parallel_for + persistent worker pool
            (not currently used by Tensor arithmetic)
```

Tensor metadata is separate from storage and execution. TensorStorage is
shared-owned internally, while public Tensor copies still deep-copy. Backend
APIs expose host pointers. This is a CPU design today, not an opaque/device-
memory abstraction.

## Pending work

These remain roadmap items, not implemented capabilities:

- `transpose` and axis permutation.
- Tensor views and their lifetime/alias semantics (shared storage ownership is
  implemented, but no public view API uses it yet).
- Non-contiguous strides, nonzero offsets, indexing, and slicing.
- Shared broadcasting and broadcast-aware elementwise operations.
- Reductions.
- Vector/matrix and batched matrix operations.
- Additional ML mathematics and initialization utilities.
- GPU vertical slice and eventual Metal, CUDA, or HIP implementations.

The detailed feature sequence remains in [roadmap.md](roadmap.md); avoid
prebuilding GPU abstractions before an actual backend requires them.

## Storage/layout metadata and contiguous reshape views (M2.2.1–M2.2.3 complete)

M2.2.1 introduces a non-virtual shared `TensorStorage` control block that owns
one backend-allocated buffer and its deleter. Each Tensor keeps its own shape
metadata and an internal shared owner; each backend keeps a shared reference
for kernel access. Ordinary Tensor copy construction/assignment and existing
`reshape()`/`flatten()` behavior remain deep copies.

The control block separates logical element count from physical allocation
capacity and records dtype and alignment. Backend kernels continue to operate
on flat buffers and do not interpret Tensor shape metadata. M2.2.2 adds
per-Tensor row-major strides and a zero storage offset, both derived/initialized
for supported constructors and shape transforms. `numel()` is calculated from
shape. Stride suffix products use checked arithmetic; shapes whose row-major
strides cannot be represented are rejected with overflow. `is_contiguous()`
reports true for all currently representable Tensor layouts.

`reshape_view(shape)` is available in C++ and Python for contiguous source
tensors. It shares the `TensorStorage` allocation but creates a distinct
backend execution object and independent shape/stride metadata. Its backend
retains the view's logical element count separately from the storage's logical
extent and padded physical capacity. The shape must have exactly the source
element count, and bounds are checked against logical storage, not padded
capacity. The view retains storage after the source is destroyed, and
mutations through either Tensor are visible through the other. Ordinary Tensor
copy construction/assignment and existing `reshape()`/`flatten()` remain deep
copies. Python data accessors still return copied lists.

Only contiguous reshape aliases are supported. Non-contiguous strides,
nonzero-offset layouts, slicing, transpose/permutation and broadcasting remain
unimplemented.

## Known limitations and unresolved validation

- No speculative GPU abstractions or GPU implementation are present.
- `reshape_view()` only supports contiguous aliases with the existing zero
  offset. Slicing, non-contiguous views, transpose/permutation and
  view-specific indexing remain pending.
- There are no non-contiguous layouts or broadcasting; arithmetic requires
identical shapes.
- Per-backend dtype/operation support is uneven. Generic FP16/INT16 are not
  supported for arithmetic; verify each backend operation rather than assuming
  every enum combination is implemented.
- AVX2 has historical x86_64 CI benchmark evidence, but the current Apple
  Silicon machine cannot compile or execute AVX2 instructions. Run the current
  AVX2 native regression suite on an AVX2-capable x86_64 runner after relevant
  backend changes.
- The F16C-specific AVX2 FP16 path is conditionally selected by feature
  detection and has a dedicated regression test, but durable local artifacts
  do not prove that the F16C vector path executed on a feature-capable runner.
  Confirm that path from an x86_64 CI/test log or a suitable host before
  claiming F16C runtime validation.
- The most recent Apple Silicon native allocation-safety run skipped AVX2
  cases. Its utility test passed; it does not establish AVX2 constructor
  runtime safety.
- The FP32 performance objective remains at least 10% faster than the faster
  comparable NumPy/PyTorch CPU path on agreed workloads. This has not been
  established broadly. FP32 performance optimization is intentionally paused;
  do not resume it without a concrete request or new evidence.
- GenericBackend's separate raw allocation helper is not covered by the shared
  aligned-allocator hardening. See the allocation-safety note above.

## Validation snapshot

On the Apple Silicon development machine, M2.2.3 validation completed with:

- Native build: all configured targets built.
- Full CTest suite: 15 passed, 6 skipped. AVX2-only cases skipped because the
  host is Apple M1; Generic and NEON tests ran.
- Python suite using the rebuilt local extension:
  `PYTHONPATH=build/local-python .venv/bin/python -m pytest -q` completed
  successfully, with 27 hardware/backend skips and 3 existing pytest
  return-value warnings.
- `git diff --check` passed.

AVX2 native tests skipped because this host is Apple M1 and cannot execute
AVX2. The new reshape-view tests ran on Generic and NEON; AVX2 reshape-view
behavior remains unverified locally. See
[feature-test-matrix.md](feature-test-matrix.md) for per-feature definition of
done.

The subsequent M2.1 reshape/flatten milestone was also validated on Apple
Silicon:

- Native build completed; full CTest passed all runnable tests, with six
  AVX2-only tests skipped on this host.
- Focused Python nD/reshape tests passed (37 tests); the full Python suite
  completed successfully with the existing warnings and hardware-specific
  skips.
- Generic and NEON reshape/flatten native tests ran. AVX2 runtime behavior
  remains unverified locally because this host is Apple M1.
