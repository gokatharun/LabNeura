# LabNeura Current Project State

Last verified: 2026-11-08. Treat this as a continuity reference, not a
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
- Numeric Tensor modes implemented at the library boundary are FP32, FP16,
  INT16, and INT8; BOOL is also available as a distinct one-byte dtype for
  canonical boolean storage. FP64 and INT32 are explicitly rejected. Arithmetic
  operations are ADD, SUB, MUL, and DIV, but support is not uniform across every
  dtype/backend/operation combination: Generic arithmetic is implemented for
  FP32, INT8, and INT16, while NEON and AVX2 have additional FP16/INT16 code.
  The six Tensor comparisons have Generic, NEON, and AVX2 implementations for
  matching FP32, FP16, INT16, and INT8 inputs, returning BOOL. AVX2 comparison
  runtime validation passed on an x86_64 AVX2 host in GitHub Actions run
  `42390563490` for implementation commit
  `895b7633e1cd6f48b1459397cd664ee1d4dd2c08`.
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
- Generic FP32, INT8, and INT16 elementwise arithmetic supports NumPy-style
  broadcasting. Non-Generic backends retain equal-shape behavior; there is no
  implicit mixed-dtype promotion.
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
  one owner; reshape and first-axis slice/select views explicitly share it.
- M2.2.1 ownership validation passed on Apple Silicon: the full native CTest
  run reported 15 passed and 6 AVX2-specific skips; the full Python suite
  reported 991 passed, 27 hardware/backend skips, and 3 pytest return-value
  warnings. Python tests used the freshly rebuilt extension.
- Storage is a flat, row-major contiguous buffer. Each Tensor records strides
  and a storage offset in elements. Constructors use row-major strides and
  offset zero; contiguous reshape and first-axis slice/select views may have a
  nonzero offset.
- Python exposes shape as a tuple. Existing `data_*` accessors return Python
  lists, which copy values out; they are not zero-copy views of Tensor storage.
- BOOL construction accepts bool or numeric sequences in Python and byte or
  numeric sequences in C++. Zero values normalize to 0 and nonzero values to 1.
  `data_bool()` exposes `std::uint8_t` storage in C++ and returns copied Python
  bool values. BOOL arithmetic, BOOL-to-BOOL comparisons, `logical_and`,
  `logical_or`, and `where()` are not implemented. `logical_not` has Generic,
  NEON, and AVX2 implementations; AVX2 x86 runtime validation is pending.
- `reshape` and `flatten` clone the backend before changing shape metadata and
  remain deep copies. `reshape_view`, `slice_first_axis`, and
  `select_first_axis` create storage-sharing views.

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
  ├─ shape, row-major strides, element-based offset; contiguous-only layouts
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

### BOOL dtype foundation (M3.3.2)

BOOL is a distinct Tensor dtype backed by one `std::uint8_t` per logical
element. The shared storage control block accounts for BOOL element width and
deallocation; Generic, NEON, and AVX2 allocations retain separate logical
element counts and backend padding. Constructors normalize zero values to 0 and
nonzero byte or numeric values to 1. Python sequence construction accepts bool
or numeric values, and Python `data_bool()` returns copied Python bools.

Mutable C++ `data_bool()` access canonicalizes existing logical elements
before returning its pointer; callers must write only 0 or 1. Const access
rejects noncanonical bytes if a retained mutable pointer was used to write an
invalid value. BOOL deep copies and views preserve backend and storage lifetime.
BOOL arithmetic, BOOL-to-BOOL comparisons, logical operations, and `where()`
remain unsupported. Generic numeric comparisons are implemented as M3.3.3
below; NEON numeric comparison kernels are implemented in M3.3.4. AVX2
numeric comparison kernels and x86_64 runtime validation are complete in
M3.3.5.

### Tensor comparisons (M3.3.3 Generic; M3.3.4 NEON; M3.3.5 AVX2)

`Tensor.eq/ne/lt/le/gt/ge` and the matching Python methods compare two tensors
of the same FP32, FP16, INT16, or INT8 dtype and return a newly allocated,
contiguous BOOL Tensor. Comparisons use the shared `BroadcastPlan`, support
scalar through nD shapes, zero-sized outputs, and offset views, and do not
mutate inputs. INT8/INT16 are signed; FP16 compares decoded binary16 values.
FP32/FP16 use IEEE comparisons (NaN is unequal to itself and ordered
comparisons with NaN are false; signed zeros compare equal). BOOL input
comparisons and mixed-dtype promotion are rejected. Generic, NEON and AVX2
execute their own kernels; dispatch rejects mixed-backend inputs and has no
silent cross-backend fallback. NEON vectorizes FP32 and decoded FP16 in four
lanes, INT16 in eight lanes, and signed INT8 in sixteen lanes. AVX2 vectorizes
FP32 and software-decoded FP16 in eight lanes, INT16 in sixteen lanes, and
signed INT8 in thirty-two lanes. AVX2 FP16 does not require F16C. Integer
comparison masks are compressed to canonical bytes while preserving lane
order. Vector tails are handled without over-reading. Broadcast coordinates
are gathered using the shared plan, so scalar/rank extension, zero-sized
outputs and offset views do not materialize inputs. Generic, NEON and AVX2
comparison sources are built without fast-math to preserve IEEE NaN/Inf
behavior.

NEON comparison validation on Apple Silicon:

- The native comparison test explicitly selects Generic and NEON and compares
  all six operations across FP32, FP16, INT16 and INT8 for equal-shape,
  broadcast, signed-boundary and floating special-value inputs.
- Native NEON tests also cover scalar and higher-rank broadcasts, zero-sized
  outputs, offset views, BOOL canonical values, and invalid inputs.
- Python comparison tests explicitly select each backend and check NumPy
  results plus Generic/NEON parity for all operations and dtypes.

Apple Silicon validation:

- Focused native comparison CTest: Generic and NEON passed; AVX2 skipped
  because this host is ARM64.
- Focused native comparison/broadcast/BOOL/FP16 regression CTest: 8 passed,
  3 AVX2-only tests skipped.
- Full native CTest: 21 passed, 8 AVX2-specific tests skipped.
- Focused Python comparison suite: 116 passed and 60 AVX2-specific tests
  skipped on ARM, including explicit Generic and NEON selection and parity.
- Focused Python broadcast/BOOL/FP16 regressions: 170 passed, 125 existing
  backend-specific skips.
- Full Python suite on the normal ARM host: 1158 passed, 213 skipped, and 3
  existing pytest return-value warnings. AVX2-specific tests account for
  architecture-gated skips.
- AVX2-specific tests were skipped locally because this host is ARM64.

x86_64 AVX2 validation:

- GitHub Actions run `42390563490` on implementation commit
  `895b7633e1cd6f48b1459397cd664ee1d4dd2c08` completed successfully.
- The explicit `tensor_comparison_avx2` native CTest passed. The full native
  CTest run had 29 registered tests, 0 failures, and 7 expected NEON-only
  skips on x86_64.
- The full Python suite ran with AVX2 selected: 1156 passed, 215 skipped, and
  3 warnings. AVX2 comparison Python tests passed; the skips include
  platform/backend-specific cases.
- AVX2 arithmetic regression passed. FP16/F16C dispatch and AVX2 FP16
  arithmetic passed, with logs confirming both the F16C vector kernel and
  scalar fallback ran.
- The ARM64 CI job passed its Generic and NEON comparison CTests; AVX2 was
  skipped on ARM64 as expected.
- These are functional correctness results; no comparison performance
  improvement is claimed.

`git diff --check` passed for the documentation update.

### BOOL logical_not (M3.4.1 Generic, M3.4.2 NEON, M3.4.3 AVX2)

`Tensor::logical_not()` and Python `Tensor.logical_not()` accept BOOL tensors
and return independent, contiguous BOOL tensors with the same shape. The
Generic inverts canonical values (`0` to `1`, `1` to `0`) without mutation or
conversion. NEON uses 16-byte vector comparisons and canonicalizes each vector
mask to BOOL bytes, followed by a scalar tail. AVX2 uses 32-byte unaligned
loads/stores, compares against zero, masks with one to produce canonical BOOL
bytes, and handles the remaining elements with a scalar tail. All backends
support scalar and zero-element tensors and contiguous offset views.
Non-contiguous Tensor layouts are not currently provided. `logical_and`,
`logical_or`, and `where()` remain unsupported.

M3.4.2 validation on Apple Silicon:

- Focused BOOL/comparison CTests: 4 passed and 2 AVX2 tests skipped. Generic
  and NEON tests executed; `bool_dtype_neon` passed.
- Focused BOOL/comparison Python tests: 128 passed and 61 AVX2 cases skipped.
  The explicit NEON test executed, not skipped.
- Native BOOL CTests explicitly select Generic and NEON and test sizes 0–33,
  all vector tail lengths, scalar and empty tensors, canonical output,
  immutability, independent output storage, and an offset view whose storage
  offset exercises unaligned vector loads.
- Python BOOL tests explicitly select Generic and NEON and cover equivalent
  vector-boundary, scalar, empty, offset-view, repeat-call, and immutability
  cases.
- Full native CTest: 21 passed, 8 AVX2-specific tests skipped, 0 failed.
- Full Python suite: 1162 passed, 214 skipped, 3 existing warnings.

M3.4.3 AVX2 tests and dispatch have been added. Native and Python cases
explicitly select AVX2 and cover sizes 0–66 (including all 32 tail remainders),
canonical output, scalar/empty inputs, independent results, input immutability,
repeated calls, and an offset-33 view. This development host is ARM64, so the
AVX2 CTest and Python case are skipped here. An x86_64 host with AVX2 support
must run those tests before M3.4.3 can be marked complete.

Tensor metadata is separate from storage and execution. TensorStorage is
shared-owned internally, while public Tensor copies still deep-copy. Backend
APIs expose host pointers. This is a CPU design today, not an opaque/device-
memory abstraction.

## Pending work

These remain roadmap items, not implemented capabilities:

- `transpose` and axis permutation.
- Non-contiguous strides and layouts.
- Arbitrary-axis, negative-stride, reverse, and advanced indexing.
- Shared broadcasting and broadcast-aware elementwise operations.
- Reductions.
- Vector/matrix and batched matrix operations.
- Additional ML mathematics and initialization utilities.
- GPU vertical slice and eventual Metal, CUDA, or HIP implementations.

The detailed feature sequence remains in [roadmap.md](roadmap.md); avoid
prebuilding GPU abstractions before an actual backend requires them.

## Storage/layout metadata and contiguous views (M2.2.1–M2.3 implemented)

M2.2.1 introduces a non-virtual shared `TensorStorage` control block that owns
one backend-allocated buffer and its deleter. Each Tensor keeps its own shape
metadata and an internal shared owner; each backend keeps a shared reference
for kernel access. Ordinary Tensor copy construction/assignment and existing
`reshape()`/`flatten()` behavior remain deep copies.

The control block separates logical element count from physical allocation
capacity and records dtype and alignment. Backend kernels continue to operate
on flat buffers and do not interpret Tensor shape metadata. M2.2.2 adds
per-Tensor row-major strides and an element-based storage offset. Constructors
start at offset zero; `numel()` is calculated from shape. Stride suffix
products use checked arithmetic; shapes whose row-major strides cannot be
represented are rejected with overflow. `is_contiguous()` describes strides
independently of the storage offset.

`reshape_view(shape)`, `slice_first_axis(start, stop)`, and
`select_first_axis(index)` are available in C++ and Python. They share the
`TensorStorage` allocation but create distinct backend execution objects and
independent shape/stride/offset metadata. A backend alias carries its logical
element count and storage offset centrally; the base validates the requested
range against storage's logical extent, never its padded capacity. Backend
data pointers address the view's first element, while kernels remain bounded
by the view's logical size. AVX2 FP32 and INT16 kernels select aligned or
unaligned accesses from the actual backend data-pointer alignment, so valid
element offsets do not violate SIMD alignment assumptions.

`reshape_view` requires a contiguous source and an equal element count.
`slice_first_axis` accepts half-open forward ranges and preserves rank;
`select_first_axis` removes the first dimension and produces a scalar for a
1D source. These first-axis views remain contiguous and retain storage after
the source is destroyed. Mutations through views are shared. Ordinary Tensor
copy construction/assignment, `reshape()`, and `flatten()` remain independent
deep copies with offset zero. Python data accessors still return copied lists.
Exact in-place aliases are supported, while partially overlapping write ranges
are rejected to avoid order-dependent mutation.

The M3.1 internal broadcast planner combines two shapes and element-stride
vectors into a result shape and effective operand strides. It aligns axes from
the right, gives missing leading dimensions zero effective strides, and sets
expanded singleton dimensions to zero stride. It supports scalar and zero-sized
shapes, checks output element-count overflow, and does not allocate, mutate
Tensor metadata, or touch storage. The pure planner accepts non-contiguous
stride metadata, although Tensor currently exposes only contiguous views.

M3.2.1 has defined the proposed semantics for ADD/SUB/MUL/DIV, same-dtype
behavior, output dtype, saturation, integer and floating division, IEEE
special values, in-place shape restrictions, and API errors. M3.2.2 implements
FP32 ADD/SUB/MUL/DIV broadcasting on Generic; M3.2.3 adds Generic INT8
ADD/SUB/MUL/DIV broadcasting. Both use the same typed iterator and M3.1
planner. The iterator uses the M3.1
`BroadcastPlan` output shape/effective strides and each backend alias's element
storage offset; it indexes output in row-major order, checks accessed ranges
against TensorStorage logical size, and does not materialize expanded inputs.
Allocating unequal-shape results use independent Generic storage and contiguous
output metadata. In-place broadcasting is allowed only if the result shape
equals the destination shape; partial overlaps and broadcast scalar aliases
that would be overwritten during iteration are rejected. INT8 ADD/SUB/MUL use
int16 intermediates and clamp to `[-128, 127]`; INT16 operations use int32
intermediates and clamp ADD/SUB/MUL to `[-32768, 32767]`. Integer DIV truncates
toward zero, returns zero for a zero divisor, and safely saturates the minimum
value divided by `-1`. Same-shape operations and non-Generic mismatch behavior
retain their existing backend paths. NEON/AVX2 broadcasting and FP16
broadcasting remain unsupported.

M3.2.4 validation on Apple Silicon:

- Focused native `tensor_broadcast_generic`: passed.
- Full native CTest: 16 passed, 6 AVX2-only tests skipped.
- Focused Python INT16 broadcasting with Generic: 37 passed, 1 intentional
  non-Generic behavior check skipped. All Generic broadcast tests: 125 passed,
  3 intentional non-Generic behavior checks skipped.
- Full Python suite on the normal host backend: 1021 passed, 153 skipped, and
  3 existing pytest return-value warnings after updating stale tests that
  expected Generic INT16 support to be absent.
- The initial full Python suite with `LABNEURA_BACKEND=GENERIC` had 45
  failures. Every traceback ended with `FP16 data access not implemented for
  this backend` during FP16 tensor construction. They were FP16 constructors,
  accessors, arithmetic/operator, mixed-mode error, division, and corner-case
  coverage tests; no failure involved FP32, INT8, INT16, broadcasting, or
  storage aliasing. Generic FP16 remains outside this milestone.
- FP16-specific tests now skip explicitly when Generic is selected. Combined
  checks retain their FP32/INT8/INT16 coverage: the scalar-construction and
  backend-consistency tests conditionally omit only the FP16 case. Final
  Generic-forced full Python suite: 983 passed, 191 skipped, 0 failed, with
  3 existing pytest return-value warnings. Of the prior 45 failures, 43 are
  now intentional backend-specific skips and the two combined tests pass
  while checking supported dtypes.
- Focused Generic INT16 broadcasting: 37 passed, 1 intentional skip;
  focused Generic INT8 broadcasting: 37 passed, 1 intentional skip.
- AVX2 runtime validation is unavailable on this Apple Silicon host.

M3.2.2 validation on Apple Silicon: focused native Generic broadcasting test
passed; full CTest passed all 16 runnable tests with 6 AVX2-only tests skipped.
Focused Python broadcasting tests with `LABNEURA_BACKEND=GENERIC` passed 51
tests, with one intentional non-Generic-only case skipped. The full Python
suite against the rebuilt local extension passed 1019 tests, skipped 79
(including Generic-only cases when the normal NEON backend was selected), and
reported 3 existing pytest return-value warnings. AVX2 was not runtime-tested
on this ARM host.

Non-contiguous Tensor layouts, arbitrary-axis or reverse slicing, and
transpose/permutation remain unimplemented.

## Known limitations and unresolved validation

- No speculative GPU abstractions or GPU implementation are present.
- View APIs only support contiguous `reshape_view()` and first-axis
  slice/select operations. Negative/reverse slicing, arbitrary axes,
  non-contiguous views and transpose/permutation remain pending.
- Tensor views remain contiguous-only. Broadcasting is implemented for Generic
  FP32, INT8, and INT16 unequal-shape arithmetic; equal-shape operations retain
  existing backend behavior. NEON/AVX2 and FP16 broadcasting remain pending.
- Per-backend dtype/operation support is uneven. Generic FP16 is not
  supported for arithmetic; verify each backend operation rather than assuming
  every enum combination is implemented.
- AVX2 has historical x86_64 CI benchmark evidence, but the current Apple
  Silicon machine cannot compile or execute AVX2 instructions. Run the current
  AVX2 native regression suite on an AVX2-capable x86_64 runner after relevant
  backend changes.
- The F16C-specific AVX2 FP16 path is conditionally selected by feature
  detection and has a dedicated regression test. GitHub Actions run
  `42390563490` confirmed F16C support and logged execution of both the vector
  kernel and scalar fallback; see the M3.3.5 validation record above.
- The most recent Apple Silicon native allocation-safety run skipped AVX2
  cases. Its utility test passed; it does not establish AVX2 constructor
  runtime safety. The x86_64 run above validates AVX2 constructor/allocation
  tests for the M3.3.5 implementation revision.
- The FP32 performance objective remains at least 10% faster than the faster
  comparable NumPy/PyTorch CPU path on agreed workloads. This has not been
  established broadly. FP32 performance optimization is intentionally paused;
  do not resume it without a concrete request or new evidence.
- M2.3 AVX2 first-axis view runtime coverage remains pending. Successful x86
  CI run `42331993081` (commit `88dce7a21716af43f2c3d5adc81493c78fac4a0c`)
  executed AVX2 tests, but that committed revision does not contain the
  uncommitted `slice_first_axis()`/`select_first_axis()` API or regression
  tests. Its results therefore do not validate M2.3. After the M2.3 changes
  are available to CI, `tensor_nd_avx2` must run and pass before AVX2
  slice/select behavior is considered validated.
- GenericBackend's separate raw allocation helper is not covered by the shared
  aligned-allocator hardening. See the allocation-safety note above.

## Validation snapshot

M3.3.2 BOOL dtype foundation was validated on Apple Silicon:

- Native BOOL CTest: Generic and NEON passed; AVX2 skipped because the host is
  ARM64.
- Full native CTest: 19 passed, 7 AVX2-specific skips.
- Focused BOOL Python tests: 7 passed with Generic explicitly selected.
- Full Generic-selected Python suite using the rebuilt extension: 1003 passed,
  191 skipped, with 3 existing pytest return-value warnings.
- No sanitizer option/configuration was found in the repository; sanitizer
  validation was not run. M3.3.5 AVX2 runtime validation is recorded above.

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

M2.3 first-axis slice/select validation on Apple Silicon:

- Native full CTest: 15 passed, 6 AVX2-only tests skipped. Focused
  `tensor_nd_generic` and `tensor_nd_neon` passed, including offset aliases,
  lifetime, mutation, dtype, range-boundary, odd-sized tail, exact-alias, and
  partial-overlap rejection cases.
- Full Python suite using the rebuilt extension: 1007 passed, 28 skipped, and
  3 existing pytest return-value warnings. Focused `tests/test_tensor_nd.py`:
  53 passed and 1 unsupported-backend case skipped.
- Partial-overlap checks reject in-place writes and `add_into` when the output
  overlaps an input view; exact in-place aliases remain supported. The full
  native and Python suites passed after these guards were added.
- AVX2 is not available on this Apple M1 host. The x86 CI workflow has explicit
  AVX2 dispatch and full CTest/Python validation, but no new x86 run was
  observed for this milestone. AVX2 offset-view behavior remains unverified
  at runtime.
- The latest available x86 CI log was inspected: run `42331993081`, job
  `Native SIMD tests and benchmarks (x86_64)`, on commit
  `88dce7a21716af43f2c3d5adc81493c78fac4a0c`. It reported an AMD EPYC 9V45
  96-Core Processor, AVX2 and F16C CPU support, and GNU 13.3.0. With
  `LABNEURA_BACKEND=AVX2`, the native backend-selection check passed; the
  Python package reported `Selected backend: AVX2`; the full native CTest
  command reported 0 failures out of 21 tests; and the focused verbose
  `tensor_nd_avx2`, `avx2_allocation_safety`, `fp16_f16c_dispatch`, and
  `avx2_fp16_arithmetic` tests all passed (4/4). The F16C arithmetic log
  explicitly reported the F16C vector kernel running. These are verified
  AVX2/F16C results for that commit, not for the current uncommitted M2.3
  changes; no workflow run can validate those changes until they are available
  to the remote runner.

M3.1 broadcast shape/stride planner validation on Apple Silicon:

- Native full CTest: 15 passed, 6 AVX2-only tests skipped. Focused
  `tensor_nd_generic` and `tensor_nd_neon` passed.
- Focused Python planner/nD tests: 64 passed, 1 unsupported-backend case
  skipped. Full Python suite: 1018 passed, 28 skipped, and 3 existing
  pytest return-value warnings.
- Planner checks cover scalar, same- and different-rank, multi-axis,
  singleton, zero-sized, non-contiguous stride metadata, offset-view
  non-mutation, malformed ranks, incompatible dimensions, and output
  element-count overflow.
- The planner is backend-independent and allocates no Tensor storage.
  AVX2-specific runtime validation was not available locally; the x86 CI
  workflow must execute `tensor_nd_avx2` and the Python suite against this
  change before CI validation is complete. Arithmetic broadcasting and actual
  non-contiguous Tensor views remain unimplemented.

The subsequent M2.1 reshape/flatten milestone was also validated on Apple
Silicon:

- Native build completed; full CTest passed all runnable tests, with six
  AVX2-only tests skipped on this host.
- Focused Python nD/reshape tests passed (37 tests); the full Python suite
  completed successfully with the existing warnings and hardware-specific
  skips.
- Generic and NEON reshape/flatten native tests ran. AVX2 runtime behavior
  remains unverified locally because this host is Apple M1.
