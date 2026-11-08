# LabNeura Product and Engineering Roadmap

## Product definition
LabNeura is a high-performance C++ tensor/numerical-computing library with Python bindings. Long-term: broad tensor functionality across CPU and GPU. Near-term: excellent, measurable CPU performance.

Maintain two tracks together:
- **Functionality:** what users can express.
- **Execution:** how data is stored, scheduled, dispatched and accelerated.

Do not attempt full PyTorch parity in one project phase. Build a well-defined tensor core first.

## Architecture boundaries
1. **Public API:** Tensor construction, metadata, shape operations, math, reductions, linear algebra, ML functions and explicit device transfer.
2. **Tensor engine:** shape, strides, offset, dtype, layout, storage ownership, broadcasting, iteration planning and promotion rules.
3. **Runtime:** CPU pool/parallel_for, CPU allocator, backend capabilities and dispatch. Later, a separate GPU runtime.
4. **Kernels:** Generic reference, ARM NEON, x86 AVX2 (later AVX-512), and future Metal/CUDA/HIP.

CPU worker threads are not GPU scheduling. GPU queues/streams and events require separate semantics.

## Tensor foundation
Use one nD Tensor abstraction:
- 1D `[N]`, 2D `[M,N]`, 3D `[B,M,N]`, arbitrary nD.
- Metadata: shape, strides, offset, ndim, numel, dtype, layout/contiguity, storage owner; later device.
- Check shape-product and byte-size overflow.
- Define views, aliasing, mutation and lifetime explicitly.
- Begin with contiguous tensors; add non-contiguous views deliberately.
- Never assume `numel` equals storage allocation size for a view.

Initial numerical dtypes: FP32, FP16, INT16, INT8. Add a Boolean dtype for logical/comparison outputs. Consider FP64 only if requirements justify it. Define promotion, integer overflow/saturation, reduction accumulation and floating special-value behavior.

## Functionality modules

### 1. Creation and metadata
Constructors from C++ vectors/Python sequences; `empty`, `zeros`, `ones`, `full`; shape, ndim, numel, dtype, strides, contiguity.

### 2. Shape manipulation, views and indexing
`reshape`, `flatten`, `squeeze`, `unsqueeze`, `transpose`, `permute`, basic slicing/select, `contiguous`, clone/copy. Later: gather, scatter, advanced indexing, concatenate, stack, split. A view must have defined shared-storage and lifetime semantics; reject unsupported views clearly.

The initial basic-indexing slice/select surface is limited to contiguous first-
axis ranges and selections. Views carry an element-based storage offset and a
bounded logical size; backend data pointers address the view start. Other axes
and negative strides remain unsupported.

### 3. Broadcasting and elementwise operations
Build one shared trailing-dimension broadcast planner. Rules: equal dimensions or dimension 1 are compatible; missing leading dimensions behave as 1. Incompatible shapes error. In-place operations must reject expansion of the destination.
Operations: add/sub/mul/div; later pow/mod; neg/abs/square/sqrt/exp/log/rounding; eq/ne/gt/ge/lt/le; logical and/or/xor/not; where/clamp/min/max. Support tensor-tensor and scalar-tensor forms; allocating/in-place where valid.

### 4. Reductions
`sum`, `mean`, `min`, `max`, `prod`, `argmin`, `argmax`, `all`, `any`, `count_nonzero`, `norm`. Whole tensor, one/multiple axes, `keepdims`; define empty reduction and accumulation dtype.

### 5. Linear algebra
Priority: vector dot; matrix-vector; 2D matmul; batched matmul; outer product, trace and norms; later determinant, inverse and QR/decompositions. Matmul is its own kernel family, not an elementwise loop. Plan cache blocking, packing, SIMD microkernels and CPU tiling; GPU kernels will have separate execution strategies.

### 6. ML mathematics
ReLU, Leaky ReLU, sigmoid, tanh, GELU, softmax, log-softmax, MSE, cross-entropy and normalization. Use numerically stable formulations. Backward/autograd is a later dedicated milestone.

### 7. Random and initialization
Uniform, normal, integer random, explicit seed/reproducibility contract, Xavier and He initialization.

### 8. Later ML framework
Only after the tensor core is stable: autograd/graph, gradient accumulation, optimizers, neural-network modules, data loading and checkpointing. Treat this as a separate product expansion.

## Backend policy
Track support by operation × dtype × backend × allocating/in-place × layout/broadcast pattern.
For each feature: define semantics → Generic implementation → Generic tests → one optimized backend → explicit backend tests → benchmark → next backend.
Generic is the semantic reference. NEON/AVX2 must match it. Never claim support based only on automatic dispatch.

## Runtime and memory
Continue CPU work on feature detection, thread pool, parallel thresholds, aligned allocation, copy avoidance, memory bandwidth and cache locality.
Introduce device-aware storage only with the first actual GPU implementation. That vertical slice must define device identity, allocation/deallocation, host/device transfer, queue/stream, events, synchronization, pending-work lifetime and capability checks.
Do not build speculative generic GPU abstractions before a concrete backend.

## Testing policy
Every feature and every backend implementation gets tests in the same change:
- typical, empty/singleton, odd and boundary sizes
- invalid shapes/dtypes/arguments
- broadcasting success/failure
- numerical tolerance or exact integer equality
- NaN/Inf and integer overflow/saturation boundaries where relevant
- allocating/in-place, aliasing and views
- SIMD alignment and tails
- explicit backend parity against Generic
- Python API behavior
- concurrency/lifetime tests when relevant

CI: ARM64 for NEON; x86_64 with AVX2 for AVX2; Generic on both. GPU tests require actual supported hardware; compile-only is not runtime validation.

## Benchmark policy
Long-term target: at least 1.10x versus the faster comparable NumPy/PyTorch CPU implementation on agreed workloads. This is a measured target, not a blanket claim.
Record dtype, operation, shape, allocating/in-place, layout, cache regime, backend, threads, native-vs-Python timing, median/IQR/p95, correctness and environment. For GPU, report both end-to-end including transfers and device-resident time. Never compare GPU kernel-only against CPU end-to-end.

## Milestones
| Milestone | Deliverable | Exit criteria |
|---|---|---|
| M0 | Stabilize runtime and benchmark foundation | Reproducible baseline, tests green |
| M1 | nD metadata/storage invariants | Rank/shape/stride/overflow tests |
| M2 | Creation, views, shape transforms, basic indexing | Explicit alias/copy semantics |
| M2.3 | First-axis slice/select views | Offset-aware contiguous views, bounds/lifetime and backend tests |
| M3.1 | Broadcasting shape/stride planner | Pure nD output-shape and effective-stride plan; no arithmetic |
| M3.2.1 | Elementwise arithmetic semantics | Contract and semantic test matrix documented; no kernels |
| M3.2.2 | Generic FP32 arithmetic with broadcasting | ADD/SUB/MUL/DIV; allocating and supported in-place operations; NumPy parity |
| M3.2.3 | Generic INT8 arithmetic with broadcasting | Saturating ADD/SUB/MUL, safe integer DIV, allocating/in-place and NumPy parity |
| M3.2.4 | Generic INT16 arithmetic with broadcasting | Widened saturating ADD/SUB/MUL, safe integer DIV, allocating/in-place and NumPy parity |
| M3.2.5+ | Remaining dtype/backend broadcast kernels | Generic FP16 decision, then NEON/AVX2 with explicit backend parity |
| M3.3.2 | BOOL dtype foundation | Canonical one-byte storage, construction/access/copy/lifetime across Generic, NEON, and AVX2 |
| M3.3.3 | Generic tensor comparisons | EQ/NE/LT/LE/GT/GE with BOOL output, shared broadcasting, explicit non-Generic rejection |
| M3.3.4 | NEON tensor comparisons | Native NEON EQ/NE/LT/LE/GT/GE for FP32/FP16/INT16/INT8, explicit Generic parity |
| M3.3.5 | AVX2 tensor comparisons | Complete: AVX2 kernels and parity tests passed x86_64 runtime validation |
| M3.4.1 | Generic BOOL logical NOT | Complete: elementwise `logical_not` for BOOL tensors |
| M3.4.2 | NEON BOOL logical NOT | Complete: native NEON vector kernel; explicit backend tests and ARM64 runtime validation passed |
| M3.4.3 | AVX2 BOOL logical NOT | Complete: AVX2 kernel and explicit tests passed x86_64 runtime validation in CI run `42449211301` |
| M3.4.4 | BOOL logical AND | Generic, NEON, and AVX2 kernels, broadcasting, dispatch, and explicit tests; AVX2 x86_64 runtime validation pending |
| M3 (later) | Logical operations | BOOL logical OR and `where()` remain future work |
| M4 | Reductions | Axis/keepdims/dtype/edge tests |
| M5 | Dot and matmul | 2D and batched semantics, dedicated kernels |
| M6 | ML math | Numerical stability and reference tests |
| M7 | CPU optimization across dtypes/ops | Per-operation benchmark reports |
| M8 | First GPU vertical slice (Metal or CUDA) | Allocate → execute → transfer → test |
| M9 | More GPU ops/dtypes and vendor backend | Capability matrix and hardware CI |
| M10 | Autograd/training utilities if desired | Separate design and milestones |

Backend optimization is continuous during M1–M7: Generic first, then optimized CPU backends with immediate tests.

M3.3.5 was validated in GitHub Actions run `42390563490` on implementation
commit `895b7633e1cd6f48b1459397cd664ee1d4dd2c08`: AVX2 native and Python
comparison tests passed; the full native suite had zero failures, and the
full AVX2-selected Python suite reported 1156 passed, 215 skipped, and 3
warnings. AVX2 arithmetic and FP16/F16C vector/scalar regression tests passed.
This is functional validation; no performance improvement is claimed.

M2.3 first-axis slice/select implementation and Generic/NEON local tests are
present in the current worktree. AVX2 runtime validation remains pending on an
AVX2-capable x86_64 runner.

M3.1 is complete: the shared metadata-only nD broadcast planner returns the
output shape and both operands' effective element strides. Arithmetic
broadcasting remains pending.

M3.2.1 defined the same-dtype, broadcast, overflow, division, rounding,
aliasing, and error contract recorded in `references/arithmetic-semantics.md`.
M3.2.2 implements Generic FP32 ADD/SUB/MUL/DIV broadcasting, M3.2.3 adds
Generic INT8 broadcasting with saturating arithmetic, and M3.2.4 adds Generic
INT16 broadcasting and equal-shape arithmetic. Generic FP16 and optimized
backend broadcast kernels remain pending and should be implemented in bounded,
explicitly tested stages.

## Good agent task boundaries
Good: "Define and test nD shape/numel overflow invariants; no operations."
Good: "Implement contiguous 2D transpose metadata and tests; no non-contiguous kernels."
Good: "Implement broadcast shape inference as a pure helper; do not change arithmetic kernels."
Good: "Add Generic FP32 ADD broadcasting and tests."
Good: "Add NEON FP32 ADD broadcasting using the shared iterator and explicit NEON tests."
Good: "Implement Generic 2D FP32 matmul and tests; do not optimize yet."
Bad: "Implement all tensor features and all backends."
