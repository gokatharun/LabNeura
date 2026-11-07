# M3.2 Elementwise Arithmetic Semantics

Status: **M3.2.1 semantics defined; M3.2.2 Generic FP32, M3.2.3 Generic
INT8, and M3.2.4 Generic INT16 broadcasting implemented.** The broader
dtype/backend contract below remains a target for later milestones; current
coverage is called out separately.

## API

Keep the existing C++ methods `add`, `sub`, `mul`, `div`, and their
`*_inplace` counterparts. Python exposes the same methods and the existing
`+`, `-`, `*`, `/`, `+=`, `-=`, `*=`, `/=` operators. Each binary operand is a
Tensor; scalar arithmetic overloads are deferred. A scalar Tensor has shape
`[]` and can participate in arithmetic.

Allocating operations return a new Tensor with independent storage, the
broadcast output shape, the left operand's dtype, and the left operand's
backend. The result is contiguous row-major storage. In-place operations mutate
the left operand.

## Shape and mutation rules

- Use M3.1 `BroadcastPlan`: right-align dimensions; equal dimensions or a
  dimension of one are compatible; absent leading dimensions act as one.
- The result dimension is the compatible maximum. `[]` broadcasts as a scalar.
  `[0,3]` with `[1,3]` produces `[0,3]`; incompatible dimensions throw.
- Broadcasting is metadata-only: do not materialize expanded operands.
- In-place operations are allowed only when the broadcast result shape equals
  the destination's shape. The right operand may broadcast to the destination;
  the destination must not be expanded. Thus a tensor plus a scalar Tensor may
  update the tensor, but a scalar destination plus a non-scalar Tensor is
  rejected.
- Exact elementwise aliases are allowed. Reject partial/overlapping aliases
  whenever writes could overwrite values still needed by the broadcast read
  pattern. Allocating results do not alias either input.
- Zero-element broadcast results are valid and perform no element accesses.

## Dtypes and promotion

Only same-dtype Tensor operands are supported in this milestone; there is no
implicit promotion. The output dtype is the common input dtype:

| Input dtype | Output dtype | Intended operations | Current backend availability |
|---|---|---|---|
| FP32 + FP32 | FP32 | ADD, SUB, MUL, DIV | Generic, NEON, AVX2 |
| FP16 + FP16 | FP16 | ADD, SUB, MUL, DIV | NEON, AVX2; not Generic |
| INT16 + INT16 | INT16 | ADD, SUB, MUL, DIV | NEON, AVX2; not Generic |
| INT8 + INT8 | INT8 | ADD, SUB, MUL, DIV | Generic, NEON, AVX2 |

FP64 and INT32 remain unsupported. The table describes dtype semantics and
current backend operation coverage, not runtime validation of every backend.
An unsupported dtype/backend combination must raise an explicit error.

Mixed-dtype arithmetic and Python scalar overloads are deferred. Promotion
would require an explicit policy for integer/float combinations, FP16
accumulation, and quantized integer values; implicit conversion now would risk
silently changing existing results. The integer dtypes are currently plain
signed stored values, not affine quantized tensors with per-tensor scale and
zero-point metadata. Existing quantize/dequantize helper functions do not
change arithmetic semantics.

## Operation semantics

| Dtype | ADD / SUB / MUL | DIV |
|---|---|---|
| FP32 | IEEE-754 binary32 operation | IEEE-754 binary32 division |
| FP16 | IEEE-754 binary16 result; round once to binary16 using round-to-nearest, ties-to-even | IEEE-754 binary16 result; same rounding rule |
| INT8 | Signed integer operation, saturate to `[-128, 127]` | Truncate quotient toward zero; divisor zero produces `0`; saturate the `-128 / -1` result to `127` |
| INT16 | Signed integer operation, saturate to `[-32768, 32767]` | Truncate quotient toward zero; divisor zero produces `0`; saturate the `-32768 / -1` result to `32767` |

Integer ADD/SUB/MUL must use sufficiently wide intermediates and clamp before
conversion; signed overflow must never occur in C++. Integer division must
check for a zero divisor before division and widen before dividing so the
minimum-signed-value / `-1` case is defined and saturates.

FP32 uses IEEE-754 behavior under the default round-to-nearest, ties-to-even
environment. Division by positive or negative zero yields signed infinity for
nonzero numerators, and NaN for zero divided by zero. NaNs and infinities
propagate according to the corresponding IEEE operation; signed zero is
preserved according to IEEE rules. No custom NaN payload guarantee is made.

FP16 follows IEEE binary16 special-value behavior, including infinities and
NaNs from division by zero, and binary16 overflow to infinity. Results,
including subnormal results, are rounded once to binary16 with ties-to-even.
No NaN payload preservation guarantee is made.

## Errors and compatibility

The public methods and operators share one validation path in C++ and Python.
The intended errors are:

- Incompatible shapes, mixed dtypes, and in-place destination expansion:
  `std::runtime_error` in C++, translated to Python `RuntimeError`.
- Unsupported dtype/backend operations and arithmetic execution failures:
  `std::runtime_error` in C++, translated to Python `RuntimeError`.

Today, arithmetic requires identical shapes and dtypes; shape mismatch
currently raises a runtime error in some paths. Keep mismatched-shape and
mismatched-dtype failures as `std::runtime_error` / Python `RuntimeError` for
compatibility. Incompatible broadcast shapes and in-place destination
expansion use the same error category. Accepting broadcast-compatible shape
pairs and returning a broadcast-shaped result is an intentional behavior
change; update tests that currently require equal shapes. Existing same-shape
operations, integer saturation, integer division-by-zero result `0`, and FP
floating-point behavior are intended to remain compatible.

M3.2.2/M3.2.3/M3.2.4 implement unequal-shape broadcasting for FP32, INT8,
and INT16 tensors whose backend is Generic. Equal-shape operations continue
to use their existing backend paths. NEON and AVX2 are not silently replaced
by Generic for broadcasting: their unequal-shape behavior remains the
existing size/shape error until those backends gain broadcast kernels. INT8
and INT16 use widened signed integer intermediates and saturate to their
documented ranges; they do not convert values through FP32. Generic INT16
also now supports equal-shape arithmetic through the existing Tensor methods;
this does not change optimized-backend behavior.

## Semantic test matrix

Each case applies to all four operations unless the cell says otherwise.
Compare integer results exactly. Compare FP32 against an IEEE-compatible
reference with exact special-value/sign checks; compare FP16 at binary16 bit
level, treating NaN payloads as unspecified.

| Area | Required cases |
|---|---|
| Dtype × operation | For every supported backend combination: FP32, FP16, INT16, INT8 × ADD/SUB/MUL/DIV; allocating and in-place; output dtype preserved; unsupported combinations must fail explicitly |
| Broadcast shape | Same shape; `[2,3]` with `[3]`; `[4,1,3]` with `[1,5,3]`; scalar `[]`; singleton dimensions; `[0,3]` with `[1,3]`; incompatible `[2,3]` with `[4,3]` and `[0]` with `[2]` |
| Effective strides | Missing leading and expanded singleton axes have zero effective stride; offset and non-contiguous source metadata preserved; source shape/strides/storage unchanged; no expanded allocation |
| In-place constraints | RHS scalar/singleton expansion to destination; destination-expanding case rejected; result shape remains destination shape; zero-sized destination |
| INT8/INT16 boundaries | ADD/SUB/MUL positive and negative saturation; exact extrema; non-overflow values; positive and negative quotient truncation; divisor zero; minimum divided by `-1`; vector and scalar-tail boundaries |
| FP32 boundaries | `+0`/`-0`; finite divided by signed zero; `0/0`; finite/infinite combinations; NaN in either operand; subnormals; overflow to infinity; representative rounding cases |
| FP16 boundaries | Round-to-nearest ties-to-even; smallest/largest subnormals and normal values; underflow; overflow; signed zero; NaN/Inf combinations; division by zero; binary16 result bits |
| Dtype and API errors | Every mixed-dtype pair rejected; FP64/INT32 rejected; unsupported backend/dtype rejected; Python methods/operators and C++ methods report equivalent failures |
| Aliasing / storage | Independent allocating output; exact in-place alias; partial overlap rejection; views and nonzero storage offsets; shared-storage mutation remains correct |
| Backend parity | Generic reference where supported; explicit NEON and AVX2 tests for each claimed dtype/operation, including SIMD boundaries and tails; AVX2 only on capable x86 hardware |

## M3.2 implementation sequence

1. **M3.2.2 — Generic FP32 broadcasting:** shared broadcast iteration, allocating
   and in-place ADD/SUB/MUL/DIV, shape and alias checks, native/Python tests.
2. **M3.2.3 — Generic INT8 broadcasting:** reuse the typed iterator; widened
   intermediates, saturation, division edge cases, and parity tests.
3. **M3.2.4 — Generic INT16 broadcasting:** reuse the typed iterator and add
   Generic INT16 storage access/equal-shape kernels with widened intermediates,
   saturation, division edge cases, and parity tests.
4. **M3.2.5 — Generic FP16 decision and implementation:** establish executable
   Generic support or explicitly mark Generic FP16 unsupported; do not
   silently fall back.
5. **M3.2.6 — NEON broadcasting:** add supported dtype/operation combinations
   with explicit backend parity and tail tests.
6. **M3.2.7 — AVX2 broadcasting:** same contract, explicit boundary/tail
   tests, and execution on x86 AVX2 CI. Validate F16C dispatch separately.
7. **M3.2.8 — API and regression closure:** C++/Python error parity, operator
   coverage, mixed-dtype rejection, views/offset/overlap, full suites and
   documentation.

Generic FP16 arithmetic remains unsupported. Do not silently use a different
dtype or backend fallback.

The strict floating-point clauses above are intended semantics, not assertions
that every current path already meets them. In particular, the ARM32 NEON FP32
division path uses a reciprocal estimate and Newton-Raphson refinement rather
than a correctly rounded divide, and the AVX2 FP16 path evaluates through
FP32 before converting to FP16. Validate difficult rounding boundaries and
either correct these paths or explicitly narrow the guarantee before declaring
backend parity.
