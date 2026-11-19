# EDA Verilog lexer benchmark

The benchmark runs the scalar and, when compiled and supported by the current
CPU, NEON and AVX2 Verilog/SystemVerilog byte classifiers and lexer over the
same representative source files. It reports
classification-only and end-to-end results separately, including input bytes,
corpus token count, median time, IQR, p95, MB/s, lexer tokens/s, selected
backend, target architecture, CPU model when available, OS, compiler, build
type, and hardware thread count. The retained-capacity metric is an estimate of
output-vector storage, not a process-wide peak-memory or allocation count.

Configure and build outside the source tree:

```sh
cmake -S . -B "$TMPDIR/labneura-lexer-build" \
  -DLABNEURA_BUILD_TENSOR=OFF \
  -DLABNEURA_BUILD_COMPAT=OFF \
  -DBUILD_TESTING=ON
cmake --build "$TMPDIR/labneura-lexer-build" --target labneura_lexer_benchmark
"$TMPDIR/labneura-lexer-build/labneura_lexer_benchmark" \
  examples/verilog/ibex/ibex_alu.sv \
  examples/verilog/8080_core/i8080.v
```

The benchmark performs two warmups and nine measured passes per backend and
case. It reports both the aggregate corpus and each input file separately.
NEON results are reported as unavailable on builds without compiled ARM64
support. Results describe the current host and compiler configuration; they are
not performance claims against another frontend.

The experimental C++ selector reads `LABNEURA_VERILOG_LEXER` (`slang` by
default, or `scalar`; other values are rejected). Existing `parse_verilog`
calls remain Slang-backed regardless of this setting; the selector is
available to experimental lexer clients without changing production parsing.

The independent `LABNEURA_VERILOG_LEXER_BACKEND` setting accepts `auto`,
`scalar`, `neon`, or `avx2` and does not affect frontend selection. `auto`
selects runtime-supported AVX2, then NEON, then scalar. Requesting an
unavailable backend throws an error. C++ callers can also choose a backend in
`VerilogLexerOptions`.

## Native preprocessor benchmark

The separate `labneura_preprocessor_benchmark` target measures native
preprocessing over the supplied source files. It reports input/output bytes,
median and IQR over nine samples after two warmups, macro/directive counts,
loaded source count (including includes), throughput, and hardware thread
count. It does not compare against Slang:

```sh
cmake --build "$TMPDIR/labneura-lexer-build" \
  --target labneura_preprocessor_benchmark
"$TMPDIR/labneura-lexer-build/labneura_preprocessor_benchmark" \
  examples/verilog/ibex/ibex_alu.sv \
  examples/verilog/8080_core/i8080.v
```

The independent C++ API is in `include/labneura/eda/verilog_preprocessor.h`.
It supports object-like and function-like macros, nested expansion and
arguments, conditional compilation, nested relative/search-path includes,
token concatenation, macro stringification, LF/CRLF macro-definition line
continuations, timescale and default-nettype metadata, and source-map lookup
through `VerilogPreprocessorResult::origin_at`. Token paste validates the
concatenated spelling as one lexer token; stringification substitutes
parameters and escapes quote and backslash characters. `begin_keywords`,
`line`, `resetall`, cell-definition, and drive directives are retained as
deferred metadata rather than applied as semantic state. Undefined macro uses
are errors and remain in the output.

Preprocessing is an independent utility; it is not routed through
`parse_verilog`, whose production Slang path remains unchanged.

## Native syntax parser foundation

The experimental C++ syntax API is in
`include/labneura/eda/verilog_syntax.h`. `parse_verilog_syntax(preprocessed)`
parses an existing preprocessor result, while the string and file overloads
run native preprocessing, lexing, and parsing in sequence. The owned syntax
tree covers modules, interfaces, packages, parameters, ports, simple net and
variable declarations, continuous assignments, instance connections, a subset
of expressions, and behavioral statement trees. Procedural support includes
`always`, `always_comb`, `always_ff`, `always_latch`, and `initial`; nested
`begin`/`end`, `if`/`else`, `case`/`casez`/`casex`, blocking and nonblocking
assignments, `for`/`while`/`repeat`/`forever` loops, event controls, delays,
and `wait`. Event terms support `posedge`, `negedge`, level events, comma, and
`or` separators. Loop AST nodes own initializer and update assignments,
conditions/counts, declared loop-variable metadata, and nested bodies.
Statement kinds distinguish event controls from delay controls. AST nodes
retain source-map and macro-expansion origins.

This is not a complete SystemVerilog parser. Fork/join, assertions,
callable syntax outside the subset described below, user-defined types outside
the type subset below, and many SystemVerilog declarations remain unsupported
and produce diagnostics rather than being accepted as parsed statements. The
bounded typedef/enum/struct/union/array subset is described in the native
type-system section below. Procedural `assign`/`deassign`/
`force`/`release` are explicitly unsupported; intra-assignment timing and
comma-separated loop initializer declarations are also outside the current
grammar.
`initial` and delay controls are represented syntactically; no simulation or
synthesis timing semantics are provided. Generate constructs are supported
separately by the native elaborator described below. This API does not replace
the Slang-backed production `parse_verilog` path.

## Native semantic-analysis foundation

`include/labneura/eda/verilog_semantic.h` exposes
`analyze_verilog_semantics(const VerilogCompilationUnit&)`. It builds an owned
compilation-unit/design-unit scope tree, collects parameters, localparams,
ports, nets, variables, procedural blocks, and instances before resolving
references in supported expressions and statements. Resolved references point
to semantic symbols; declarations and diagnostics retain syntax source-map
locations. Hierarchical lookup is available through
`lookup_verilog_symbol`.

The analyzer diagnoses duplicate declarations and instance names, unresolved
identifiers, unknown module types, and invalid named parameter and port
connections when the target module is in the compilation unit. It also
collects callable symbols and scopes, resolves supported function/task bodies
and calls, and diagnoses callable kind and positional argument-count errors.
Package-qualified callable references resolve when the package is present.
This is not a full SystemVerilog semantic implementation: imports, full type
checking, elaboration, and general cross-design-unit visibility are not
modeled. Production parsing remains Slang-backed.

## Native constant evaluation and elaboration

`include/labneura/eda/verilog_constant.h` exposes
`evaluate_verilog_constant()` for the expression subset represented by the
native syntax AST. Values own their bits (least-significant first), width,
signedness, four-state values, source location, and unbased-unsized fill state.
The evaluator supports integer literals, parameter identifiers, unary and
binary arithmetic/bitwise/logical/comparison/shift operators, conditional
expressions, concatenations, replications, constant selects, and the
`$clog2`, `$signed`, and `$unsigned` system functions. It reports structured
errors for unsupported expressions and invalid constant operations. Its
default maximum value width is 65,536 bits.

`include/labneura/eda/verilog_elaboration.h` exposes
`elaborate_verilog_design()`. It evaluates module parameters and localparams,
resolves forward dependencies, applies named or positional instance
overrides in per-instance environments, and returns an owned module hierarchy
with parent/child links. Package-qualified parameter references are supported.
Callers can select a top module; otherwise a unique uninstantiated module is
selected, and ambiguous candidates are diagnosed. Missing definitions,
parameter cycles, invalid overrides, and recursive module instantiation are
reported as structured diagnostics.

This is a bounded native foundation, not complete SystemVerilog elaboration.
At this stage, generate constructs are handled by the M7.5 extension below;
`defparam`, general type parameters, user-defined type evaluation, arbitrary
constant functions, complete context-determined sizing, full four-state
arithmetic, and complete port/type checking are not supported. Slang
value/width differential checks have not been performed. Both APIs are
independently callable; production `parse_verilog` remains Slang-backed.

## Native generate construct elaboration (M7.5)

The native syntax AST retains explicit `generate` regions, `if`/`else if`/
`else` generate conditionals, `case`/`casez`/`casex` branches, and `for`
generate loops. These constructs reuse existing expression, declaration,
instance, parameter, and semantic-scope representations. The elaborator
selects only the active conditional/case branch, evaluates loop bounds and
steps using the native constant evaluator, and creates independent generated
scopes for each selected branch and loop iteration. Generated declarations,
localparams, assignments, and module instances are attached to their scope;
generated child modules receive the scope's parameter environment. Paths
include named blocks and loop indices (for example `top.lanes[1].lane`), and
every module instance is elaborated independently. Generate syntax and
diagnostics preserve preprocessor source-map and macro-expansion locations.

The default generate iteration limit is 1024 and can be changed through
`VerilogElaborationOptions::maximum_generate_iterations`. Genvars are modeled
as signed 32-bit values; for-generate bodies must be named `begin` blocks.
Case wildcard matching covers the supported exact, `casez`, and `casex`
four-state bit patterns. Nonconstant conditions, invalid/no-progress steps,
duplicate selected block names, iteration-limit exhaustion, and recursive
module hierarchies produce structured diagnostics.

This remains a subset: generate-if/case/for declarations and instances are
supported, but generate subroutines, `defparam`, type parameters, complete
four-state/case semantics, and the rest of the SystemVerilog generate grammar
are not. Slang is used as a differential reference for a basic three-iteration
generated-instance count only; full AST or hierarchy equivalence is not
claimed. Production `parse_verilog` remains Slang-backed.

## Native procedural statement AST and parser (M7.6)

The procedural AST is owned by each `VerilogProceduralBlock` and uses typed
statement kinds for sequential blocks, conditionals, case statements, loops,
procedural assignments, event controls, delay controls, wait, null statements,
and unsupported recovery placeholders. It retains statement/loop/case
locations, optional block labels, expressions, assignment kind, event terms,
and nested statements. Semantic analysis walks the represented expressions
and assignment targets, creates a local semantic scope for a declared
procedural loop variable, resolves visible references, and diagnoses
non-assignable lvalues without attempting full type checking.

`for` supports an optional initializer, condition, and update; supported
updates include `=`, arithmetic/bitwise compound updates, and postfix `++` or
`--`. A single simple declared loop variable (for example `int i = 0`) is
represented with its own semantic scope. `while`, `repeat`, and `forever` are
parsed with nested statement bodies. Event and delay controls are explicit
statement forms. Procedural continuous assignment forms are diagnosed and
preserved as unsupported nodes rather than silently accepted.

This is syntax and reference analysis only: loops and delays are not executed,
`initial` is not simulated or synthesized, and event scheduling, timing
semantics, procedural continuous assignment, intra-assignment delays,
comma-separated loop declarations, `do` loops, fork/join, assertions, tasks,
functions, and full SystemVerilog statement semantics remain unsupported.
Focused Slang comparison confirms syntax acceptance for the supported
procedural-loop fixture; AST equivalence is not claimed. Production
`parse_verilog` remains Slang-backed.

## Native expression type analysis (M7.7)

`include/labneura/eda/verilog_type.h` exposes
`analyze_verilog_expression_types()` for a syntax tree and its semantic model,
`infer_verilog_expression_type()` for individual expressions, and
`infer_verilog_declared_type()` for declared types. The owned type record is
separate from constant values and represents integral/real/string/unknown
categories, optional bit width, signedness, two-state/four-state behavior,
unbased-unsized status, and resolved packed dimensions. Symbol-type map keys
point into the supplied semantic model, which must remain alive while those
keys are used. Elaboration also retains parameter types alongside
instance-specific parameter values.

The supported sizing subset treats unsized decimal integers as signed, at
least 32-bit values; sized based literals use their declared width, while
unsized based literals use at least 32 bits and the current constant evaluator's
digit-based width. Built-in `bit`, `logic`/net, integer-atom, `integer`, and
`time` declarations use their built-in width and state model. Constant packed
dimensions multiply the base integral width. Binary arithmetic and bitwise
expressions use the larger operand width and are signed only when both inputs
are signed; shifts use the left operand type. Comparisons, logical operators,
and reductions produce one-bit unsigned results. Conditional expressions use
the larger branch width, concatenations sum operand widths, replications
multiply their constant count by the concatenated operand width, bit-selects
are one bit, and constant part-selects use the absolute range width.

Known assignment width and signedness conversions produce source-located
warnings; they are not treated as illegal SystemVerilog. Unknown widths remain
unknown rather than being guessed. Instance parameter values retain their
resolved declared type and receive supported truncation/extension. The native
result is compared with Slang for a supported binary arithmetic expression
and a signed enum's width, signedness, and state model; no general
type-equivalence claim is made. The production `parse_verilog` path remains
Slang-backed.

## Native SystemVerilog type-system expansion (M7.8)

The syntax AST represents typedef aliases, enum declarations and enumerators,
struct/union members, packed dimensions, unpacked dimensions, and source
locations. The type analyzer resolves aliases within their owning
module/interface/package/generate scope, supports forward typedef references
within that scope, and resolves package-qualified types without leaking
module-local names into sibling scopes. Type records retain alias and
declaration identity separately from their underlying representation.

Enums support explicit or implicit constant values and an integral base type
(defaulting to the parser's supported integer base). Their enum identity is
preserved alongside the underlying width, signedness, and state model.
Enumerator constants can be referenced through a package qualifier and are
available to the native constant/elaboration path. Structs retain typed
members; packed structs compute total width and member offsets from the
least-significant end. Unpacked structs retain member types but have no
invented packed width. Packed unions use shared offset zero and require equal,
known integral member widths. Unpacked unions retain member types without
claiming a packed layout. Unknown member types leave layout unknown and
produce diagnostics.

Fixed unpacked arrays are represented as nested array types with per-dimension
range/count metadata and a distinct element type; repeated indexing returns
the next nested element type. Packed dimensions contribute to integral/packed
width independently of unpacked element counts. Package typedefs are usable
for module declarations and typed parameters, and the elaborator preserves
enum/typedef parameter types and per-instance values.

This remains a bounded subset, not complete SystemVerilog type checking.
Dynamic/associative/queue arrays, type parameters, imports, packed-array and
aggregate assignment compatibility beyond known widths, arbitrary member
selection semantics, full enum range/overflow checking, and complete
context-determined sizing are unsupported. Unpacked-array part-selects are
not inferred. Unknown custom types stay unknown and are diagnosed; unsupported
types are never silently coerced to integers. Native tests cover scoped and
forward typedefs, signed/unsigned enums and values, packed/unpacked
structs/unions, nested and multidimensional arrays/indexing, invalid and
unknown members, duplicate declarations, package separation, generate-scope
types, parameter type preservation, and an enum width/signedness/state
differential against Slang. This does not claim semantic equivalence with
Slang or change production parsing.

## Native function and task declarations (M7.9)

The owned syntax AST represents function and task declarations in modules,
interfaces, and packages. The supported ANSI-style subset retains callable
kind, `automatic`/`static` lifetime, function return type and packed ranges,
ordered typed/directed formals, leading local declarations, nested statement
bodies, and source-map locations. An omitted function return type defaults to
one-bit `logic`. Task invocations in procedural bodies use a distinct
`TaskCall` statement carrying the ordinary call-expression node; functions
remain expression calls.

Semantic analysis creates Function/Task symbols and callable child scopes.
Formal argument symbols retain type, dimensions, direction, and source
location in declaration order. Callable bodies resolve against formal and
local symbols. Known calls are checked for function-versus-task context and
positional argument count; output/inout/ref actuals must be assignable.
Package-qualified callable references are resolved when the package is
present. Type analysis resolves function return types using the existing type
environment, assigns function-call expressions that result type, and checks
known integral actual/formal width and signedness conversions with
source-located warnings.

This is not full SystemVerilog subroutine support. Old-style non-ANSI formal
declarations, named actual argument syntax, nested callable declarations,
full direction/type compatibility, task execution, function evaluation,
automatic lifetime behavior, recursion checks, and timing/simulation
semantics are unsupported. Slang is used to confirm acceptance of a
representative syntax fixture, not semantic equivalence. The production
`parse_verilog` path remains Slang-backed and unchanged.

Validation on Apple M1/arm64 with Apple Clang 21.0.0: the fresh full native
build succeeded and CTest ran 37 tests (28 passed, 9 AVX2-specific tests
skipped, no failures). The fresh EDA-only build succeeded and CTest ran 8
tests (7 passed, 1 AVX2 lexer test skipped, no failures). The four EDA Python
regression suites ran 216 tests successfully. The callable-specific test
covers parsing, semantic resolution and diagnostics, type inference,
macro/source locations, package-qualified calls, and Slang syntax acceptance.
