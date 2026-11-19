# LabNeura Current Project State

Last verified: 2026-11-19. Treat this as a continuity reference, not a
substitute for inspecting the current source, tests, worktree, and benchmark
artifacts.

## Project identity

LabNeura is a C++ numerical-computing and Tensor library with Python bindings.
Its long-term direction is broad nD Tensor functionality, high-performance CPU
execution, and eventual GPU execution. CPU functionality and performance remain
the immediate priorities; GPU implementation is not active work.

### EDA scalar Verilog/SystemVerilog lexer (M5)

The public experimental C++ API is in `include/labneura/eda/verilog_lexer.h`;
its implementation is `src/labneura/eda/verilog_lexer.cpp`. Tokens and
diagnostics carry a source ID, byte offset/length, and one-based byte
line/column. Token spellings are views into immutable source buffers. The
scalar lexer handles versioned Verilog/SystemVerilog keywords, identifiers,
numeric forms, strings, comments, compiler directives, longest-match
operators/punctuation, EOF, and malformed-input diagnostics. It has no Tensor
or acceleration-core dependency and does not preprocess, parse, or elaborate.

The `LABNEURA_VERILOG_LEXER` selector defaults to `slang` and accepts `scalar`;
unknown values throw. The existing `parse_verilog` and Slang parser flow do not
consult this selector and remain unchanged. The selector is available to
experimental lexer clients; it does not switch the production parser.

The native lexer test uses Slang's supported public `slang::parsing::Lexer`
API for spelling, source-offset, line/column, and keyword-classification
differential checks on `tests/data/verilog/lexer_corpus.sv`. Slang groups some
based literals differently, so the differential check combines contiguous
Slang tokens where needed. A separate test checks every keyword in Slang's
public language-version tables against scalar classification for all supported
versions. Native lexer tests also exercise every category, malformed inputs,
longest-match operators, every prefix boundary of a sample,
repeated/independent buffers, and a 2 MiB input.
Existing Slang parsing behavior remains the production default.

Apple Silicon validation: the lexer CTest and existing EDA Python tests passed;
full native CTest had 22 passes, 8 expected AVX2 skips, and 0 failures.
Benchmarking two existing source files (69,727 bytes, 13,256 tokens) reported
6.534 ms median, 1.404 ms IQR, 16.214 ms p95, 10.67 MB/s, and 2.03 M tokens/s
with Apple Clang 21.0.0, Apple M1, arm64, and 8 reported hardware threads. The
retained token/error vector capacity estimate was 819,200 bytes. This is a local
scalar baseline, not a SIMD or cross-frontend performance claim.

### Verilog byte classification and NEON (M6.1/M6.2)

The EDA-internal classifier is in `src/labneura/eda/verilog_byte_classifier.*`.
It returns fixed 32-byte blocks with two compact per-byte flag arrays and
block-level any/all summaries. The scalar ASCII classifier is always compiled;
ARM64 builds enable the NEON implementation only after a compiler intrinsic
check. NEON classifies full blocks using two unaligned-safe `vld1q_u8` loads and
uses the scalar reference for shorter tails. The classifier remains EDA-owned
and does not link Tensor or Acceleration Core.

`VerilogLexerOptions::backend` and the independent
`LABNEURA_VERILOG_LEXER_BACKEND=auto|scalar|neon` setting select the byte
classifier. Auto selects NEON when compiled, otherwise scalar. An unsupported
explicit NEON request and invalid environment values throw; frontend selection
through `LABNEURA_VERILOG_LEXER` and Slang-backed `parse_verilog` remain
unchanged. NEON block scanning is integrated into whitespace, identifier,
escaped-identifier, line/block-comment, and string candidate searches.

Validation on Apple M1/arm64: the native lexer test exercises all 256 byte
values, every partial-block size, unaligned addresses, embedded NUL/high-bit
bytes, large inputs, scalar/NEON full-lexer token and diagnostic parity, and
the existing Slang corpus/keyword differential. Full native CTest passed 22
tests with 8 expected AVX2 skips; the EDA Python regression selection passed.

Three benchmark invocations, each with two warmups and nine samples, used the
same two designs as M5 (69,727 bytes, 13,256 tokens; Release, Apple Clang
21.0.0). Median-of-invocation medians: classifier-only scalar 0.294 ms
(237.4 MB/s), NEON 0.061 ms (1,150.1 MB/s), about 4.85x; end-to-end scalar
5.757 ms (12.11 MB/s, 2.30 M tokens/s), NEON 5.074 ms (13.74 MB/s, 2.61 M
tokens/s), about 1.13x. These are local baseline measurements, not universal
performance claims. The benchmark remains documented in
`benchmarks/eda/README.md`.

### Verilog AVX2 byte classification (M6.3)

The x86 AVX2 classifier is isolated in
`src/labneura/eda/verilog_byte_classifier_avx2.cpp`; CMake compiles that one
translation unit with `-mavx2` (or `/arch:AVX2`) only after a compiler/target
check. The generic EDA sources retain baseline compiler flags. The classifier
uses 32-byte vectors, compact movemasks, and scalar lane-mask packing into the
existing fixed block representation. Unaligned loads are expressed through
`memcpy`, which optimizes to a vector load in the AVX2 translation unit without
forming an out-of-bounds or alignment-sensitive pointer.

Runtime support is checked independently of compile support. `auto` selects
AVX2 only when both the AVX2 translation unit is built and OS/CPU runtime
capabilities report AVX2; otherwise it falls back to NEON or scalar. Explicit
unsupported requests throw. The existing scalar and NEON paths remain
available, and NEON handles the new 32-byte classifier block as two 16-byte
vectors.

AVX2-specific byte and full-lexer parity tests are a separate CTest entry with
skip return code 77 when the executing machine cannot run AVX2. On the Apple
M1 host that test is explicitly skipped; AVX2 is not compiled for the native
target, so no AVX2 runtime correctness or performance claim is made. The
benchmark identifies unavailable backends and retains the M6.1/M6.2 corpus.

The AVX2 translation unit and baseline dispatcher were cross-compiled with
Apple Clang for `x86_64-apple-macosx14.0`; CMake's x86_64 configuration also
confirmed that only the AVX2 translation unit receives `-mavx2`. This is
compile validation only, not AVX2 execution validation. On the native M1,
three benchmark invocations (same sources, two warmups and nine samples each)
had median-of-invocation medians of 0.272 ms scalar versus 0.055 ms NEON for
classification-only (256.7 versus 1,270.7 MB/s), and 6.621 ms scalar versus
5.251 ms NEON end-to-end (10.53 versus 13.28 MB/s; 2.00 versus 2.52 M
tokens/s). AVX2 was reported unavailable; no performance comparison is made
for it.

### Verilog classifier-to-lexer block reuse (M6.4)

`ClassificationScanner` now belongs to each native lexer invocation and
caches one source-aligned 32-byte classification block. Whitespace, identifier,
escaped-identifier, directive, comment, and string candidate searches reuse
that block until advancing to another block. The scalar/NEON/AVX2 block output
representation and lexical state transitions remain unchanged; cached searches
use existing aggregate flags to skip blocks and preserve byte-exact checks in
the candidate lanes. Backend availability is validated once when the scanner
is created rather than on every block.

The lexer test adds repeated range-search checks across every offset and short
range of an unaligned mixed-byte buffer for scalar and any available optimized
backend. Full-lexer parity also covers long comments, strings, escaped
identifiers, whitespace, and a CRLF positioned on a block boundary. Existing
all-byte, corpus, and Slang differential checks remain enabled.

On Apple M1/arm64 (Release, Apple Clang 21.0.0, two warmups and nine samples
per measured case), the aggregate corpus (69,727 bytes, 13,256 tokens) was
measured in three invocations before and after the cache change. The
median-of-invocation medians were:

| Case | Scalar before | Scalar after | NEON before | NEON after |
| --- | ---: | ---: | ---: | ---: |
| Classification-only | 0.280 ms | 0.283 ms | 0.0538 ms | 0.0571 ms |
| End-to-end lexer | 6.294 ms | 5.462 ms | 5.104 ms | 5.009 ms |

End-to-end throughput changed from 11.08 to 12.77 MB/s for scalar and from
13.66 to 13.92 MB/s for NEON. Classification-only throughput was essentially
unchanged for scalar and about 6% slower for NEON, so this change is not
claimed as a classification throughput win. Per-file end-to-end medians also
improved: scalar Ibex 1.803 to 1.347 ms and 8080 5.001 to 3.870 ms; NEON Ibex
1.364 to 1.234 ms and 8080 4.013 to 3.755 ms. The benchmark now reports each
input separately as well as the aggregate. AVX2 was unavailable at runtime on
the M1, so it remains unmeasured.

Validation on this host: full native CTest passed 22 tests, explicitly
skipped 9 AVX2-only tests, and had 0 failures. The selected EDA Python and EDA
packaging regressions passed 216 tests; 3 extension tests were skipped because
no built `_eda` extension was supplied. The existing Slang-backed production
frontend and `parse_verilog()` path are unchanged.

### Native Verilog/SystemVerilog preprocessor (M7.1)

The independent C++ API is in `include/labneura/eda/verilog_preprocessor.h`,
implemented by `src/labneura/eda/verilog_preprocessor.cpp` and built as part of
the EDA target, including EDA-only builds. Each call owns its macro state and
returns preprocessed text, source-map segments, structured diagnostics,
directive metadata, final macros, and retained source buffers. `origin_at()`
maps output offsets to original source locations and records macro invocation
and definition frames for expanded text.

The engine supports object/function macros, token-aware parameter substitution,
nested expansion and arguments, `undef`, redefinition warnings, recursive and
depth-limit diagnostics, all five conditional directives, and cached relative
or search-path includes with nested-include identities, cycle detection, and
missing-file errors. Inactive text is suppressed without evaluating its
macros/includes, while line endings remain mapped. Timescale and default
nettype values are recorded; begin/end-keywords, line/resetall, cell, and drive
directives are retained as deferred metadata rather than applied. Token paste,
macro stringification, and LF/CRLF macro-definition line continuations are
implemented in M7.1.1. Pasted fragments are re-lexed and malformed
concatenations are diagnosed; stringification substitutes parameters, expands
nested macro arguments, and escapes quotes and backslashes. Undefined macro
references are errors and remain in output. The engine is independent of Slang
types, though tests use Slang's public preprocessor for active-token
differential checks. The existing Slang-backed production parser and
`parse_verilog()` are unchanged.

Focused native tests cover macro definitions and arguments (including empty
arguments, nested parentheses, and nested expansion), recursion, diagnostics,
conditionals, inactive side effects, nested/repeated/missing/cyclic includes,
CRLF/LF and no-final-newline input, large inputs, source-map coverage and
origins, directive metadata, and Slang active-token parity. Release validation
on Apple M1/arm64 with Apple Clang 21.0.0:

- Fresh full native build succeeded. Full CTest: 32 tests, 23 passed, 9
  explicitly skipped AVX2-only tests, 0 failures. The lexer and preprocessor
  tests both passed.
- EDA-only configuration/build and preprocessor CTest passed without Tensor,
  Acceleration Core, or the compatibility archive.
- EDA Python regression selection: 109 passed.
- AVX2 runtime testing was not applicable on the M1.
- Preprocessor benchmark (2 warmups, 9 measured samples): representative
  69,727-byte corpus, 0.883 ms median / 0.029 ms IQR / 79.01 MB/s; synthetic
  607,849-byte macro workload, 15.096 ms / 3.145 ms / 40.27 MB/s; repeated
  include workload counting each processed header (95,000 bytes), 29.037 ms /
  6.031 ms / 3.27 MB/s. These are local measurements, not comparisons against
  Slang or a previous native-preprocessor baseline.

M7.1.1 validation on a fresh Apple M1/arm64 Release build with Apple Clang
21.0.0:

- Identifier, numeric-literal, chained, and object-like token pasting pass
  focused tests. The result is re-lexed as exactly one token; malformed pastes
  produce structured diagnostics while retaining expansion-origin mappings.
- Macro stringification tests cover parameter substitution, nested macro
  expansion, escaped quotes/backslashes, ordinary-string isolation, and
  expansion source mapping. Quote/nested-expansion active-token cases match
  Slang's public preprocessor; backslash escaping is checked against the
  required string spelling because Slang retained the raw `\path` backslash
  in this test case.
- LF and CRLF definition continuations pass native and Slang token-stream
  comparisons. Definition locations continue to point to the original
  physical definition line.
- Full native CTest: 32 tests, 23 passed, 9 AVX2-specific skips, 0 failures.
  The fresh EDA-only CTest selection: 2 passed, 1 AVX2 skip. EDA Python
  regressions: 109 passed. AVX2 runtime remains unavailable on this host.
- `git diff --check` passes. No benchmark was needed for this correctness-only
  change.

### Native Verilog/SystemVerilog syntax parser foundation (M7.2)

The independent C++ API is in `include/labneura/eda/verilog_syntax.h`,
implemented by `src/labneura/eda/verilog_syntax.cpp`. Its primary entry point
accepts a `VerilogPreprocessorResult`; string and file convenience overloads
run native preprocessing, lexing, and parsing. The parser uses only the native
lexer/preprocessor APIs and Slang-independent types. Existing production
`parse_verilog()` remains backed by Slang and was not changed.

The owned AST currently models compilation units, modules, interfaces,
packages, parameter declarations and overrides, ports, simple net/variable
declarations, packed/unpacked ranges, continuous assignments, module
instantiations, named/positional connections, and expression nodes for
identifiers, literals, unary/binary/conditional expressions, calls, selects,
member access, and concatenations. Syntax locations map lexer token offsets
through preprocessor source maps to original files and macro expansion frames.
Diagnostics are structured and source-mapped; recovery synchronizes on
semicolons and design-unit end/begin boundaries.

M7.2 supports the declaration subset documented below; M7.2.1 adds owned
statement nodes for `begin`/`end` blocks (with labels), nested `if`/`else`,
`case`/`casez`/`casex`, blocking/nonblocking assignments, event controls, delay
controls, `wait`, and `always`, `always_comb`, `always_ff`, `always_latch`, and
`initial` blocks. Event controls represent wildcard, level, posedge, and
negedge terms with comma or `or` separators. Assignment targets include
identifiers, member/select expressions, and concatenations. `initial` is
represented syntactically; no simulation or synthesis semantics are implied.
Unsupported procedural loops, fork/join, assertions, tasks/functions, and
other unimplemented statements receive structured diagnostics and are retained
as unsupported statement nodes where encountered. Generate constructs,
typedefs and user-defined types, structs/unions/enums, modports, attributes,
and complete SystemVerilog expression/type grammar remain unsupported.
Package and interface bodies still accept only the earlier declaration
subset. Slang's public `SyntaxTree::fromText` accepts both representative
fixtures; this is a syntax-acceptance differential, not a full AST equivalence
claim.

Fresh Apple M1/arm64 Release validation with Apple Clang 21.0.0:

- Full native build and CTest: 33 tests, 24 passed, 9 AVX2-specific skips,
  0 failures. Lexer, preprocessor, and syntax parser tests passed.
- Fresh EDA-only configuration/build and test selection: 4 tests, 3 passed,
  1 AVX2 skip, 0 failures.
- EDA Python regression selection: 109 passed.
- Parser tests cover module/interface/package declarations, ANSI/non-ANSI
  ports, inherited parameter types, packed/unpacked dimensions, assignments,
  expression precedence, named and positional overrides/connections,
  preprocessor macros/includes, source locations, Slang syntax acceptance,
  and error recovery.
- `git diff --check` passes. AVX2 lexer tests skipped because runtime support
  is unavailable on this Apple Silicon host.

### Native behavioral RTL parser increment (M7.2.1)

The public parser API remains unchanged. `VerilogModuleDeclaration` now owns
procedural blocks; each block owns an optional event control and a recursive
statement tree. The statement AST retains per-node source locations, optional
block labels, expression nodes for conditions/delays/assignments, event edge
terms, and case-item expressions including `default`. Parsing supports
blocking `=`, nonblocking `<=`, identifier/member/bit-select/part-select and
concatenation assignment targets, nested `if`/`else`, all three case variants,
`begin`/`end`, wildcard and edge event controls, comma/`or` event lists,
procedural `#` delays, event-control statements, `wait`, and the four `always*`
forms plus `initial`.

At M7.2.1, loop forms (`for`, `while`, `repeat`, `forever`), fork/join,
assertions, and tasks/functions produced an `UnsupportedProceduralStatement`
diagnostic and an unsupported AST node. M7.6 adds owned AST and parsing for
`for`, `while`, `repeat`, and `forever`; fork/join, assertions, tasks/functions,
and other unimplemented statements remain explicitly unsupported. Recovery
recognizes statement terminators and design-unit starts; malformed procedural
blocks do not consume the next module. The parser does not execute or
synthesize `initial` blocks and does not implement full SystemVerilog
statement/declaration semantics.

The new `tests/data/verilog/behavioral_rtl.sv` fixture covers combinational,
clocked, latch, and initial procedures; reset edges; nested conditionals;
case/casez/casex; labeled blocks; selects and concatenation lvalues; multiple
event separators; delays, wildcard events, wait, macro expansion locations,
and Slang syntax acceptance. Malformed condition, assignment, block, and event
control cases check diagnostics and recovery.

Fresh Apple M1/arm64 Release validation with Apple Clang 21.0.0:

- Full native build and CTest: 33 tests, 24 passed, 9 AVX2-specific skips,
  0 failures.
- Fresh EDA-only build and CTest: 4 tests, 3 passed, 1 AVX2 skip, 0 failures.
- EDA Python regression selection: 109 passed.
- Both the original M7.2 parser fixture and the behavioral RTL fixture were
  accepted by Slang's public syntax parser; no AST equivalence is claimed.
- `git diff --check` passes. AVX2 lexer testing skipped because the Apple
  Silicon host does not support AVX2.

### Native Verilog semantic-analysis foundation (M7.3)

The independent semantic API is declared in
`include/labneura/eda/verilog_semantic.h` and implemented in
`src/labneura/eda/verilog_semantic.cpp`. `analyze_verilog_semantics()` accepts
the native compilation-unit syntax AST and returns an owned scope/symbol
model, resolved/unresolved reference records, and structured source-mapped
diagnostics. The compilation-unit scope owns module, interface, and package
scopes; each design-unit scope collects parameters/localparams, ports,
nets/variables, procedural-block entries, and module instance names before
reference resolution. `lookup_verilog_symbol()` supports nearest-scope and
parent-scope lookup. A reference records the resolving scope and points into
the owned semantic symbol table; module instance symbols link to their module
declaration symbol.

The analyzer resolves identifiers in represented declaration dimensions and
initializers, continuous assignments, supported procedural expressions,
conditions, event controls, case expressions/items, and instance connections
and override values. Package member accesses are resolved for declarations
within this compilation unit. `$` system names and supported type/builtin
forms are not diagnosed as unresolved. M7.9 adds bounded function/task
declaration and call support; details and limitations are recorded below.
Unsupported statement nodes are skipped to avoid cascaded semantic errors.

Diagnostics cover duplicate design/local declarations (with the original
declaration location), duplicate instance names, unresolved visible-scope
identifiers, unknown module types, and unknown named parameter/port
connections when module declarations are available. Positional override
counts are checked. Ordinary module-local lookup does not search sibling
design-unit scopes; qualified package references use the package's own scope.
Imports, general type checking, elaboration, and full language visibility
remain outside this foundation. Production `parse_verilog()` remains
Slang-backed and unchanged.

Semantic tests cover independent module/package scopes, parameter/localparam
and port/net/variable collection, continuous and procedural references,
package-qualified members, valid module/instance/connection resolution,
invalid named overrides/ports, duplicates, unresolved and cross-module
identifiers, diagnostic locations, macro-expansion origins, built-in calls,
and skipping unsupported statement bodies.

Fresh Apple M1/arm64 Release validation with Apple Clang 21.0.0:

- Full native build and CTest: 34 tests, 25 passed, 9 AVX2-specific skips,
  0 failures.
- Fresh EDA-only build and CTest: 5 tests, 4 passed, 1 AVX2 skip, 0 failures.
- EDA Python regression selection: 109 passed.
- `git diff --check` passes. AVX2 lexer test skipped because the Apple
  Silicon host does not support AVX2.

### Native constant evaluation and parameter elaboration (M7.4)

The independent constant evaluator API is declared in
`include/labneura/eda/verilog_constant.h`; hierarchy elaboration is declared
in `include/labneura/eda/verilog_elaboration.h`. They consume the owned native
syntax AST and semantic model. `evaluate_verilog_constant()` returns an
owned bit-vector value with explicit width, signedness, four-state bits,
unbased-unsized fill state, and syntax source location, plus structured
diagnostics. Bits are stored least-significant first; the evaluator applies a
65,536-bit default cap. Supported expressions include integer/based literals,
identifiers, unary/binary arithmetic and bitwise/logical/comparison/shift
operators, conditional expressions, concatenations, replications, constant
selects, `$clog2`, `$signed`, and `$unsigned`. Divide-by-zero, invalid widths
and shifts, unresolved identifiers, unsupported expressions, and invalid
system-function arguments are diagnosed.

The evaluator handles the implemented sizing subset: unsized decimal integer
literals use at least 32 signed bits, unsized based literals use at least 32
bits, sized literals retain their declared width, and binary arithmetic uses
the wider operand width and is signed only when both operands are signed.
Arithmetic results wrap to that width. X/Z bits are retained for literals and
supported bitwise/logical operations; this is not a complete implementation
of all IEEE 1800 context-determined sizing, four-state arithmetic, casts, or
expression rules.

`elaborate_verilog_design()` returns the semantic model, evaluated module
parameter/localparam environments, top-module candidates, an owned parent/
child module-instance tree, and combined diagnostics. Parameter dependencies
are resolved recursively, including forward references; cycles are reported.
Named and positional overrides are evaluated in the caller's parameter
environment and produce fresh per-instance values. Declared integral widths
and packed parameter ranges are applied. Package parameters can be referenced
with package qualification. Explicit top selection is supported; otherwise
the API chooses a single uninstantiated module candidate and reports
ambiguity instead of choosing arbitrarily. Missing module definitions and
recursive module-instantiation paths are diagnosed. Production
`parse_verilog()` remains Slang-backed and unchanged.

At the M7.4 milestone, generate constructs were not yet elaborated; their
current support is documented in M7.5 below. `defparam`, full
type-parameter/type resolution, function-based constant evaluation beyond
the listed system functions, full port/type compatibility, interface
hierarchy, and complete SystemVerilog sizing and four-state semantics remain
unsupported. Slang value/width differential comparison was not performed;
Slang remains a syntax reference in the earlier parser tests, not evidence of
evaluator equivalence.

Focused tests cover sized/unsized and signed/unsigned values, operator
precedence results, X-aware logic, concat/replication, invalid shifts and
division, `$clog2`, source locations, parameter dependencies/localparams,
cycles, packed parameter truncation, package references, per-instance named
and positional overrides, invalid overrides, nested/missing/recursive
hierarchy, top inference, ambiguity, and macro-origin locations.

Fresh Apple M1/arm64 Release validation with Apple Clang 21.0.0:

- Fresh full native build succeeded. Full CTest: 35 tests, 26 passed, 9
  AVX2-specific skips, 0 failures.
- Fresh EDA-only build succeeded. CTest: 6 tests, 5 passed, 1 AVX2-specific
  skip, 0 failures.
- EDA Python regression suites (`test_eda.py`, `test_eda_graph.py`,
  `test_eda_equivalence.py`, and `test_eda_fpga.py`): 216 passed.
- `git diff --check` passes. No Slang constant-value/width differential was
  run; no equivalence claim is made.

### Native generate construct elaboration (M7.5)

The native syntax AST (`include/labneura/eda/verilog_syntax.h`) owns generate
regions and recursive `if`/`else if`/`else`, `case`/`casez`/`casex`, and
`for` constructs. Conditions, case labels, loop initialization/condition/step,
and generated scopes reuse the existing expression and declaration nodes.
The parser recognizes both explicit `generate ... endgenerate` regions and
direct module-item generate constructs, along with nested generate items.
Generate-for bodies are required to be named `begin` blocks; loop update
forms include assignment, `+=`, `-=`, `++`, and `--`.

The semantic analyzer creates scopes for generated blocks and loop genvars,
collects their parameters, declarations, and instance names, and resolves
references through enclosing generated/module scopes without leaking
compilation-unit sibling design units into local lookup. The elaborator uses
the existing constant evaluator and each module instance's own parameter
environment to select conditional/case branches and expand loops. It creates
an independent owned hierarchy for every module instance and loop iteration,
including generated declarations, localparams, assignments, child instances,
and parent links. Paths include generate-block names and loop indices, e.g.
`top.lanes[1].lane`; explicit generate regions do not add a path component.
`casez` and `casex` wildcard matching is supported for the represented
four-state bit patterns. Genvars use signed 32-bit values and the default
iteration limit is 1024, configurable with
`VerilogElaborationOptions::maximum_generate_iterations`.

Nonconstant or unknown conditions, invalid initialization/steps, nonprogress,
genvar overflow, duplicate selected generated names, and iteration-cap
exhaustion produce structured diagnostics. Macro-generated constructs retain
source-map invocation locations and expansion frames. Automatic top-module
inference includes module types instantiated in generate scopes. Production
`parse_verilog()` remains Slang-backed and unchanged; the native APIs remain
independently callable and do not depend on Tensor.

The grammar is deliberately bounded. Generate subroutines, `defparam`, type
parameters, complete SystemVerilog expression and scope semantics, and
complete IEEE case/generate sizing behavior remain unsupported. The Slang
differential checks acceptance of one simple three-iteration generate loop
and compares the elaborated generated-module-instance count; it does not
compare full ASTs or hierarchy names.

Focused native generate tests cover true/false and else-if branches, case
selection/default and casez/casex wildcards, zero and multiple iterations,
nested constructs, per-instance parameter conditions, generated declarations
and instances, hierarchy paths, enclosing-scope semantic lookup, unresolved
conditions, invalid progress, iteration limits, duplicate names, macro
locations, and the Slang instance-count differential.

Fresh Apple M1/arm64 Release validation with Apple Clang 21.0.0:

- Fresh full native build succeeded. Full CTest: 35 tests, 26 passed,
  9 AVX2-specific tests skipped, 0 failures.
- Fresh EDA-only configure/build succeeded with Tensor, Acceleration Core,
  and compatibility archive disabled. CTest: 6 tests, 5 passed, 1
  AVX2-specific lexer test skipped, 0 failures.
- EDA Python regression suites (`test_eda.py`, `test_eda_graph.py`,
  `test_eda_equivalence.py`, and `test_eda_fpga.py`): 216 passed.
- `git diff --check` passes. AVX2 runtime testing was unavailable on this
  Apple Silicon host.

### Native procedural statement AST and parser (M7.6)

The existing procedural AST was extended with a `Loop` statement kind and
typed `for`, `while`, `repeat`, and `forever` loop forms. `for` statements own
optional initializer and step assignment statements, an optional condition,
and a nested body; the parser supports simple assignments, compound arithmetic
and bitwise updates, and postfix `++`/`--`. A single declared loop variable
(for example `int i = 0`) is retained with its type and source location, and
semantic analysis creates a local procedural scope for it. Standard
module-declared loop variables resolve through the surrounding module scope.
Loop initializer lists and more complex declaration forms are not supported.

Event-control and delay-control statements now have distinct AST kinds rather
than sharing a generic timing-control kind. Existing support for all five
procedural block forms (`initial`, `always`, `always_comb`, `always_ff`, and
`always_latch`), labeled `begin`/`end`, blocking/nonblocking assignments,
`if`/`else`, case variants/default, event lists/wildcards, delays, and `wait`
is retained. Procedural `assign`, `deassign`, `force`, and `release` are
diagnosed and retained as unsupported statement nodes. Semantic traversal
resolves loop conditions, initializer/update expressions, event expressions,
case items and nested bodies, and diagnoses invalid assignment targets.

This is not simulation or synthesis support: loops are not executed, delays
and event scheduling have no runtime semantics, and `initial` remains a
syntax-only construct. Intra-assignment timing, comma-separated loop
declaration lists, `do` loops, fork/join, assertions, callable syntax outside
the M7.9 subset, and full SystemVerilog statement/type semantics remain
unsupported. The Slang
differential accepts the representative loop fixture; AST equivalence is not
claimed. Production `parse_verilog()` remains Slang-backed and unchanged.

Tests cover all procedural block kinds, blocking/nonblocking assignments,
nested blocks/labels and conditionals, all case variants/default, for/while/
repeat/forever, declared loop variables, event and delay controls, macros,
source locations, unsupported procedural-continuous forms, malformed-loop
recovery, semantic loop-variable resolution, unresolved references, invalid
assignment targets, and Slang syntax acceptance.

Fresh Apple M1/arm64 Release validation with Apple Clang 21.0.0:

- Fresh full native build and CTest: 35 tests, 26 passed, 9 AVX2-specific
  skips, 0 failures.
- Fresh EDA-only build and CTest: 6 tests, 5 passed, 1 AVX2-specific lexer
  skip, 0 failures.
- EDA Python regression suites (`test_eda.py`, `test_eda_graph.py`,
  `test_eda_equivalence.py`, and `test_eda_fpga.py`): 216 passed.
- Slang accepted the procedural loop syntax fixture; no AST equivalence is
  claimed. AVX2 runtime testing was unavailable on this Apple Silicon host.
- `git diff --check` passes.

### Native expression type analysis (M7.7)

The independent type API is in `include/labneura/eda/verilog_type.h` and
`src/labneura/eda/verilog_type.cpp`. It returns owned integral, real, string,
and unknown type records with optional width, signedness, two-/four-state
behavior, packed-dimension widths, and unbased-unsized status. Type information
is separate from constant values. `analyze_verilog_expression_types()` walks
module, interface, package, declaration, continuous-assignment, procedural,
and generate expressions using the native semantic model. Expression type
results are keyed by AST node; symbol type keys point into the caller-owned
semantic result, which must outlive their use. Standalone expression and
declared-type inference functions are also available.

The supported sizing subset is explicit: unsized decimal integer widths reuse
the native constant evaluator's at-least-32-bit signed sizing; sized based
literals use their explicit width, and unsized based literals follow the
evaluator's minimum-32-bit, digit-based width. Built-in `bit`, `logic`/net,
integer atom, `integer`, and `time` declarations derive width, signedness, and
state model from the declared form and constant packed dimensions. Binary
arithmetic/bitwise expressions use the maximum operand width and are signed
only when both operands are signed; shifts use the left operand type.
Comparisons, logical/reduction operators, bit-selects, and constant
part-selects use one-bit or range-derived unsigned results; conditional
expressions use the wider branch; concatenations sum widths and replications
multiply the constant count by operand width. User-defined types and widths
that cannot be resolved remain unknown.

Known assignment width and signedness conversions produce source-located
warnings, not hard errors. Parameter and localparam types are collected,
including forward-inferred untyped parameter expressions. Elaborated module
instances retain `parameter_types` separately from their parameter values;
declared parameter types also drive the existing supported truncation and
extension in elaboration. The native type result agrees with Slang on a tested
binary arithmetic expression for width, signedness, and state model. This is
not complete IEEE context-determined sizing or full type checking; dynamic
arrays, type parameters, broad assignment-context propagation, full mixed-type
promotion, and exhaustive invalid-type diagnostics are unsupported.

Fresh Apple M1/arm64 Release validation with Apple Clang 21.0.0:

- Fresh full native build and CTest: 36 tests, 27 passed, 9 AVX2-specific
  skips, 0 failures.
- Fresh EDA-only build and CTest: 7 tests, 6 passed, 1 AVX2-specific lexer
  skip, 0 failures.
- EDA Python regression suites (`test_eda.py`, `test_eda_graph.py`,
  `test_eda_equivalence.py`, and `test_eda_fpga.py`): 216 passed.
- Focused Slang comparison matched native binary-expression width,
  signedness, and four-state status.
- `git diff --check` passes. AVX2 runtime testing was unavailable on this
  Apple Silicon host.

### Native SystemVerilog data type expansion (M7.8)

The native syntax AST owns typedef aliases, enum declarations/enumerators,
struct/union member declarations, packed/unpacked dimensions, and declaration
locations. Type definitions resolve recursively within module/interface,
package, and generate scopes. Forward typedef references
within a scope are supported; package-qualified types are available without
leaking module-local names between sibling scopes. Alias metadata and original
type identity remain available on resolved type records.

Enums preserve enum identity plus their underlying integral width,
signedness, and four-state status. Explicit and implicit values use the native
constant evaluator; package-qualified enumerator values are available to
constant evaluation and elaboration. Struct member types are retained; packed
structs compute aggregate width and member offsets, while unpacked structs do
not claim a packed width. Packed unions use shared offset zero and require
equal known integral member widths. Unpacked unions preserve member types
without inventing a layout. Unknown member types and invalid member selection
are diagnosed.

Fixed unpacked arrays are represented as nested array types with separate
dimension/count metadata and element types. Multidimensional indexing returns
the next array or element type. Packed dimensions are independent of unpacked
element counts. Typedef-based parameter types, including enum and integral
typedefs, propagate into each elaborated instance's parameter type/value
environment.

This is not full SystemVerilog type compliance. Dynamic/associative/queue
arrays, type parameters, imports, complete aggregate assignment compatibility,
unpacked-array part-selects, full enum range/overflow checking, and complete
context-determined sizing are unsupported. Unknown custom types remain
unknown and produce diagnostics instead of being coerced to integers.
Differential tests compare a supported enum's width, signedness, and
four-state status with Slang; this is not a general equivalence claim.

Fresh Apple M1/arm64 Release validation with Apple Clang 21.0.0:

- Fresh full native configure/build succeeded. CTest: 36 tests, 27 passed,
  9 AVX2-specific tests skipped, 0 failures.
- Fresh EDA-only configure/build succeeded with Tensor, Acceleration Core,
  and compatibility disabled. CTest: 7 tests, 6 passed, 1 AVX2-specific
  lexer test skipped, 0 failures.
- EDA Python regression suites (`test_eda.py`, `test_eda_graph.py`,
  `test_eda_equivalence.py`, and `test_eda_fpga.py`): 216 passed.
- Focused native tests compare enum width, signedness, and four-state status
  with Slang; no broader equivalence claim is made.
- `git diff --check` passes. AVX2 runtime testing was unavailable on this
  Apple Silicon host.

### Native callable and legacy-port support (M7.10)

The owned AST and parser support ANSI and legacy/non-ANSI module ports,
including declaration-list ports followed by body declarations. Bounded
function and task declarations remain supported in modules, interfaces, and
packages. Declarations retain function/task kind, `automatic`/`static`
lifetime, return type spelling and packed dimensions, ordered typed/directed
formals, local declarations, nested statement bodies, and source-map
locations. Calls preserve positional or `.formal(actual)` named argument
syntax, and explicit return statements are represented in the statement AST.
An omitted function return type defaults to one-bit `logic`; task calls are
distinct procedural statements while function calls remain expressions.

Semantic analysis creates Function/Task symbols and callable child scopes.
Ordered formal symbols preserve direction, type, dimensions, and locations.
Callable bodies resolve against formal and local declarations. Known calls
are checked for function-versus-task context. Binding supports either all
positional or all named actuals, diagnoses mixed/duplicate/unknown/missing
arguments, checks output/inout/ref lvalues, and reports known integral
width/signedness conversions as warnings. Ref compatibility requires matching
known types. Package-qualified callable references resolve when the package
is present.

The function evaluator and task executor synchronously execute a bounded
subset using the existing four-state constant values and per-invocation
environments. Functions support locals, blocking assignments, return values
or function-name assignment, and nested supported function calls without
mutating the caller environment. Tasks support locals, nested task calls,
supported sequential statements, and successful output/inout/ref writeback to
simple identifier actuals. Recursive re-entry and configurable shared
statement/call-depth limits are diagnosed. Nonblocking assignments,
event/delay controls, `wait`, unsupported expressions/statements, and
non-identifier task output writeback are diagnosed rather than simulated.
This is not simulation scheduling or complete SystemVerilog compliance.

Slang differential coverage is syntax acceptance only, not semantic or
execution equivalence. Production `parse_verilog()` remains Slang-backed and
unchanged; the native APIs remain independently callable and Tensor-independent.

Fresh Apple M1/arm64 Release validation with Apple Clang 21.0.0:

- Fresh full native configure/build succeeded. CTest ran 41 tests: 32 passed,
  9 AVX2-specific tests skipped, 0 failures.
- Fresh EDA-only configure/build succeeded with Tensor, Acceleration Core,
  and compatibility disabled. CTest ran 12 tests: 11 passed, 1 AVX2-specific
  lexer test skipped, 0 failures.
- EDA Python regression suites (`test_eda.py`, `test_eda_graph.py`,
  `test_eda_equivalence.py`, and `test_eda_fpga.py`): 216 passed.
- Focused callable tests cover legacy ports and Slang syntax acceptance,
  named/positional binding and compatibility, function evaluation, task
  execution including casez/casex wildcard behavior, source locations,
  recursion, and execution limits.
- `git diff --check` passes. AVX2 runtime testing was unavailable on this
  Apple Silicon host.

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
  bool values. BOOL arithmetic and BOOL-to-BOOL comparisons are not implemented.
  `logical_not`, `logical_and`, and `logical_or` have Generic, NEON, and AVX2
  implementations. `logical_not` and `logical_and` passed x86_64 runtime
  validation; `logical_or` still requires x86_64 runtime validation.
  `where()` remains unsupported.
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

### BOOL logical operations (M3.4.1–M3.4.5)

`Tensor::logical_not()` and Python `Tensor.logical_not()` accept BOOL tensors
and return independent, contiguous BOOL tensors with the same shape. The
Generic inverts canonical values (`0` to `1`, `1` to `0`) without mutation or
conversion. NEON uses 16-byte vector comparisons and canonicalizes each vector
mask to BOOL bytes, followed by a scalar tail. AVX2 uses 32-byte unaligned
loads/stores, compares against zero, masks with one to produce canonical BOOL
bytes, and handles the remaining elements with a scalar tail. All backends
support scalar and zero-element tensors and contiguous offset views.
`Tensor::logical_and()` and Python `Tensor.logical_and()` require two BOOL
inputs, use the shared broadcast plan, and return a new independent BOOL tensor
with the broadcast output shape. Generic iterates effective broadcast strides;
NEON and AVX2 use native vector bytewise AND for contiguous inputs and gather
broadcast lanes before vector AND otherwise. All kernels retain canonical 0/1
bytes and handle the scalar tail and zero-element outputs. Backend pairs must
match; no Generic fallback is used. Existing Tensor views are contiguous
offset views; arbitrary non-contiguous layouts are not currently provided.
`Tensor::logical_or()` and Python `Tensor.logical_or()` follow the same BOOL,
broadcasting, allocation, and backend-pair contract, using bytewise OR in the
SIMD kernels and canonical BOOL output. `where()` remains unsupported.

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

M3.4.3 AVX2 `logical_not` runtime validation passed in GitHub Actions run
`42449211301` on implementation commit
`665991bdf5f790297f29dc1423674d6278947d68`. The x86_64 runner reported AVX2
support and selected the AVX2 backend; `bool_dtype_avx2` passed. Full native
CTest reported 29 tests, 0 failures, and 7 expected NEON skips. The full
AVX2-selected Python suite reported 1160 passed, 216 skipped, and 3 warnings.

M3.4.4 `logical_and` passed x86_64 AVX2 runtime validation in GitHub Actions
run `42455884336`, commit
`1089815717ce37815210ec425ec295678c2d96ef`. The runner reported x86_64 and
AVX2 support; `bool_dtype_avx2` passed, full CTest reported 29 passed and 0
failed, and the full AVX2-selected Python suite reported 1162 passed, 217
skipped, and 3 existing warnings. The ARM64 development host also ran its
Generic and NEON logical-and tests.

M3.4.5 adds Generic, NEON, and AVX2 `logical_or` kernels with explicit
backend tests. Focused native and Python BOOL tests passed on the ARM64
development host; AVX2 cases were skipped because this build does not provide
that backend. x86_64 runtime validation for `logical_or` remains pending.

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

### Callable semantics hardening and differential validation (M7.11)

M7.11 hardens the native callable behavior through semantic and execution
tests. Binding coverage now checks positional and named ordering, missing and
excess actuals, parameterized formal widths, input width/signedness warnings,
`ref` two-state/four-state compatibility, and source-located diagnostics.
Function evaluation coverage checks fresh local frames across invocations,
local initialization, blocking execution, nested calls, signed extension and
width truncation on return, parameterized formal/return types, conversion from
four-state actuals to two-state formals, unsupported expressions, recursion,
and execution limits. Task execution coverage checks reversed named actual
ordering and writeback, repeated task-local initialization, transactional
output writeback failure for select actuals, nested calls, and case/casez/casex
execution.

Slang differential tests compare actual constant-function values, result bit
patterns, return width/signedness/four-state status, nested named-call binding,
parameterized callable types, repeat-invocation local isolation, and rejection
of a missing required actual. The test filters Slang warnings separately from
errors. Slang's public constant evaluator does not execute tasks as a
procedural simulation oracle, so task side effects/output values are not
claimed as Slang differentials. Slang syntax acceptance alone is not treated
as semantic parity.

Unsupported behavior remains explicit: task output/inout/ref writeback is
limited to simple identifier actuals; non-identifier targets are diagnosed.
Global side effects, recursive callable re-entry, persistent `static` local
state, nonblocking assignments, timing/event controls, `wait`, and
four-state procedural branch merging are not simulated. Conditions and case
selectors in execution require known integral values. Unsupported expressions
or statements fail with diagnostics. This is a bounded deterministic
interpreter, not full SystemVerilog simulation.

Fresh Apple Silicon M1/arm64 Release validation on 2026-11-16:

- Full native build succeeded. CTest ran 42 tests: 33 passed, 9 AVX2-only
  tests skipped, 0 failures. NEON tests executed.
- Fresh EDA-only Release build succeeded with Tensor, Acceleration Core, and
  compatibility disabled. CTest ran 13 tests: 12 passed, 1 AVX2 lexer test
  skipped, 0 failures.
- Python EDA regression suites (`test_eda.py`, `test_eda_graph.py`,
  `test_eda_equivalence.py`, and `test_eda_fpga.py`): 216 passed.
- `git diff --check` passed. Production `parse_verilog()` remains backed by
  Slang; the native EDA frontend remains independently callable and Tensor-
  independent. No existing user worktree changes were discarded.

### Native callable timing controls (M7.12)

The native syntax AST reuses the existing distinct delay/event statement
representations and event term edge/expression fields. The parser now also
accepts bare `@signal` controls, in addition to `@(signal)`, edge terms,
comma/`or` event lists, and wildcard syntax. Semantic analysis resolves bare
event signal references through the same existing term traversal.

Task execution interprets `#expr` and `#(expr)` as nonnegative known integral
logical ticks up to 64 bits. Delays preserve statement order, evaluate against
the current task frame (including formals and supported nested function
calls), and advance a checked `uint64_t` virtual clock without sleeping. `#0`
continues immediately in source order and does not model a delta region. The
task result reports elapsed ticks. During positive delays, a caller-supplied
ordered external transition trace can update visible task-frame signals at
absolute logical ticks.

Task event controls consume transitions in trace order. Level events trigger
on an actual value change; `posedge`/`negedge` use four-state transition
classes, including X/Z transitions. Event-list terms are OR-combined. Nested
and repeated waits share the trace cursor and virtual time. Exhausting the
trace returns `NoMatchingEvent`; it is a failed, transactional execution, not
a completed task or a resumable suspension. Outputs are written back only on
successful completion.

Explicit diagnostics reject unsupported delay expressions, negative/unknown/
too-wide delay values, tick overflow, out-of-order or malformed transition
entries, wildcard event controls, and edge events that do not evaluate to one
bit. `wait`, nonblocking assignments, timing in functions, and event controls
in top-level `initial`/`always*` procedural blocks remain unsupported by the
callable interpreter. It does not implement a scheduler, delta cycles,
timescale conversion, event regions, or resumable task frames. Slang tests
confirm timing syntax acceptance only; Slang's available APIs here do not
provide procedural simulation results for a behavioral differential.

Fresh Apple Silicon M1/arm64 Release validation on 2026-11-19:

- Fresh full native Release build succeeded. CTest ran 44 tests: 35 passed,
  9 AVX2-only tests skipped, 0 failures. NEON tests executed.
- Fresh EDA-only Release build succeeded with Tensor, Acceleration Core, and
  compatibility disabled. CTest ran 15 tests: 14 passed, 1 AVX2 lexer test
  skipped, 0 failures.
- Prescribed Python EDA regression suites (`test_eda.py`, `test_eda_graph.py`,
  `test_eda_equivalence.py`, and `test_eda_fpga.py`): 216 passed.
- Slang accepted the bare `@signal` syntax fixture. No procedural event or
  delay execution differential is claimed because this environment's Slang
  API does not provide a behavioral simulation result oracle.
- `git diff --check` and checks for the new untracked delay/event test files
  passed. Production `parse_verilog()` remains Slang-backed and unchanged.

The subsequent M2.1 reshape/flatten milestone was also validated on Apple
Silicon:

- Native build completed; full CTest passed all runnable tests, with six
  AVX2-only tests skipped on this host.
- Focused Python nD/reshape tests passed (37 tests); the full Python suite
  completed successfully with the existing warnings and hardware-specific
  skips.
- Generic and NEON reshape/flatten native tests ran. AVX2 runtime behavior
  remains unverified locally because this host is Apple M1.

### Native lvalue resolution and assignment semantics (M7.13)

The independent native Verilog frontend exposes a shared lvalue resolver and
assignment engine using the existing `VerilogSemanticSymbol`,
`VerilogIntegralType`, and `VerilogConstantValue` representations. Resolved
targets carry symbol/storage identity, declared type/range, destination width,
signedness and state kind, plus LSB-first physical bit mappings. The syntax AST
now distinguishes bit, constant part, indexed upward, and indexed downward
selects; the parser records `+:` and `-:` explicitly.

The resolver supports simple integral identifiers, packed bit-selects,
constant part-selects, indexed `+:`/`-:` selects with runtime bases, and nested
concatenations. It maps ascending and descending declared ranges to physical
storage bits. Runtime bit/base expressions are evaluated in the active
callable frame. Known out-of-range selections preserve their lvalue width,
ignore corresponding writes, and read as `X`. Unknown indices, nonconstant
part-select bounds/indexed widths, invalid widths, unsupported target kinds,
unpacked arrays, multidimensional packed arrays, and arithmetic overflow
produce explicit diagnostics.

The assignment engine handles full-width and masked writes, RHS truncation and
signed extension, unbased unsized values, two-state `X`/`Z` to zero conversion,
four-state preservation, and concatenation bit splitting. It stages all
destination values and commits only on success. Ambiguous overlapping writes
inside one concatenation fail transactionally with
`OverlappingAssignmentDestinations`.

Function and task blocking assignments use the shared resolver/engine for
callable-local variables and function return storage. Task output and inout
actuals accept selected or concatenated lvalues; inout reads and output target
resolution happen at task-call entry, and caller writeback is committed only
after successful execution. `ref` execution is explicitly rejected because
aliasing is not modeled. Nonblocking assignments, global procedural side
effects, unpacked array lvalues, multidimensional packed selects, and full
simulation scheduling remain unsupported. Indexed part-select reads in general
expressions are diagnosed as unsupported by the constant evaluator; indexed
part-select lvalue targets remain supported.

Slang differential coverage executes a constant function that writes a packed
part-select and compares its value with Slang's constant evaluation. Callable
selected/concatenated task output tests use explicit expected values; no task
simulation equivalence is claimed because the available Slang API does not
provide that oracle. Production `parse_verilog()` remains Slang-backed.

Fresh Apple Silicon M1/arm64 Release validation on 2026-11-19:

- Full native Release build succeeded. CTest ran 47 tests: 38 passed,
  9 AVX2-only tests skipped, 0 failures.
- Fresh EDA-only Release build succeeded with Tensor, Acceleration Core, and
  compatibility disabled. CTest ran 18 tests: 17 passed, 1 AVX2 lexer test
  skipped, 0 failures.
- Prescribed Python EDA regression suites (`test_eda.py`, `test_eda_graph.py`,
  `test_eda_equivalence.py`, and `test_eda_fpga.py`): 216 passed.
- The assignment differential CTest passed alongside the native expected-value
  tests. No procedural task or simulator equivalence is claimed.
- `git diff --check` and new-file whitespace checks passed. Existing dirty and
  untracked worktree changes were preserved; no commit or push was made.

### Slang versus native frontend comparative profiling (M7.18.1, partial)

Opt-in Release benchmarks now measure matched syntax, forced semantic, and
complete public frontend boundaries without changing production behavior.
Deterministic generation covers eleven workload families; schema, generator,
profiling-wrapper, and aggregation tests are registered or runnable
independently. Slang remains pinned to `11.0.0+7ddf4059f` and its hierarchical
time trace is captured without modifying the dependency.

The initial Apple M1 baseline covers 33 matched workloads at 1 KiB, 10 KiB,
and 100 KiB. Median Slang/native ratios across all workloads were 0.203 for
syntax, 0.195 for semantic, and 0.207 for complete paths; values below one mean
native was slower. Complete output structures differ materially because native
lowering expands generic gates while public `parse_verilog()` retains
behavioral structure, so these are boundary-cost measurements rather than
equivalent-netlist speedups. On the 100 KiB Boolean fixture native generated
12,039 nets and 10,030 cells; peak process RSS was 84.3 MB versus Slang's
17.2 MB. NEON improved native syntax median by about 4.2% over scalar.

Native sampling identified semantic-reference linear searches and their
underlying string/location comparisons as the dominant hotspot. Slang's trace
reported `parseFile` and `AST Instance` as its largest captured scopes. Full
raw results and the report are under
`benchmarking/results/m7181-m1-20261119/`; methodology and limitations are in
`docs/eda-frontend-profiling.md`. Linux AVX2/perf, allocation tracing, Slang
top-20 sampling, and controlled 1 MiB/4 MiB complete runs remain pending, so
M7.18.1 is not declared complete. No optimization was implemented.

Fresh local validation: full Release CTest registered 70 tests, with 60 passed,
10 expected ARM64 AVX2 skips, and no failures. EDA-only Release CTest registered
40 tests, with 39 passed, one expected AVX2 lexer skip, and no failures. The
prescribed Python EDA suites passed all 216 tests. Twelve new Python profiling,
workload, and aggregation tests passed, as did the native schema CTest.

### Cross-platform stabilization and Linux AVX2 verification (M7.17.1)

The baseline checkpoint `9d11240` is present on `origin/main` through merge
`69508d2`. Stabilization changes add a baseline-compiled AVX2 constructor that
checks CPU/OS support before allocation, a strict AVX2 integer arithmetic
regression, portable signed-reference boundary handling, explicit scalar
lexer coverage, and a package-version test that follows the package source
instead of a stale literal. CI now covers Linux x86-64 GCC/Clang, Linux ARM64,
macOS ARM64, strict Generic/AVX2/NEON execution, EDA-only configurations,
Yosys formal CTest, and EDA sanitizers.

Fresh local Apple Silicon M1/arm64 Release validation on 2026-11-19:

- Full native CTest: 69 registered; 59 passed and 10 skipped because this
  ARM64 host cannot execute AVX2. The skipped tests are
  `avx2_int8_multiplication`, `avx2_integer_arithmetic`,
  `fp32_avx2_allocating`, `add_into_avx2`, `tensor_nd_avx2`,
  `tensor_comparison_avx2`, `bool_dtype_avx2`, `avx2_allocation_safety`,
  `avx2_fp16_arithmetic`, and `eda_verilog_lexer_avx2`.
- Forced Generic CTest: all 7 focused tests passed with no skips. Forced NEON
  selection was confirmed and all 7 focused NEON tests passed with no skips.
- EDA-only Release build succeeded with Tensor, Acceleration Core, and
  compatibility all disabled. CTest ran 39 tests: 38 passed, including the
  native Yosys formal-equivalence test; only `eda_verilog_lexer_avx2` skipped
  because AVX2 is unavailable.
- Prescribed EDA Python regressions: 216 passed. Full Python suite: 1,212
  passed, 225 skipped, and 3 pytest return-value warnings. The skips divide
  into 91 AVX2-only cases unavailable on ARM64, 125 cases for broadcast
  operations documented as Generic-only, and 9 packaging cases requiring
  optional compiled-extension environment variables not supplied by this
  source-tree run. `tests/python_packaging/test_core_package.py`: 10 passed.
- The new baseline AVX2 constructor passed local Clang syntax checking; this
  ARM64 host cannot compile or execute its x86-specific CMake target.
- `git diff --check` and workflow YAML/step-schema checks passed.

Linux x86-64 GCC/Clang, Linux ARM64, hosted macOS ARM64, runtime AVX2, hosted
NEON, and sanitizer results remain unverified until the stabilization GitHub
Actions jobs complete. M7.17.1 is not complete until those required jobs pass.

### Native RTL verification and differential validation (M7.17)

Test support now provides a deterministic, two-state evaluator for AND2, OR2,
XOR2, INV, MUX2, BUF, TIEHI, and TIELO. It maps bits least-significant first
from declared right bounds and reports malformed connectivity, multiple
drivers, undriven nets, cycles, resource exhaustion, and unsupported cells.
Optional sequential cutpoints are explicit. A separate bounded evaluator
models DFF/DFFR/DFFS transitions from explicit initial state and atomic input
frames; all registers triggered by a transition use the same pre-update state.
It covers both clock polarities, hold/enable, synchronous and supported
asynchronous active-high/low reset/set, including reset asserted in frame zero.
It is not a SystemVerilog scheduler.

Seven combinational fixtures exhaustively compare native netlist outputs with
independent expected functions over 4,856 two-state vectors. Slang generic-map
differential checks agree over 160 vectors in two fixtures; five fixtures
report Slang comparison as Unavailable with explicit unsupported cell or
mapping diagnostics while completing native exhaustive checks. This is not a
syntax-acceptance equivalence claim. Randomized checks use fixed seed
0x4d3731375eed for 24 depth-three expression trees and four procedural
branches, exhaustively checked over 512 vectors. Six sequential test scenarios
submit 36 bounded stimulus frames, including negative validation cases.
Yosys proves three combinational gold-versus-native designs and detects one
deliberately wrong AND2 model. No sequential formal equivalence is claimed.

The tests exposed two production lowering defects and added regressions. Type
inference now recognizes e/E as hexadecimal digits in sized based literals
rather than real exponents. Assignment context widths propagate through
supported arithmetic, bitwise, shift, unary, and conditional expressions for
continuous, combinational procedural, and sequential assignments. Continuous
assignment analysis also allows precise expression diagnostics for dynamic
selects rather than stopping at generic process eligibility uncertainty.

Fresh Release validation on 2026-11-19:

- Fresh full native build succeeded. CTest: 68 total, 59 passed, 9 expected
  AVX2 skips, 0 failures.
- Fresh EDA-only Release build succeeded with Tensor, Acceleration Core, and
  compatibility disabled. CTest: 39 total, 38 passed, 1 expected AVX2 lexer
  skip, 0 failures.
- Prescribed Python EDA regression suites: 216 passed.
- All five Slang differential CTests passed. The two new supported generic-map
  fixtures agreed for all 160 vectors; five additional new fixtures report
  Slang mapping unavailable, with native expected checks exhaustive for all
  4,856 vectors.
- Optional Yosys formal run: 3 combinational proofs passed; deliberate
  inequivalent candidate detected. Yosys was available, so the formal CTest
  was not skipped.
- The native lowering dependency-boundary audit, strict compilation through
  fresh Release builds, and whitespace checks passed. Existing modified and
  untracked worktree changes were preserved; no commit or push was made.

### Independent native AST-to-RTL lowering (M7.16)

The explicit `lower_verilog_native()` API consumes a native compilation unit,
matching elaboration and type results, and M7.15 synthesis eligibility. It
builds the existing `Module`, `Port`, `Net`, and `Instance` model directly and
returns a module only after full validation. The entry point does not invoke
Slang. Production `parse_verilog()` remains Slang-backed.

Supported lowering is deliberately bounded to one flat elaborated module,
integral signals with at most one resolved packed range, supported known
constants and operators, constant selections and concatenations, continuous
assignments, complete blocking combinational processes with supported if/case
control, and canonical single-clock nonblocking sequential processes. Cells
include AND2, OR2, XOR2, INV, MUX2, BUF, TIEHI/TIELO, DFF, DFFR, and DFFS.
Sequential forms include positive and negative edges, conditional hold and
enable muxing, synchronous reset data muxing, and recognized constant-value
asynchronous resets. Unsupported expressions, four-state constants,
dynamic-selects, unsupported local storage, incomplete combinational logic,
latches, unresolved/multiple drivers, unsupported resets, inout ports, and
hierarchy fail with diagnostics and no partial module.

Fresh Release validation on 2026-11-19:

- Full native build succeeded. CTest: 63 total, 54 passed, 9 expected AVX2
  skips, 0 failures.
- EDA-only build succeeded with Tensor, Acceleration Core, and compatibility
  disabled. CTest: 34 total, 33 passed, 1 expected AVX2 skip, 0 failures.
- Prescribed Python EDA regression suites: 216 passed.
- Four existing Slang differential CTests and the new native-vs-Slang
  differential CTest passed. The new fixture compares all four AND input
  vectors and passes an optional Yosys `equiv_make`/`equiv_simple` proof.
- Sequential checks are structural; no sequential equivalence claim is made.
- Strict Clang syntax checks, tracked and new-file whitespace checks, and the
  native lowering dependency-boundary audit passed.
- Existing modified and untracked worktree changes were preserved. No commit
  or push was made.

### Native procedural control flow and execution completeness (M7.14)

The native statement AST now represents do-while, break, and continue
explicitly. A reusable structural pass reports maximum loop nesting, return
presence, unsupported statement classes, and source-located diagnostics for
break/continue outside loops. Function and task entry points run this
validation before executing callable bodies.

Execution uses one shared typed outcome: normal completion, break, continue,
return (with optional value), unsupported execution, budget exhaustion,
diagnostic failure, or no matching event. Shared branch and loop helpers serve
both function and task executors, replacing duplicated conditional, case,
and loop selection logic. Supported loops are for, while, do-while, repeat,
and forever; break exits its nearest loop, continue begins its next iteration
and still runs a for update, and return/failure propagates through nested
blocks, branches, and loops. The shared statement budget is charged for each
loop iteration, including empty forever iterations, and remains shared across
nested callable invocations.

Procedural if selects true only when the condition contains a definite 1;
X/Z-only conditions select else. Case compares four-state values exactly,
casez treats Z as wildcard, and casex treats X/Z as wildcards on either side.
Case items are checked in source order, with default used only when no item
matches. Timing controls retain M7.12 logical ticks and transition-trace
behavior.

Unsupported behavior remains explicit: wait, nonblocking scheduling, global
procedural side effects, process execution, and a full simulation scheduler
are not modeled. Forever loops terminate only through break, return/failure,
or the configured step budget. Slang differentials compare constant-function
results only where its public constant-evaluation API is a valid oracle;
task effects use expected-value tests, not simulation equivalence.

Fresh Apple Silicon M1/arm64 Release validation on 2026-11-19:

- Fresh full native Release build succeeded. CTest ran 51 tests: 42 passed,
  9 AVX2-only tests skipped, 0 failures.
- Fresh EDA-only Release build succeeded with Tensor, Acceleration Core, and
  compatibility disabled. CTest ran 22 tests: 21 passed, 1 AVX2 lexer test
  skipped, 0 failures.
- Prescribed Python EDA regression suites: 216 passed.
- M7.14 Slang control-flow differential test passed. It compares function
  values for nested loops, returns, four-state if, and case/casez/casex.
  Existing task tests verify transactional rollback on execution-limit errors.
- All CTests, including existing delay/event and callable regressions, passed.
  The four new helpers have no Tensor includes or API dependencies.
- `git diff --check` and all new-file whitespace checks passed. Existing dirty
  and untracked worktree changes were preserved; no commit or push was made.

### Native process analysis and synthesis eligibility (M7.15)

The public `verilog_process_analysis.h` API discovers source-ordered
continuous assignments and `always`, `always_comb`, `always_ff`,
`always_latch`, and `initial` blocks. Process declaration intent is retained
independently from inferred combinational/sequential classification. Records
carry semantic module/signal identities, source locations, event terms,
assignment categories, timing/unsupported flags, and diagnostics.

Dependency analysis records raw signal reads and writes, constant packed
selection ranges, dynamic-select uncertainty, control/case/loop expressions,
and implicit sensitivity separately. Same-module function bodies contribute
captured signal reads; unresolved call side effects remain unknown. Assignment
coverage is path-sensitive and bit-level for packed signals, branches, cases,
concatenations, and supported loop execution guarantees. Unknown coverage
remains unknown; only known incomplete assignments imply potential storage.

Sequential analysis recognizes one positive- or negative-edge clock,
structurally matched synchronous/asynchronous reset candidates, active
polarity, enable/hold patterns, and blocking versus nonblocking assignment
categories. Additional unmatched edge events are ambiguous. This analysis
does not simulate processes or schedule nonblocking assignments.

The coordinator aggregates `Eligible`, `Unsupported`, and `Indeterminate`
results and source-located diagnostics. The current supported subset includes
resolved continuous assignments, complete `always_comb`, latch-producing
`always_latch`, and resolved single-clock sequential candidates. Explicitly
unsupported cases include timing-controlled processes, incomplete
`always_comb`, unsupported statement forms, ambiguous multiple-edge clocks,
and conflicting procedural writers. Initial processes are retained but
indeterminate; dynamic or unresolved dependencies are indeterminate. Multiple
continuous drivers on declared net symbols/ports remain subject to net
resolution policy. Cross-process writers are deduplicated by process identity,
so multiple assignments within one process do not create a false conflict.

The analysis currently runs per source module. It does not merge generated or
instance-hierarchy drivers, resolve instance-port connections, lower RTL, or
generate cells. The process differential test compares native versus Slang
process inventories and then checks native eligibility expectations; it does
not infer synthesis equivalence from Slang syntax acceptance.

Fresh Apple Silicon M1/arm64 Release validation on 2026-11-19:

- Fresh full native Release build succeeded. CTest ran 57 tests: 48 passed,
  9 AVX2-only tests skipped, and 0 failed.
- Fresh EDA-only Release build succeeded with Tensor, Acceleration Core, and
  compatibility disabled. CTest ran 28 tests: 27 passed, 1 AVX2 lexer test
  skipped, and 0 failed.
- Prescribed Python EDA regression suites (`test_eda.py`, `test_eda_graph.py`,
  `test_eda_equivalence.py`, and `test_eda_fpga.py`): 216 passed.
- All four Slang differential CTests passed, including process-inventory and
  native expected-result checks. No procedural synthesis equivalence is
  claimed.
- The native EDA implementation has no Tensor or SIMD API dependency.
- `git diff --check` and new-file whitespace checks passed. Existing dirty and
  untracked worktree changes were preserved; no commit or push was made.
