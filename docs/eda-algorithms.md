# EDA algorithms and pseudocode

This is the algorithm-focused companion to [the flow guide](eda-flow.md).
Pseudocode describes the current implementation at a useful level; it does not
imply that every production-EDA optimization is implemented.

## A. Parsing and elaboration

### Native Verilog parser

Production parsing uses the native lexer, preprocessor, syntax, semantic, type,
elaboration, and project-projection pipeline. Unsupported constructs diagnose
explicitly; Slang is optional differential-validation infrastructure only.

```text
parse_verilog(file):
    choose language mode from file extension (.v or .sv)
    preprocess and parse native syntax; report errors
    elaborate and type-check native compilation; report errors
    find the unique non-library top module
    summarize top ports, nets, instances, assignments, and process presence
    return Module summary
```

Project parsing runs the frontend over recursively discovered `.v`/`.sv` files,
then stores an AST JSON snapshot and summarizes modules reachable from
top-level instances. Slang elaboration resolves parameters and generate
constructs before LabNeura lowers the result.

### Independently callable native frontend

The experimental native frontend has separate lexer, preprocessor, syntax
parser, semantic analyzer, type analyzer, constant evaluator, and elaborator
APIs. It owns its token/source-map, syntax, scope, type, constant, and
elaboration representations and has no Tensor dependency. Its current
supported syntax includes legacy/non-ANSI module ports, typed callable
declarations, named and positional function/task actuals, and owned procedural
statements. Semantic analysis resolves callable scopes; binding handles
positional or named arguments, checks required formals and output lvalues, and
reports known integral conversion warnings. The native function evaluator and
task executor synchronously handle the documented statement subset with
per-call frames, nested supported calls, recursion detection, and execution
limits.

Native callable execution is a bounded compile-time-style subset, not a
simulator: function evaluation returns a value without mutating the caller;
task output/inout values are written back transactionally after successful
execution. After M7.13, those actuals may be identifiers, supported packed
selections, or concatenations. `ref` aliasing, nonblocking assignments,
`wait`, unsupported expression forms, and unsupported writeback targets
produce diagnostics. The native frontend is independently callable and is not
the implementation behind production `parse_verilog()`, which remains
Slang-backed.

## B. Behavioral synthesis: statements to a bit-level graph

The behavioral synthesizer operates on the **elaborated AST**. It walks scopes
to collect variables, continuous assignments, procedural blocks, instances,
and instantiated generate blocks. Values are represented as least-significant
bit first words (`Word = vector<Bit>`). Each Boolean operation creates or
reuses a node in a directed graph.

### Structural hashing and constant folding

The `BitGraph` builder simplifies common cases as nodes are created:

- `x AND 0` becomes `0`, `x AND 1` becomes `x`, and `x AND x` becomes `x`;
- `x OR 1` becomes `1`, `x OR 0` becomes `x`;
- `x XOR 0` becomes `x`, `x XOR x` becomes `0`;
- double negation is removed;
- commutative operands are put in stable order and identical operations share a
  structurally hashed node;
- muxes with a constant select or identical data arms simplify.

This is local Boolean simplification and common-subexpression sharing, not a
full global Boolean optimizer.

### Expressions and arithmetic

Expressions are evaluated into bit vectors of the elaborated width, with
sign-extension rules where required. `WordOps` lowers implemented arithmetic
into Boolean operators. Addition is a ripple-carry implementation:

```text
ADD(a, b):
    carry = 0
    for bit i from least significant to most significant:
        partial = a[i] XOR b[i]
        sum[i] = partial XOR carry
        carry = (a[i] AND b[i]) OR (partial AND carry)
    return sum
```

Subtraction is `a + NOT(b) + 1`; negation is `0 - a`. Multiplication forms
shifted AND partial products and adds them. Unsigned division uses restoring
division: shift the partial remainder, compare it with the divisor, subtract
when it fits, and record the quotient bit. Signed division takes absolute
values, performs unsigned division, then restores quotient and remainder
signs. Shifts use conditional mux stages, one stage per shift-amount bit.

Comparisons, reductions, equality, selection, and conditional expressions are
also composed from Boolean nodes and muxes. Constant expressions can be
evaluated during lowering; nonconstant calculations become logic. This can
create large graphs for expensive operations such as division.

### Procedural control flow and state

`if` conditions become Boolean values. Each branch executes against an
independent symbolic state; branch results are merged bitwise with a mux.
`case` arms compare the subject with each item and merge symbolic states
similarly. Static loops are unrolled by repeatedly executing the body until
the elaborated constant stop condition is false.

Blocking assignments update the current process state immediately.
Nonblocking assignments in sequential processes are accumulated as next-state
values. On process completion:

- a combinational process must assign a value on every path to be purely
  combinational;
- if a value retains an unassigned marker, the synthesizer derives data and an
  enable condition and emits a latch cell;
- if a clocked process assigns a variable, the synthesizer emits a flip-flop
  with D, clock, and optional asynchronous controls;
- asynchronous reset/set values must be constant;
- dual-edge processes use paired rising/falling state elements in the current
  lowering.

```text
lower clocked process:
    state = symbolic copy of current signal values
    execute statements:
        blocking write -> update current symbolic value
        nonblocking write -> update scheduled next value
        branch -> mux-merge branch states
    for each assigned signal bit:
        D = scheduled nonblocking value if present, else current value
        decode asynchronous reset/set conditions and values
        create DFF variant with D, clock, and controls
        drive signal bit from the DFF output
```

### Rebuild and emit generic cells

After symbolic execution, a memoized graph rebuild copies only logic reachable
from outputs and state inputs. It detects combinational cycles and rejects
inferred transparent storage where the emitter cannot legally represent an
incomplete assignment. It then emits generic instances:

```text
Not -> INV              And -> AND2          Or -> OR2
Xor -> XOR2             Mux -> MUX2
Flop -> DFF / DFFR / DFFS / DFFSR
LatchCell -> LATCH      Tri -> TRIBUF
constant -> TIEHI / TIELO when a connection requires one
```

## C. Generic mapping for continuous assignments

The structural path recursively lowers an expression into a destination net.
Associative binary operators with the same operator are flattened into one
operand list. For example, `(a & b) & c` becomes `AND3(a,b,c)` rather than two
serial AND cells. A conditional `s ? x : y` maps to
`MUX2(S=s,A1=x,A0=y)`.

For a dynamic bit-select on a power-of-two-sized bus, it makes a binary mux
tree:

```text
values = [bus[0], bus[1], ..., bus[N-1]]
for selector bit j from low to high:
    pair adjacent values
    each pair -> MUX2(select=selector[j], A0=low, A1=high)
    values = pair outputs
return the single remaining value
```

Constant literals create tie cells. Expressions the mapper cannot translate
are rejected instead of producing a partial netlist.

## D. Technology mapping: generic cell family matching

This is a deterministic compatible-cell lookup, not a Boolean matching engine.

```text
for each generic instance:
    if an exact cell exists in Liberty:
        attach it and keep its connections
    else:
        recognize generic family and required input count
        candidates = library cells matching family and arity
        if no candidate: fail with missing-cell error
        prefer candidate whose drive suffix is "_1"
        among candidates with same drive preference, prefer smallest known area
        map generic input pins to library input pins with matching names
        map generic Y to the cell's functional output pin
        attach selected Liberty cell metadata
```

Aliases such as `AND2_X1` may first be normalized to a generic alias. Existing
non-generic instances are retained, and an exact library match is attached if
available. Because this stage does not change Boolean structure, correctness
depends on the library cell's pin/function semantics agreeing with the
generic-cell convention.

## E. FPGA LUT mapping: cut enumeration

### What is a cut?

For a combinational output node, a **cut** is a set of boundary nets such that
the node's value can be computed from those nets without crossing the boundary.
If the cut has at most `K` leaves, its whole cone is representable by one
K-input LUT. For `f = (a AND b) XOR c`, `{a,b,c}` is a 3-leaf cut; if an
intermediate `t = a AND b` is already available, `{t,c}` is a 2-leaf cut.

### Candidate generation

The mapper classifies gates, builds producer/consumer links, records input and
constant leaves, and topologically sorts combinational nodes using Kahn's
algorithm. A cycle causes an error. Wide associative AND/OR/XOR families are
rewritten as 2-input trees so the same cut algorithm handles them.

At each node, candidate cuts are built by combining choices at each input:

```text
cuts(node):
    partial = { empty set }
    for each input:
        if input is a constant: choices = { empty set }
        else:
            choices = { {input} } plus each retained cut of input's driver
        partial = every union(base, choice) with at most K leaves
        deduplicate partial cuts
        keep up to 4 * 8 candidates ranked by score
    retain the best 8 cuts for this node
    best_cut = first retained cut
    depth(node) = 1 + max(depth(leaf) for leaf in best_cut)
    flow(node) = 1 + sum(flow(leaf) / max(1, fanout(leaf)))
```

Primary inputs have depth and flow zero; constants contribute no leaf. Cuts
are ranked lexicographically by lower estimated depth, then lower area-flow
estimate, then fewer leaves. Because only a bounded set is retained and each
node selects its own best cut, this is a depth-oriented, area-recovering heuristic
over the enumerated choices—not an exhaustive globally optimal mapper.
Fanout division helps avoid overcharging shared logic in the estimated flow.

### Selecting live nodes

The mapper starts from nets that must remain observable: output ports and
inputs of cells that pass through FPGA mapping (for example DFFs). It walks
backwards through each node's selected cut and marks the needed producers.
Combinational logic outside these live cones is not emitted.

### LUT truth-table construction

For a cut of `m` leaves, there are `2^m` input patterns. The mapper evaluates
the cut's root logic in parallel using a 64-bit machine word: bit `p` of each
word represents that signal's value under pattern `p`.

The pattern word for input `i` alternates blocks of `2^i` zeros and `2^i`
ones. This yields all 64 assignment combinations for up to six LUT inputs.
Gate operations such as AND and XOR become native word operations; a mux
selects bitwise between its two data words. The resulting 64-bit word is the
truth table; only the lowest `2^m` bits are retained.

```text
truth_table(root, cut):
    for leaf i:
        value[leaf_i] = pattern_word(i)
    evaluate nodes from cut boundary toward root:
        value[node] = Boolean_function(value[input_0], ...)
    return low 2^m bits of value[root]
```

The integer truth table is formatted as uppercase hexadecimal `INIT`, padded
to `ceil(2^m / 4)` hex digits. Its least significant bit is pattern zero.
Input `I0` is pattern bit zero, so it is the least-significant truth-table
variable.

For a full adder with inputs `A,B,Cin`, FPGA tests assert two LUT3s:

- Sum = A XOR B XOR Cin, `INIT = 96` hex;
- Cout = majority(A,B,Cin), `INIT = E8` hex.

This small example is useful for manually verifying the `I0`/INIT bit order.

## F. BLIF output

BLIF output maps each LUT to a `.names` truth-table cover. It walks pattern
numbers from zero through `2^K - 1`; for each INIT bit that is one, it writes a
row of input bits followed by output `1`. Tie cells become constant `.names`
records. A plain `DFF` becomes a rising-edge `.latch`; other cells are emitted
as `.subckt` so their type and named pin connections remain visible.

This is serialization only. A downstream tool must interpret, optimize, place,
route, and/or implement the BLIF if a physical FPGA result is desired.

## Native callable assignment semantics (M7.13)

The independent native callable interpreter resolves assignment lvalues to
semantic storage symbols and explicit least-significant-first physical bit
maps. It supports integral identifiers, packed bit-selects, constant
part-selects, indexed `+:`/`-:` part-selects, and concatenations, including
ascending and descending packed ranges. Runtime select indices are evaluated
in the active callable frame. Known out-of-range selected bits are ignored on
write and read as `X`; unknown indices, unsupported targets, invalid widths,
and arithmetic overflow produce diagnostics.

Assignments resize the right-hand value to the complete lvalue width, using
the expression's signedness for extension, then split concatenation bits in
Verilog order. Two-state destinations map `X` and `Z` to zero; four-state
destinations preserve them. The assignment engine stages all changed storage
and commits only after the complete target validates. Ambiguous writes where
multiple concatenation components target the same storage bit fail
transactionally with `OverlappingAssignmentDestinations`.

Function and task blocking assignments use this resolver and assignment
engine for callable-local storage. Task output and inout actuals can be selected
or concatenated lvalues. Inout values are read at call entry; output targets
are resolved at call entry and all caller writeback remains transactional until
successful task completion. `ref` execution, nonblocking assignments, global
procedural side effects, unpacked arrays, multidimensional packed selections,
indexed part-select reads in general expressions, and a general procedural
simulation scheduler remain unsupported. Production
`parse_verilog()` remains Slang-backed; the native syntax parser is independent
and is not a replacement for that path.

## Native procedural control flow (M7.14)

Callable execution now shares a structured outcome contract for normal
completion, break, continue, return, unsupported execution, budget exhaustion,
diagnostic failure, and the existing no-matching-event timing result. Reusable
structural analysis reports loop nesting, possible returns, unsupported
statements, and break/continue placement errors. The parser retains
do-while/break/continue syntax in the native AST.

Function and task execution share loop and branch helpers. Supported loops are
for, while, do-while, repeat, and bounded forever; break exits the nearest
loop, continue proceeds to the next iteration (including the for update), and
return/failures propagate outward. The shared statement-step budget covers
each iteration and nested callable execution. If treats X/Z-only truth as
false. Case matching supports exact four-state case, casez, and casex
wildcards, source-ordered first match, and default.

This remains bounded callable execution, not procedural simulation. Timing
controls keep M7.12's virtual logical ticks and finite transition trace.
Nonblocking scheduling, wait, global side effects, and process/scheduler
semantics remain unsupported. Slang differential tests compare constant
function values where its public constant evaluator is a valid oracle; task
side effects use expected-value tests only.

## Native process analysis and synthesis eligibility (M7.15)

The independent native AST now has a static process-analysis API. It discovers
continuous assignments, generic `always`, `always_comb`, `always_ff`,
`always_latch`, and `initial` blocks in source order and preserves declared
process intent separately from inferred combinational or sequential structure.
Dependency records resolve signal accesses by semantic symbol identity, retain
constant bit/part-select ranges, mark dynamic selects unknown, and expose raw
reads separately from implicit sensitivity. Known function bodies contribute
captured signal dependencies; unresolved task/function side effects remain
unknown.

Assignment coverage follows reachable branches and case alternatives with
per-bit states for packed signals. It handles static selections and
concatenations, treats potentially zero-iteration loops conservatively, and
does not infer storage from reads alone. Sequential analysis records blocking
and nonblocking assignment categories, single edge clocks, structurally
matched synchronous and asynchronous reset candidates, and ambiguous extra
edge events. This is static analysis only: no process is executed and no
nonblocking assignment scheduling is implemented.

Eligibility is reported as `Eligible`, `Unsupported`, or `Indeterminate` with
source-located diagnostics. The current subset accepts complete
`always_comb`, latch-producing `always_latch`, supported single-clock
sequential processes, and resolved continuous assignments. It rejects
incomplete `always_comb`, timing-controlled processes, unsupported statements,
unmatched multi-edge clock candidates, and conflicting procedural writers.
Initial blocks remain classified but indeterminate because native RTL
initialization lowering is not implemented. Dynamic or unresolved dependencies
also remain indeterminate. Multiple continuous drivers on nets are retained
for net-resolution policy instead of being blanket-rejected.

This API analyzes modules independently and does not yet merge drivers across
elaborated hierarchy or resolve instance-port connectivity. It does not lower
RTL, generate cells, or claim hardware equivalence. Production
`parse_verilog()` remains Slang-backed; the native analysis is independent of
Tensor and SIMD APIs.

## Independent native AST-to-RTL lowering (M7.16)

`lower_verilog_native()` is an explicit opt-in entry point that consumes the
existing native syntax, elaboration, type, and synthesis-eligibility results.
It does not parse or reanalyze source and does not call Slang. Production
`parse_verilog()` remains Slang-backed. Successful lowering returns the
existing `Module`/`Port`/`Net`/`Instance` representation; failure is atomic and
returns source-located diagnostics without a partial module.

The first subset is one flat elaborated module with resolved integral signals,
at most one packed range, and no instance or generate hierarchy. It lowers
Boolean gates, supported unary/binary/conditional expressions, known constants,
constant selects, concatenations, blocking combinational processes with
complete assignments and supported case/if control, and canonical
single-clock nonblocking register processes. The bit-level cells used are
AND2, OR2, XOR2, INV, MUX2, BUF, TIEHI, TIELO, DFF, DFFR, and DFFS. Negative
clock edges are represented by an INV before DFF. Signed values are extended
from their sign bit and unsigned values from zero; truncation removes high
bits, and packed vectors are connected least-significant bit first while
retaining their declared port ranges.

The subset accepts dynamic packed bit-select reads only when a known unsigned
index has a proven contiguous range within the declared one-dimensional packed
range. The proof accepts unsigned constants and identifiers, concatenations
(including zero extension), unambiguous continuously driven computed selectors,
and overflow/underflow-free addition or subtraction by a known constant. The
interval must be power-of-two aligned, so its low index bits directly select a
balanced MUX2 tree; this avoids inventing outputs for impossible index values.
The tree uses the declaration's actual packed ordering, including nonzero,
ascending ranges. Dynamic writes, dynamic part-selects, signed or unknown-width
indices, sparse or unproven ranges, and arithmetic that can overflow or
underflow remain unsupported. The subset rejects X/Z constants, unsupported arithmetic (including
nonconstant multiplication/division), unsupported local
storage, ambiguous drivers, incomplete combinational assignments, latches,
unsupported reset patterns, inout ports, and hierarchy. Eligibility is only a
necessary precondition, not proof that every operation has a hardware mapping.
No technology mapping is performed. Native-vs-Slang differential validation
currently compares all four truth-table rows for a simple AND fixture; it does
not establish general or sequential equivalence. Sequential fixtures use
structural cell and connection checks.

## Native RTL functional and differential verification (M7.17)

The test-support netlist evaluator implements a bounded two-state semantics
for AND2, OR2, XOR2, INV, MUX2, BUF, TIEHI, and TIELO. It maps each port bit
from its declared right bound in least-significant-bit-first order, propagates
constants and fanout, evaluates dependencies deterministically, and reports
malformed pins, duplicate or multiple drivers, undriven nets, combinational
cycles, resource limits, and unsupported cells. Sequential cutpoints must be
enabled and supplied explicitly. This is test infrastructure; it does not add
a second production netlist engine.

The sequential test evaluator requires explicit initial state for every
lowered register and consumes atomic input snapshots. It implements DFF,
DFFR, and DFFS transitions, positive and negative clock edges, hold/enables,
synchronous reset behavior embedded in the data path, and supported active
high/low asynchronous reset pins. Registers triggered by a shared edge compute
their next values from the old state and commit together. Asynchronous reset
and set assertion is applied before observing a frame, including frame zero.
The evaluator bounds traces and cell evaluations. These bounded traces are
functional checks, not an event scheduler or an unbounded sequential proof.

Seven combinational fixtures exhaustively checked 4,856 two-state input
vectors against independently authored expected functions. They cover gates
and adders, width and truncation behavior, signed and unsigned comparisons,
logical and bitwise operators, shifts and extensions, packed ranges and
selections, concatenation and replication, conditionals and case priority,
blocking assignment order, partial-bit writes, and parameters. Native checks
agree across the complete domains. Slang-backed generic mapping agrees on 160
vectors in the conditional/case and partial-bit/parameter fixtures. Five
fixtures report Slang comparison as unavailable with explicit mapping or
cell-subset diagnostics; those fixtures still complete native oracle checks.
This does not infer equivalence from syntax acceptance.

Deterministic randomized checks use seed `0x4d3731375eed` and exhaust 512
vectors over 24 depth-three expression trees and four procedural branch
expressions. Negative cases check unsupported operators, X constants,
dynamic selects, latches, conflicting drivers, unresolved references,
hierarchy, inout ports, invalid reset patterns, analysis identity, atomic
failure, and deterministic repeated lowering.

Bounded sequential checks cover positive/negative edges, multi-cycle hold,
enable, synchronous reset, active-high/low asynchronous assertion and
deassertion, frame-zero reset, and simultaneous pre-update state. Initial
state is never guessed. Yosys proves three combinational gold-versus-native
designs using the existing generic cell models, and the harness confirms that
a deliberately wrong AND2 model fails equivalence. Sequential formal proof
is not claimed.

M7.17 verification also exposed and fixed two lowering defects. Hexadecimal
digits `e` and `E` are no longer mistaken for real-number exponents in sized
based literals. Assignment context width now propagates through supported
arithmetic, bitwise, shift, unary, and conditional expressions in continuous,
combinational procedural, and sequential assignment lowering. Continuous
assignments reach expression-level checks so dynamic selects report their
specific unsupported diagnostic instead of being masked by generic process
eligibility uncertainty.
