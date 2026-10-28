# EDA algorithms and pseudocode

This is the algorithm-focused companion to [the flow guide](eda-flow.md).
Pseudocode describes the current implementation at a useful level; it does not
imply that every production-EDA optimization is implemented.

## A. Parsing and elaboration

### Full Verilog parser

The full parser delegates language grammar, type checking, hierarchy, and
elaboration to slang rather than trying to parse Verilog with regular
expressions.

```text
parse_verilog(file):
    choose language mode from file extension (.v or .sv)
    configure slang with file and include directory
    parse syntax trees; report errors
    elaborate and type-check compilation; report errors
    find the unique non-library top module
    summarize top ports, nets, instances, assignments, and process presence
    return Module summary
```

Project parsing runs the frontend over recursively discovered `.v`/`.sv` files,
then stores an AST JSON snapshot and summarizes modules reachable from
top-level instances. Slang elaboration resolves parameters and generate
constructs before LabNeura lowers the result.

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
