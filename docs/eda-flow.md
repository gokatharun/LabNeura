# From Verilog to a mapped netlist

This guide explains the EDA implementation in LabNeura. The short version is:

```text
Verilog source
    |
    +-- slang parse + elaboration --> RTL summary (Module or Project)
                                            |
                                            +-- generic_map --> generic Boolean/state netlist
                                                                  |
                                     +----------------------------+------------------+
                                     |                                               |
                                  tech_map                                      fpga_map
                                     |                                               |
                         Liberty standard cells                       LUTs + passthrough state
                                     |                                               |
                                     +---------------------- BLIF / JSON / DOT ------+
```

`tech_map` and `fpga_map` are alternative targets for the generic netlist.
Technology mapping is not a prerequisite for FPGA mapping.

## 1. The main data model

The EDA API is declared in [netlist.h](../include/labneura/eda/netlist.h).

- A **Module** has ports, nets, instances, continuous assignments, and a flag
  indicating whether procedural behavioral RTL still needs synthesis.
- A **Project** contains summarized modules, their source-file paths, and an
  elaborated AST serialized as JSON.
- An **Instance** has a cell type, named pin-to-net connections, optional cell
  metadata, and optional parameters. For LUTs, `parameters["INIT"]` is the
  truth table.
- **Net names** for behavioral designs are bit-oriented (for example
  `count[0]`). A bus port may therefore be represented as multiple named bit
  nets after synthesis.
- A **StandardCellLibrary** holds cell area, pin direction/function and
  capacitance, and timing arcs/look-up tables.

## 2. Parse and elaborate Verilog

### Full file and project parsing

`parse_verilog(path)` invokes the pinned **slang** frontend. It selects the
Verilog-2005 language mode for `.v` and SystemVerilog-2017 mode for `.sv`,
parses and elaborates the design, then summarizes the selected top-level
module into LabNeura's `Module`.

The summary retains ports, declared nets, instantiated child-module and
primitive instances, and continuous assignments. Expressions are represented
as trees with their source text, operator, operands, and bit width where the
AST exposes them. `always`/procedural blocks are recorded as behavioral RTL;
they are not converted to gates by `parse_verilog`.

For a file, `parse_verilog` expects exactly one non-library top-level module
and returns a `Module`. For a directory, it recursively collects `.v` and
`.sv` source files, sorts them, asks slang to resolve/elaborate the project,
gathers modules reachable from top-level instances, and returns a `Project`
with an AST JSON snapshot. Pass a top-module name to mapping when the project
has more than one top.

The parsed model retains the Slang frontend context needed for behavioral
lowering, so `generic_map` does not reread or reparse source files. Parameter
overrides or selecting a non-root module may require re-elaboration from the
retained syntax trees, but not another source parse.

### Simulation-only and unsupported constructs

The full parser reports compilation errors instead of guessing at invalid
designs. `initial` and `final` blocks have no hardware netlist form and are
rejected by the single-file netlist summary. Project summaries may retain the
module shell, but synthesis does not turn simulation-only behavior into
hardware. Inout ports are also unsupported by behavioral synthesis.
Unsupported lowering fails explicitly.

## 3. Generic mapping: RTL to technology-independent cells

Call `generic_map(rtl_module)` for a parsed module, or
`generic_map(project, top, parameters)` for a project. The result uses logical
cells rather than a particular vendor library.

### Structural/continuous-assignment path

For a non-behavioral module, supported scalar assignment expressions are
lowered into generic cells:

- associative `&` / `&&`, `|` / `||`, and `^` expressions become multi-input
  `ANDn`, `ORn`, and `XORn` cells;
- unary `~` / `!` becomes `INV`;
- a conditional expression becomes `MUX2`;
- a literal Boolean constant becomes `TIEHI` or `TIELO`;
- a variable bit-select with a power-of-two bus and variable selector becomes
  a tree of `MUX2` cells;
- `and`, `or`, and `xor` primitive instances become `ANDn`, `ORn`, and `XORn`.

Associative operators are collected into a single multi-input generic cell
rather than needlessly making a chain. A final `BUF` may be added when lowering
an expression into a temporary signal before driving its assignment target.
Unsupported operators, non-scalar assignment targets, and X/Z constants fail
instead of being silently dropped.

### Behavioral path

When a module has procedural blocks, or a project contains hierarchy or
parameter overrides, `generic_map` re-elaborates the source through slang and
uses the behavioral synthesizer. This path traverses the elaborated hierarchy,
handles generated blocks that exist after elaboration, expands supported
arrays/words into individual bits, and creates a bit-level Boolean graph.

Procedural `if` and `case` branches are modeled as muxes between possible
values. A clocked process creates D flip-flops; asynchronous reset/set controls
are represented by the appropriate generic state-cell variants. Missing
assignments in combinational processes indicate storage: latches are inferred
when the assignment condition is not exhaustive. A for-loop is statically
unrolled and must have a constant bound.

The bit graph structurally hashes repeated Boolean operations and folds
constants as it is built. Word operations are expanded to bit logic: addition
uses a ripple carry, subtraction uses two's complement, multiplication uses
shifted partial products and addition, and variable shifts/selections use
muxes. The behavioral emitter writes generic `AND2`, `OR2`, `XOR2`, `INV`,
`MUX2`, `DFF`-family, `LATCH`, `TRIBUF`, and tie cells. High-level arithmetic
is therefore Boolean logic before either target mapper sees it.

The behavioral path supports a useful synthesis subset, not the complete
Verilog language. Some dynamic selects, nonconstant loops, unsupported
statement kinds, non-integral values, and particular power expressions are
rejected with an error. Parse success alone does not guarantee synthesis.

## 4. Standard-cell technology mapping

Load a Liberty library using `parse_liberty(path)` and call
`tech_map(generic, library)`. A project overload performs generic lowering
first.

For each generic instance, the mapper:

1. keeps a cell with an exact type match when one exists;
2. otherwise recognizes a generic gate family and finds a compatible library
   cell (including matching input count);
3. prefers a unit-drive (`_1`) cell if available, then chooses the smallest
   area among equivalent candidates when area data is present;
4. connects generic input names to same-named input pins and translates the
   generic `Y` output to the cell's output pin;
5. stores the selected Liberty cell on the mapped instance.

Non-generic instances that already match an exact cell are attached to that
cell; otherwise they are preserved. A generic cell without a compatible
library implementation is an error. This stage is a cell-family matcher, not
Boolean optimization, logic restructuring, timing-driven sizing, or physical
design.

### What Liberty parsing supplies

The Liberty reader tokenizes nested groups, parses library/cell/pin attributes,
collects lookup-table templates and timing-table values, and attaches
combinational and sequential timing arcs to pins. The returned model includes
time and capacitance units, cell area, pin direction/function/capacitance,
related pins, timing types/senses, and supported rise/fall/constraint tables.

It is a targeted reader for commonly used data, not a complete Liberty
semantic validator. `assets/sky130_fd_sc_hd__tt_025C_1v80.lib` is the bundled
default; the education-oriented library is `assets/cells.lib`.

## 5. FPGA mapping: generic logic to LUTs

Call `fpga_map(generic, lut_size=4)` for LUT sizes 2 through 6, or pass a
project and top. Run `generic_map` first when starting from a parsed behavioral
module. Sequential and other cells that should not be absorbed (such as
flops, latches, and tristates) pass through; their combinational input cones
are still mapped to LUTs.

The mapper classifies combinational nodes and their nets, topologically orders
the combinational graph, enumerates candidate **cuts** of at most `K` inputs,
scores each cut first by logic depth and then estimated area flow, and turns
selected cuts into LUT truth tables. It also:

- breaks wide AND/OR/XOR-family gates into a 2-input tree for cut enumeration;
- keeps a bounded number of promising cuts per node (currently 8), so this is
  a practical heuristic, not exhaustive global optimization;
- preserves logic required by output ports and by inputs of pass-through cells;
- computes each LUT truth table by evaluating its cone for all input patterns;
- preserves constants with tie cells and preserves pass-through sequential
  instances.

`LUTn` pins are `I0` through `I(n-1)`, then `O`. The `INIT` hex parameter is
indexed with `I0` as the least significant input bit: bit `p` gives the output
for input pattern `p`. Each LUT instance includes cell metadata. The default
FPGA library is an illustrative architecture model (configuration-bit area,
simple delay estimates, pF capacitance, no routing delay); pass a Liberty
library if you need that library's cell attributes.

The mapper rejects combinational loops, missing gate outputs, multiple
combinational drivers, invalid LUT sizes, and modules that have not yet been
generic-mapped. Its area-flow score is a heuristic used to compare cuts; it
does not claim a globally minimum LUT count or routed timing.

## 6. Exports and inspection

- `to_json` / `write_json`: serialize the LabNeura netlist model.
- `to_dot` / `write_dot`: draw ports, cells, and driver-to-sink edges for
  Graphviz.
- `to_blif` / `write_blif`: output `.names` truth tables for LUTs and ties,
  `.latch` records for plain rising-edge `DFF`s, and `.subckt` records for
  other cells such as asynchronous flops.

BLIF is an interchange representation for tools such as Yosys, ABC, or VPR.
Emitting BLIF is not the same as running synthesis, placement, routing, or
bitstream generation in those tools.

## 7. Implemented versus planned

The code currently implements parsing/elaboration, generic lowering,
technology-cell mapping, FPGA LUT mapping, and netlist inspection/export.
Directories for RTL/generic/technology/FPGA optimization, placement, and
routing are placeholders. LabNeura's FPGA mapper creates a LUT netlist; it
does not place LUTs on a device, route wires, model routing congestion, or
produce a bitstream.

## 8. Source map

| Stage | Implementation |
|---|---|
| Verilog parse/elaboration | [`verilog_parser.cpp`](../src/labneura/eda/parser/verilog_parser.cpp) |
| Liberty reader | [`liberty_parser.cpp`](../src/labneura/eda/parser/liberty_parser.cpp) |
| Generic assignment mapping | [`generic_mapping.cpp`](../src/labneura/eda/mapping/generic/generic_mapping.cpp) |
| Behavioral synthesis | [`behavioral_synthesis.cpp`](../src/labneura/eda/mapping/generic/behavioral_synthesis.cpp), [`bit_graph.h`](../src/labneura/eda/mapping/generic/bit_graph.h) |
| Standard-cell mapping | [`technology_mapping.cpp`](../src/labneura/eda/mapping/technology/technology_mapping.cpp) |
| FPGA LUT mapping | [`fpga_mapping.cpp`](../src/labneura/eda/mapping/fpga/fpga_mapping.cpp) |
| Default FPGA model | [`fpga_library.cpp`](../src/labneura/eda/mapping/fpga/fpga_library.cpp) |
| BLIF output | [`netlist_blif.cpp`](../src/labneura/eda/netlist/netlist_blif.cpp) |
| Tests | [`test_eda.py`](../tests/test_eda.py), [`test_eda_fpga.py`](../tests/test_eda_fpga.py), [`test_eda_graph.py`](../tests/test_eda_graph.py) |
