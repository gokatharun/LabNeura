# Walkthrough: the sequential calculator

This walkthrough uses [`calculator.v`](../examples/verilog/calculator.v). It is
a compact example of how RTL constructs become state and combinational logic.
The exact LUT partition depends on `lut_size` and the mapper's cut heuristic.

## The RTL, in plain language

The module has:

- `clk`: clock input;
- `op[1:0]`: operation selector;
- `a[3:0]`, `b[3:0]`: 4-bit operands;
- `result[4:0]`: registered 5-bit result.

The two zero-extended operands are
`a5 = {1'b0, a}` and `b5 = {1'b0, b}`. At each rising edge:

| `op` | Next `result` |
|---|---|
| `00` | `a5 + b5` |
| `01` | `a5 - b5` |
| `10` | zero-extended `a & b` |
| `11` | zero-extended `a \| b` |

`<=` is a nonblocking assignment. This is not a software program that runs
through the table; it describes a combinational next-value function feeding
five flip-flops.

## Stage 1: parse and elaborate

```python
import labneura.eda as eda

rtl = eda.parse_verilog("examples/verilog/calculator.v")
print(rtl.name)            # calculator
print(rtl.is_behavioral)   # true: contains always
```

Slang parses the syntax, resolves widths and expression types, and supplies an
AST. LabNeura's `Module` summary records ports and marks this design
behavioral. Parsing does not yet create a DFF or calculate an adder.

## Stage 2: lower to a generic netlist

```python
generic = eda.generic_map(rtl)
```

Because the module is behavioral, `generic_map` uses the behavioral
synthesizer. Conceptually, the lowering does this:

```text
create bit inputs for clk, op[0..1], a[0..3], b[0..3]

sum        = 5-bit add(zero_extend(a), zero_extend(b))
difference = 5-bit subtract(zero_extend(a), zero_extend(b))
and_result = zero_extend(bitwise_and(a, b))
or_result  = zero_extend(bitwise_or(a, b))

next_result = case(op):
    00 -> sum
    01 -> difference
    10 -> and_result
    11 -> or_result

on rising edge of clk:
    result[0..4] <= next_result[0..4]
```

The case is converted into branch predicates and mux logic. For each bit, the
branch values become the D input of one flip-flop. There should therefore be
five state bits in the generic netlist, subject to constant or redundant logic
simplifications. The netlist contains DFFs, muxes, and two-input Boolean cells.
Addition and subtraction become bit-level carry logic; they are not still
abstract `ADD`/`SUB` cells at this stage.

Inspect the mapped cells:

```python
from collections import Counter

print(Counter(instance.cell_type for instance in generic.instances))
for instance in generic.instances:
    print(instance.name, instance.cell_type,
          [(c.pin, c.net) for c in instance.connections])
```

Exact generated names are internal and may change. Learn the netlist from cell
types, pins, and graph connectivity rather than relying on `__g123`-style
instance names.

## Stage 3A: map to FPGA LUTs

```python
mapped = eda.fpga_map(generic, lut_size=4)
```

The mapper keeps DFFs as state elements and tries to pack each live
combinational cone feeding the D pins and output ports into LUT4s or smaller
LUTs. It may combine multiple AND/XOR/mux logic levels into one LUT when a
feasible cut has no more than four distinct boundary signals.

A LUT can implement any Boolean function of at most four inputs, not only an
AND or XOR. For each selected cone the mapper finds its boundary nets,
evaluates the cone for all `2^m` assignments (`m` is the actual cut size),
records the truth table as `INIT`, then wires boundary nets to
`I0..I(m-1)` and its result to `O`.

```python
for instance in mapped.instances:
    if instance.cell_type.startswith("LUT"):
        print(instance.cell_type, instance.parameters["INIT"],
              [(c.pin, c.net) for c in instance.connections])
    else:
        print("passed through:", instance.cell_type)
```

Don't expect one LUT per RTL operator. A LUT often absorbs several operators,
and a single RTL operation may require multiple LUTs when its function has
more inputs than `K` or when cuts share/use different intermediate nets.

## Stage 3B: map to standard cells instead

For a standard-cell target, use:

```python
library = eda.parse_liberty("assets/sky130_fd_sc_hd__tt_025C_1v80.lib")
standard_cell_netlist = eda.tech_map(generic, library)
```

This matches each generic cell to an available library cell of the right
family and input count, then attaches area, pins, and timing metadata. It
does not turn the design into LUTs and does not place or route the cells.

## Stage 4: inspect or export

```python
print(eda.to_json(mapped))
print(mapped.dump_graph())  # DOT text; render with Graphviz if installed
eda.write_blif(mapped, "calculator.blif")
```

In BLIF, LUT functions are written as `.names` truth-table rows. Plain rising
edge DFFs become `.latch`; cell types with extra controls can appear as
`.subckt`. BLIF export is an interchange step, not bitstream generation.

## How to get confident with it

1. Run each stage separately and inspect cell counts after each one.
2. Start with a combinational full adder and check the two LUT3 INIT values
   (`96` for XOR3 and `E8` for majority) to understand truth-table ordering.
3. Change `lut_size` from 2 to 6 and compare LUT counts. Larger LUTs can combine
   more logic, but the result is still selected by a bounded heuristic.
4. Compare generic and mapped designs with simulation or an equivalence
   checker. The FPGA tests compare observable combinational nets around
   pass-through state cells.
5. Read the implementation links in [the flow guide](eda-flow.md) when a
   result surprises you. Parser success, synthesis success, and mapping
   success are separate milestones.
