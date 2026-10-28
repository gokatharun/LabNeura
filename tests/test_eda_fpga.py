import random
import re
import zlib
from collections import Counter
from pathlib import Path

import pytest

import labneura.eda as eda

REPO = Path(__file__).resolve().parents[1]
IBEX = REPO / "examples" / "verilog" / "ibex"

FULL_ADDER = """
module full_adder (input A, input B, input Cin, output Sum, output Cout);
    assign Sum = A ^ B ^ Cin;
    assign Cout = (A & B) | (B & Cin) | (A & Cin);
endmodule
"""

WIDE = """
module wide (input a, input b, input c, input d, input e, input f, input g, input h,
             output y, output z, output w);
    assign y = a & b & c & d & e & f & g & h;
    assign z = (a | b | c) ^ (d & e) ^ (f | g | h);
    assign w = ~(a ^ b ^ c ^ d ^ e ^ f);
endmodule
"""

COUNTER = """
module counter_4bit (input wire clk, input wire rst, output reg [3:0] count);
    always @(posedge clk) begin
        if (rst) count <= 4'b0000;
        else count <= count + 1'b1;
    end
endmodule
"""

CONSTANTS = """
module consts (input a, output y, output z, output w);
    assign y = 1'b1;
    assign z = a;
    assign w = a & 1'b0;
endmodule
"""

SEQUENTIAL_SET = """
module regs (input clk, input rst_n, input [3:0] d, output reg [3:0] q, output p);
    always @(posedge clk or negedge rst_n)
        if (!rst_n) q <= 4'b1010; else q <= d ^ {q[2:0], q[3]};
    assign p = ^q;
endmodule
"""

SEQUENTIAL_CELLS = {"DFF", "DFFR", "DFFS", "DFFSR", "LATCH", "TRIBUF", "TIEHI", "TIELO"}


def write(tmp_path, source, name="design.v"):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


def generic_of(tmp_path, source):
    return eda.generic_map(eda.parse_verilog(write(tmp_path, source)))


def is_output(pin):
    return pin in {"Y", "Q", "X", "Z", "ZN", "QN", "Q_N", "O"}


def base(net):
    return net.split("[")[0]


def gate_value(cell_type, init, values):
    """values: dict pin -> 0/1 for one combinational generic cell or LUT."""
    wide = re.fullmatch(r"(AND|OR|XOR|NAND|NOR|XNOR|MAJ)(\d*)", cell_type)
    if wide:
        bits = [values[pin] for pin in sorted(values)]
        kind = wide.group(1)
        total = sum(bits)
        return {
            "AND": int(all(bits)), "OR": int(any(bits)), "XOR": total & 1,
            "NAND": int(not all(bits)), "NOR": int(not any(bits)),
            "XNOR": 1 - (total & 1), "MAJ": int(total * 2 > len(bits)),
        }[kind]
    if cell_type in {"INV", "NOT"}:
        return 1 - values["A"]
    if cell_type == "BUF":
        return values["A"]
    if cell_type == "MUX2":
        return values["A1"] if values["S"] else values["A0"]
    lut = re.fullmatch(r"LUT(\d)", cell_type)
    if lut:
        width = int(lut.group(1))
        row = sum(values[f"I{i}"] << i for i in range(width))
        return (int(init, 16) >> row) & 1
    raise AssertionError(f"unsupported cell {cell_type}")


COMBINATIONAL = re.compile(r"^(AND|OR|XOR|NAND|NOR|XNOR|MAJ)\d*$|^(INV|NOT|BUF|MUX2|LUT\d)$")


def observable_values(module, seed):
    """Value of every observable net, given deterministic values on free nets.

    Free nets are primary inputs and the outputs of flops, latches and tristates.
    Observable nets are output-port bits and the inputs of those sequential cells.
    """
    drivers, sequential = {}, []
    for instance in module.instances:
        if instance.cell_type in {"TIEHI", "TIELO"}:
            drivers[next(c.net for c in instance.connections if is_output(c.pin))] = (
                instance, None)
        elif COMBINATIONAL.match(instance.cell_type):
            out = next(c.net for c in instance.connections if is_output(c.pin))
            drivers[out] = (instance, {c.pin: c.net for c in instance.connections
                                       if not is_output(c.pin)})
        else:
            sequential.append(instance)

    memo = {}

    def free(net):
        return zlib.crc32(f"{seed}:{net}".encode()) & 1

    def value(net):
        stack = [net]
        while stack:
            current = stack[-1]
            if current in memo:
                stack.pop()
                continue
            if current not in drivers:
                memo[current] = free(current)
                stack.pop()
                continue
            instance, pins = drivers[current]
            if pins is None:
                memo[current] = 1 if instance.cell_type == "TIEHI" else 0
                stack.pop()
                continue
            missing = [n for n in pins.values() if n not in memo]
            if missing:
                stack.extend(missing)
                continue
            memo[current] = gate_value(
                instance.cell_type, instance.parameters.get("INIT"),
                {pin: memo[n] for pin, n in pins.items()})
            stack.pop()
        return memo[net]

    observed = {}
    outputs = {p.name for p in module.ports if p.direction != eda.PortDirection.INPUT}
    nets = set()
    for instance in module.instances:
        for c in instance.connections:
            if base(c.net) in outputs:
                nets.add(c.net)
    for instance in sequential:
        for c in instance.connections:
            if not is_output(c.pin):
                nets.add(c.net)
    for net in nets:
        observed[net] = value(net)
    return observed


def assert_equivalent(generic, mapped, vectors=64):
    for seed in range(vectors):
        expected = observable_values(generic, seed)
        actual = observable_values(mapped, seed)
        assert actual.keys() == expected.keys()
        diff = [n for n in expected if expected[n] != actual[n]]
        assert not diff, f"seed {seed}: {diff[:5]}"


def counts(module):
    return Counter(i.cell_type for i in module.instances)


# ------------------------------------------------------------------ mapping


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("source", [FULL_ADDER, WIDE, COUNTER, CONSTANTS, SEQUENTIAL_SET])
def test_lut_netlist_is_equivalent_to_generic_netlist(tmp_path, source, k):
    generic = generic_of(tmp_path, source)
    mapped = eda.fpga_map(generic, k)
    assert_equivalent(generic, mapped, vectors=64)


@pytest.mark.parametrize("k", [3, 4, 6])
def test_luts_respect_the_requested_size(tmp_path, k):
    mapped = eda.fpga_map(generic_of(tmp_path, WIDE), k)
    for instance in mapped.instances:
        lut = re.fullmatch(r"LUT(\d)", instance.cell_type)
        if not lut:
            continue
        width = int(lut.group(1))
        assert 1 <= width <= k
        pins = [c.pin for c in instance.connections]
        assert pins == [f"I{i}" for i in range(width)] + ["O"]
        assert len(instance.parameters["INIT"]) == max(1, (1 << width) // 4)
    assert not any(re.fullmatch(r"(AND|OR|XOR|INV|MUX)\d*", t) for t in counts(mapped))


def test_full_adder_maps_to_two_lut3_with_known_truth_tables(tmp_path):
    mapped = eda.fpga_map(generic_of(tmp_path, FULL_ADDER), 3)
    assert counts(mapped) == {"LUT3": 2}
    init = {next(c.net for c in i.connections if c.pin == "O"): i.parameters["INIT"]
            for i in mapped.instances}
    assert init == {"Sum": "96", "Cout": "E8"}


def test_wide_gate_uses_fewer_luts_with_larger_k(tmp_path):
    generic = generic_of(tmp_path, WIDE)
    small = sum(v for t, v in counts(eda.fpga_map(generic, 2)).items() if t.startswith("LUT"))
    medium = sum(v for t, v in counts(eda.fpga_map(generic, 4)).items() if t.startswith("LUT"))
    large = sum(v for t, v in counts(eda.fpga_map(generic, 6)).items() if t.startswith("LUT"))
    assert large <= medium <= small
    assert large < len(generic.instances)


def test_flops_pass_through_and_logic_collapses(tmp_path):
    generic = generic_of(tmp_path, COUNTER)
    mapped = eda.fpga_map(generic, 4)
    assert counts(mapped)["DFF"] == 4
    luts = sum(v for t, v in counts(mapped).items() if t.startswith("LUT"))
    assert luts < len(generic.instances) - 4
    flops = [i for i in mapped.instances if i.cell_type == "DFF"]
    assert all({c.pin for c in f.connections} == {"D", "CLK", "Y"} for f in flops)


def test_async_flops_keep_their_cell_type(tmp_path):
    generic = generic_of(tmp_path, SEQUENTIAL_SET)
    mapped = eda.fpga_map(generic, 4)
    keep = lambda m: {t: v for t, v in counts(m).items() if t.startswith("DFF")}
    assert keep(generic) and keep(mapped) == keep(generic)


def test_equivalence_checker_detects_a_wrong_truth_table(tmp_path, monkeypatch):
    generic = generic_of(tmp_path, FULL_ADDER)
    mapped = eda.fpga_map(generic, 3)
    original = gate_value
    monkeypatch.setitem(
        globals(), "gate_value",
        lambda t, init, v: original(t, "97" if init == "96" else init, v))
    with pytest.raises(AssertionError):
        assert_equivalent(generic, mapped)


def test_constants_and_wires_survive(tmp_path):
    generic = generic_of(tmp_path, CONSTANTS)
    mapped = eda.fpga_map(generic, 4)
    assert_equivalent(generic, mapped)


def test_input_netlist_is_not_modified(tmp_path):
    generic = generic_of(tmp_path, WIDE)
    before = [(i.name, i.cell_type, [(c.pin, c.net) for c in i.connections])
              for i in generic.instances]
    eda.fpga_map(generic, 4)
    after = [(i.name, i.cell_type, [(c.pin, c.net) for c in i.connections])
             for i in generic.instances]
    assert before == after


def test_mapping_is_deterministic(tmp_path):
    generic = generic_of(tmp_path, WIDE)
    one = eda.to_blif(eda.fpga_map(generic, 4))
    two = eda.to_blif(eda.fpga_map(generic, 4))
    assert one == two


def test_dump_graph_works_on_a_lut_netlist(tmp_path):
    mapped = eda.fpga_map(generic_of(tmp_path, FULL_ADDER), 3)
    dot = mapped.dump_graph()
    assert "LUT3" in dot and dot.count("->") >= 6


# ------------------------------------------------------------------- errors


@pytest.mark.parametrize("size", [0, 1, 7])
def test_invalid_lut_size_is_rejected(tmp_path, size):
    generic = generic_of(tmp_path, FULL_ADDER)
    with pytest.raises(Exception, match="lut_size"):
        eda.fpga_map(generic, size)


def test_unmapped_behavioral_module_is_rejected(tmp_path):
    parsed = eda.parse_verilog(write(tmp_path, COUNTER))
    assert parsed.is_behavioral
    with pytest.raises(Exception, match="generic_map"):
        eda.fpga_map(parsed, 4)


# --------------------------------------------------------------------- BLIF


def parse_blif(text):
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines[0].startswith(".model ") and lines[-1] == ".end"
    model = {"inputs": [], "outputs": [], "names": [], "latch": [], "subckt": []}
    index = 1
    while index < len(lines) - 1:
        tokens = lines[index].split()
        keyword = tokens[0]
        if keyword == ".inputs":
            model["inputs"] = tokens[1:]
        elif keyword == ".outputs":
            model["outputs"] = tokens[1:]
        elif keyword == ".names":
            rows = []
            while index + 1 < len(lines) - 1 and not lines[index + 1].startswith("."):
                index += 1
                rows.append(lines[index].split())
            model["names"].append((tokens[1:-1], tokens[-1], rows))
        elif keyword == ".latch":
            model["latch"].append(tokens[1:])
        elif keyword == ".subckt":
            model["subckt"].append(tokens[1:])
        else:
            raise AssertionError(lines[index])
        index += 1
    return model


@pytest.mark.parametrize("source", [FULL_ADDER, WIDE, COUNTER, CONSTANTS])
def test_blif_describes_the_same_luts(tmp_path, source):
    mapped = eda.fpga_map(generic_of(tmp_path, source), 4)
    model = parse_blif(eda.to_blif(mapped))

    cover = {}
    for inputs, output, rows in model["names"]:
        table = 0
        for row in rows:
            if len(row) == 1:
                table = 1
                continue
            assert row[1] == "1" and len(row[0]) == len(inputs)
            table |= 1 << sum(int(bit) << i for i, bit in enumerate(row[0]))
        cover[output] = (inputs, table)

    for instance in mapped.instances:
        if instance.cell_type.startswith("LUT"):
            output = next(c.net for c in instance.connections if c.pin == "O")
            inputs = [c.net for c in instance.connections if c.pin != "O"]
            assert cover[output][0] == inputs
            assert cover[output][1] == int(instance.parameters["INIT"], 16)
    assert len([i for i in mapped.instances if i.cell_type.startswith("LUT")]) <= len(cover)


def test_blif_ports_flops_and_file_output(tmp_path):
    mapped = eda.fpga_map(generic_of(tmp_path, COUNTER), 4)
    text = eda.to_blif(mapped)
    model = parse_blif(text)
    assert model["inputs"] == ["clk", "rst"]
    assert model["outputs"] == [f"count[{i}]" for i in range(4)]
    assert len(model["latch"]) == 4
    assert all(latch[2] == "re" and latch[3] == "clk" for latch in model["latch"])
    path = tmp_path / "out.blif"
    eda.write_blif(mapped, str(path))
    assert path.read_text() == text
    with pytest.raises(Exception):
        eda.write_blif(mapped, str(tmp_path / "missing" / "x.blif"))


def test_blif_keeps_async_flops_as_subckt(tmp_path):
    mapped = eda.fpga_map(generic_of(tmp_path, SEQUENTIAL_SET), 4)
    model = parse_blif(eda.to_blif(mapped))
    kinds = {entry[0] for entry in model["subckt"]}
    assert kinds & {"DFFR", "DFFS", "DFFSR"}


# ---------------------------------------------------------------- projects


@pytest.mark.skipif(not IBEX.exists(), reason="ibex example not present")
def test_ibex_alu_maps_and_matches_generic():
    project = eda.parse_verilog_project(str(IBEX))
    generic = eda.generic_map(project, "ibex_alu")
    for k in (4, 6):
        mapped = eda.fpga_map(generic, k)
        luts = sum(v for t, v in counts(mapped).items() if t.startswith("LUT"))
        assert 0 < luts < len(generic.instances)
        assert_equivalent(generic, mapped, vectors=12)


@pytest.mark.skipif(not IBEX.exists(), reason="ibex example not present")
def test_project_overload_with_top_and_lut_size():
    project = eda.parse_verilog_project(str(IBEX))
    direct = eda.fpga_map(project, top="ibex_counter", lut_size=5)
    staged = eda.fpga_map(eda.generic_map(project, "ibex_counter"), 5)
    assert eda.to_blif(direct) == eda.to_blif(staged)


@pytest.mark.skipif(not IBEX.exists(), reason="ibex example not present")
def test_ibex_core_maps_and_matches_generic():
    project = eda.parse_verilog_project(str(IBEX))
    generic = eda.generic_map(project, "ibex_core")
    mapped = eda.fpga_map(generic, 6)
    seq = sum(v for t, v in counts(generic).items() if t.startswith("DFF"))
    assert sum(v for t, v in counts(mapped).items() if t.startswith("DFF")) == seq
    assert len(mapped.instances) < len(generic.instances) // 2
    assert_equivalent(generic, mapped, vectors=3)


# ----------------------------------------------------------- cell attributes


def test_every_fpga_instance_carries_cell_data(tmp_path):
    mapped = eda.fpga_map(generic_of(tmp_path, SEQUENTIAL_SET), 4)
    library = eda.default_fpga_library()
    names = {c.name for c in library.cells}
    for instance in mapped.instances:
        assert instance.cell is not None, instance.name
        assert instance.cell.name == instance.cell_type in names
        assert instance.cell.area is not None


def test_lut_cell_pins_area_and_timing(tmp_path):
    mapped = eda.fpga_map(generic_of(tmp_path, FULL_ADDER), 3)
    for instance in mapped.instances:
        cell = instance.cell
        assert cell.area == 8.0  # 2**3 configuration bits
        assert [p.name for p in cell.pins] == ["I0", "I1", "I2", "O"]
        assert [p.direction for p in cell.pins] == ["input"] * 3 + ["output"]
        # Pins on the instance match the cell definition exactly.
        assert {c.pin for c in instance.connections} == {p.name for p in cell.pins}
        output = cell.pins[-1]
        assert {a.related_pin for a in output.timing_arcs} == {"I0", "I1", "I2"}
        assert all(a.cell_rise.values[0][0] > 0 for a in output.timing_arcs)
        assert all(p.capacitance > 0 for p in cell.pins[:3])


def test_larger_luts_cost_more_area_and_delay():
    cells = {c.name: c for c in eda.default_fpga_library().cells}
    areas = [cells[f"LUT{k}"].area for k in range(2, 7)]
    delays = [cells[f"LUT{k}"].pins[-1].timing_arcs[0].cell_rise.values[0][0]
              for k in range(2, 7)]
    assert areas == sorted(areas) and len(set(areas)) == 5
    assert delays == sorted(delays) and len(set(delays)) == 5


def test_flop_cells_have_setup_and_clock_to_q_arcs(tmp_path):
    mapped = eda.fpga_map(generic_of(tmp_path, COUNTER), 4)
    flop = next(i for i in mapped.instances if i.cell_type == "DFF")
    pins = {p.name: p for p in flop.cell.pins}
    assert pins["D"].timing_arcs[0].timing_type == "setup_rising"
    assert pins["D"].timing_arcs[0].rise_constraint.values[0][0] > 0
    assert pins["Y"].timing_arcs[0].timing_type == "rising_edge"
    assert pins["Y"].timing_arcs[0].related_pin == "CLK"


def test_total_area_and_critical_path_from_instance_cells(tmp_path):
    mapped = eda.fpga_map(generic_of(tmp_path, COUNTER), 4)
    area = sum(i.cell.area for i in mapped.instances)
    lut_bits = sum(2 ** int(i.cell_type[3]) for i in mapped.instances
                   if i.cell_type.startswith("LUT"))
    assert area == lut_bits + 4 * 16.0

    # Longest register-to-register path, using only the data on the instances.
    arrival = {}
    drivers = {}
    for i in mapped.instances:
        out = next(c.net for c in i.connections if c.pin in {"O", "Y"})
        drivers[out] = i

    def at(net):
        if net in arrival:
            return arrival[net]
        inst = drivers.get(net)
        if inst is None:
            arrival[net] = 0.0
        elif inst.cell_type == "DFF":
            arrival[net] = 0.35
        else:
            arrival[net] = max(
                at(c.net) + next(a.cell_rise.values[0][0]
                                 for a in inst.cell.pins[-1].timing_arcs
                                 if a.related_pin == c.pin)
                for c in inst.connections if c.pin != "O")
        return arrival[net]

    worst = max(at(c.net) for i in mapped.instances if i.cell_type == "DFF"
                for c in i.connections if c.pin == "D")
    # Bit 3 depends on five signals, so LUT4s need two levels: clk->Q + LUT4 + LUT2.
    assert worst == pytest.approx(0.35 + 0.24 + 0.12)


def test_six_input_luts_flatten_the_counter_to_one_level(tmp_path):
    mapped = eda.fpga_map(generic_of(tmp_path, COUNTER), 6)
    luts = [i for i in mapped.instances if i.cell_type.startswith("LUT")]
    assert len(luts) == 4 and max(int(i.cell_type[3]) for i in luts) == 5


def test_dot_uses_cell_pin_directions_on_fpga_netlist(tmp_path):
    dot = eda.fpga_map(generic_of(tmp_path, COUNTER), 4).dump_graph()
    assert '"port:count"' in dot and "DFF" in dot and "LUT" in dot


LIBERTY_LUTS = """
library (my_fpga) {
  time_unit : "1ns";
  capacitive_load_unit (1, pf);
  cell (LUT3) { area : 99; pin (I0) { direction : input; } pin (I1) { direction : input; }
    pin (I2) { direction : input; } pin (O) { direction : output; } }
  cell (LUT2) { area : 50; pin (I0) { direction : input; } pin (I1) { direction : input; }
    pin (O) { direction : output; } }
}
"""


def test_custom_library_overrides_cell_data_and_missing_cells_stay_unset(tmp_path):
    library = eda.parse_liberty_source(LIBERTY_LUTS)
    generic = generic_of(tmp_path, SEQUENTIAL_SET)
    mapped = eda.fpga_map(generic, 3, library)
    seen = set()
    for instance in mapped.instances:
        if instance.cell_type in {"LUT2", "LUT3"}:
            assert instance.cell.area == {"LUT2": 50, "LUT3": 99}[instance.cell_type]
            seen.add(instance.cell_type)
        elif instance.cell_type == "DFFR" or instance.cell_type.startswith("LUT"):
            assert instance.cell is None
    assert seen


def test_project_overload_accepts_a_library():
    library = eda.parse_liberty_source(LIBERTY_LUTS)
    project = eda.parse_verilog_project(str(IBEX))
    mapped = eda.fpga_map(project, top="ibex_counter", lut_size=3, library=library)
    assert any(i.cell is not None and i.cell.area == 99 for i in mapped.instances)


def test_blif_and_equivalence_unchanged_with_cell_data(tmp_path):
    generic = generic_of(tmp_path, WIDE)
    mapped = eda.fpga_map(generic, 4)
    assert_equivalent(generic, mapped)
    assert "LUT" not in eda.to_blif(mapped)


def test_ice40_hx_library_gives_real_timing_and_unit_area(tmp_path):
    library = eda.parse_liberty(str(REPO / "assets" / "ice40_hx.lib"))
    mapped = eda.fpga_map(generic_of(tmp_path, COUNTER), 4, library)
    assert all(i.cell is not None for i in mapped.instances)
    assert sum(i.cell.area for i in mapped.instances) == len(mapped.instances)
    lut4 = next(c for c in library.cells if c.name == "LUT4")
    arcs = {a.related_pin: a.cell_rise.values[0][0] for a in lut4.pins[-1].timing_arcs}
    assert arcs == {"I0": 0.449, "I1": 0.4, "I2": 0.379, "I3": 0.316}


def test_single_input_lut_edges_appear_in_graph(tmp_path):
    two_bit = """
    module c2 (input clk, output reg [1:0] count);
        always @(posedge clk) count <= count + 1'b1;
    endmodule
    """
    mapped = eda.fpga_map(generic_of(tmp_path, two_bit), 4)
    lut1 = next(i for i in mapped.instances if i.cell_type == "LUT1")
    assert lut1.cell is not None and lut1.cell.area == 2.0
    out = next(c.net for c in lut1.connections if c.pin == "O")
    dot = mapped.dump_graph()
    # The LUT1 must drive the flop it feeds, not only read from it.
    assert f'"{lut1.name}" -> ' in dot
    assert f'label="{out}"' in dot
