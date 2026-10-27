import itertools
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import labneura.eda as eda

REPO = Path(__file__).resolve().parents[1]
LIBERTY = REPO / "assets" / "sky130_fd_sc_hd__tt_025C_1v80.lib"

COMBINATIONAL = """
module mix (input a, input b, input c, input s, output y, output z, output w);
    assign y = s ? (a ^ b) : ~(a & c);
    assign z = a | b;
    assign w = ~z & c;
endmodule
"""

FULL_ADDER = """
module full_adder (input A, input B, input Cin, output Sum, output Cout);
    assign Sum = A ^ B ^ Cin;
    assign Cout = (A & B) | (B & Cin) | (A & Cin);
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

NODE_RE = re.compile(r'^\s*"((?:[^"\\]|\\.)*)" \[shape=(\w+), label="((?:[^"\\]|\\.)*)"\];$')
EDGE_RE = re.compile(
    r'^\s*"((?:[^"\\]|\\.)*)" -> "((?:[^"\\]|\\.)*)" '
    r'\[label="((?:[^"\\]|\\.)*)", headlabel="((?:[^"\\]|\\.)*)"\];$'
)

GENERIC_FUNCTIONS = {
    "INV": lambda p: 1 - p["A"],
    "BUF": lambda p: p["A"],
    "MUX2": lambda p: p["A1"] if p["S"] else p["A0"],
    "TIEHI": lambda p: 1,
    "TIELO": lambda p: 0,
}


def parse_dot(text):
    nodes, edges = {}, []
    for line in text.splitlines():
        if (node := NODE_RE.match(line)) and "->" not in line:
            nodes[node.group(1)] = (node.group(2), node.group(3))
        elif edge := EDGE_RE.match(line):
            edges.append(edge.groups())
    return nodes, edges


def write(tmp_path, source, name="design.v"):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


def base(net):
    return net.split("[")[0]


def pin_is_output(instance, pin):
    if instance.cell is not None:
        return next(p for p in instance.cell.pins if p.name == pin).direction == "output"
    return pin in {"Y", "Q", "X", "Z", "ZN", "QN", "Q_N", "HI", "LO", "SUM", "COUT"}


@pytest.fixture(scope="module")
def library():
    return eda.parse_liberty(str(LIBERTY))


def build(tmp_path, source, mapped, library):
    generic = eda.generic_map(eda.parse_verilog(write(tmp_path, source)))
    return eda.tech_map(generic, library) if mapped else generic


@pytest.fixture(params=["generic", "mapped"])
def mode(request):
    return request.param == "mapped"


# ---------------------------------------------------------------- structure


@pytest.mark.parametrize("source", [COMBINATIONAL, FULL_ADDER, COUNTER])
def test_dot_is_well_formed_and_complete(tmp_path, library, mode, source):
    module = build(tmp_path, source, mode, library)
    dot = module.dump_graph()
    assert dot.startswith(f'digraph "{module.name}" {{')
    assert dot.rstrip().endswith("}")
    nodes, edges = parse_dot(dot)

    # Every line is either a node, an edge, or fixed boilerplate.
    for line in dot.splitlines()[2:-1]:
        assert NODE_RE.match(line) or EDGE_RE.match(line), line

    # One node per port and per instance, and nothing else.
    expected = {f"port:{p.name}" for p in module.ports}
    expected |= {i.name for i in module.instances}
    assert set(nodes) == expected
    assert len(nodes) == len(module.ports) + len(module.instances)

    for port in module.ports:
        shape = "invhouse" if port.direction == eda.PortDirection.INPUT else "house"
        assert nodes[f"port:{port.name}"][0] == shape
    for instance in module.instances:
        shape, label = nodes[instance.name]
        assert shape == "box"
        assert label == f"{instance.name}\\n{instance.cell_type}"

    # Every edge endpoint exists.
    for source_node, target, _, _ in edges:
        assert source_node in nodes and target in nodes


@pytest.mark.parametrize("source", [COMBINATIONAL, FULL_ADDER, COUNTER])
def test_dot_edges_match_netlist_connectivity(tmp_path, library, mode, source):
    module = build(tmp_path, source, mode, library)
    _, edges = parse_dot(module.dump_graph())
    inputs = {p.name for p in module.ports if p.direction == eda.PortDirection.INPUT}
    outputs = {p.name for p in module.ports if p.direction != eda.PortDirection.INPUT}

    drivers, sinks, seen = {}, {}, set()
    for instance in module.instances:
        for connection in instance.connections:
            net = connection.net
            side = drivers if pin_is_output(instance, connection.pin) else sinks
            side.setdefault(net, []).append((instance.name, connection.pin))
            if net not in seen:
                seen.add(net)
                if base(net) in inputs:
                    drivers.setdefault(net, []).append((f"port:{base(net)}", ""))
                if base(net) in outputs:
                    sinks.setdefault(net, []).append((f"port:{base(net)}", ""))

    expected = sorted(
        (src, dst, net, pin)
        for net, sources in drivers.items()
        for src, _ in sources
        for dst, pin in sinks.get(net, [])
    )
    assert sorted(edges) == expected


@pytest.mark.parametrize("source", [COMBINATIONAL, FULL_ADDER, COUNTER])
def test_every_input_pin_and_output_port_is_connected(tmp_path, library, mode, source):
    module = build(tmp_path, source, mode, library)
    nodes, edges = parse_dot(module.dump_graph())
    incoming = {}
    for _, dst, _, pin in edges:
        incoming.setdefault(dst, []).append(pin)
    outgoing = {src for src, _, _, _ in edges}

    for instance in module.instances:
        wanted = sorted(c.pin for c in instance.connections
                        if not pin_is_output(instance, c.pin))
        assert set(wanted) <= set(incoming.get(instance.name, [])), instance.name
        outputs = [c for c in instance.connections if pin_is_output(instance, c.pin)]
        if outputs:
            assert instance.name in outgoing, f"{instance.name} drives nothing"
    for port in module.ports:
        if port.direction == eda.PortDirection.OUTPUT:
            assert incoming.get(f"port:{port.name}"), f"{port.name} has no driver"


def test_multibit_output_port_receives_one_edge_per_bit(tmp_path, mode, library):
    module = build(tmp_path, COUNTER, mode, library)
    _, edges = parse_dot(module.dump_graph())
    into_count = [e for e in edges if e[1] == "port:count"]
    assert sorted(e[2] for e in into_count) == [f"count[{i}]" for i in range(4)]
    assert len({e[0] for e in into_count}) == 4


def test_bits_never_cross_connect(tmp_path, mode, library):
    module = build(tmp_path, COUNTER, mode, library)
    nodes, edges = parse_dot(module.dump_graph())
    instances = {i.name: i for i in module.instances}
    for src, dst, net, pin in edges:
        if src in instances:
            assert net in {c.net for c in instances[src].connections
                           if pin_is_output(instances[src], c.pin)}, (src, net)
        if dst in instances:
            assert (pin, net) in {(c.pin, c.net) for c in instances[dst].connections}


def test_flops_in_the_counter_graph_are_clocked_and_feed_back(tmp_path, mode, library):
    module = build(tmp_path, COUNTER, mode, library)
    nodes, edges = parse_dot(module.dump_graph())
    flops = [n for n, (_, label) in nodes.items()
             if n.startswith("__") and ("DFF" in label or "df" in label.split("\\n")[1])]
    assert len(flops) == 4
    clocked = {dst for src, dst, _, _ in edges if src == "port:clk"}
    assert set(flops) <= clocked
    # The incrementer reads the registered value, so flops also have fan-out to logic.
    for flop in flops:
        assert any(src == flop and dst != "port:count" for src, dst, _, _ in edges), flop


def test_graph_has_no_combinational_loops(tmp_path, mode, library):
    module = build(tmp_path, COMBINATIONAL, mode, library)
    nodes, edges = parse_dot(module.dump_graph())
    pending = {n: 0 for n in nodes}
    for _, dst, _, _ in edges:
        pending[dst] += 1
    ready = [n for n, count in pending.items() if count == 0]
    seen = 0
    adjacency = {}
    for src, dst, _, _ in edges:
        adjacency.setdefault(src, []).append(dst)
    while ready:
        node = ready.pop()
        seen += 1
        for nxt in adjacency.get(node, []):
            pending[nxt] -= 1
            if pending[nxt] == 0:
                ready.append(nxt)
    assert seen == len(nodes)


# -------------------------------------------------------------- file output


def test_dump_graph_writes_file_and_matches_string(tmp_path, mode, library):
    module = build(tmp_path, FULL_ADDER, mode, library)
    target = tmp_path / "out.dot"
    assert module.dump_graph(str(target)) == ""
    assert target.read_text() == module.dump_graph()


def test_dump_graph_is_deterministic(tmp_path, mode, library):
    module = build(tmp_path, COMBINATIONAL, mode, library)
    assert module.dump_graph() == module.dump_graph()


def test_dump_graph_unwritable_path_raises(tmp_path):
    module = eda.generic_map(eda.parse_verilog(write(tmp_path, FULL_ADDER)))
    with pytest.raises(RuntimeError):
        module.dump_graph(str(tmp_path / "missing_dir" / "x.dot"))


@pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz not installed")
@pytest.mark.parametrize("source", [COMBINATIONAL, FULL_ADDER, COUNTER])
def test_graphviz_accepts_the_output(tmp_path, library, mode, source):
    module = build(tmp_path, source, mode, library)
    target = tmp_path / "g.dot"
    module.dump_graph(str(target))
    result = subprocess.run(["dot", "-Tsvg", str(target)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "<svg" in result.stdout


# ------------------------------------------------------------------- logic


def liberty_function(expression):
    return (expression.replace("!", " not ").replace("&", " and ")
            .replace("|", " or ").replace("^", " ^ "))


def cell_output(instance, values):
    """Evaluate a cell's single output from a {pin: 0/1} dict."""
    if instance.cell is None:
        wide = re.fullmatch(r"(AND|OR|XOR|XNOR|MAJ)(\d+)", instance.cell_type)
        if wide:
            bits = [values[k] for k in sorted(values)]
            assert len(bits) == int(wide.group(2))
            kind = wide.group(1)
            if kind == "AND":
                return int(all(bits))
            if kind == "OR":
                return int(any(bits))
            if kind == "MAJ":
                return int(sum(bits) * 2 > len(bits))
            parity = sum(bits) & 1
            return parity if kind == "XOR" else 1 - parity
        return GENERIC_FUNCTIONS[instance.cell_type](values)
    outputs = [p for p in instance.cell.pins if p.direction == "output"]
    assert len(outputs) == 1
    scope = {k: bool(v) for k, v in values.items()}
    return int(bool(eval(liberty_function(outputs[0].function), {}, scope)))


def evaluate_graph(dot, module, inputs):
    """Evaluate a purely combinational design using only the DOT text."""
    nodes, edges = parse_dot(dot)
    instances = {i.name: i for i in module.instances}
    value = {f"port:{n}": v for n, v in inputs.items()}
    incoming = {}
    for src, dst, _, pin in edges:
        incoming.setdefault(dst, []).append((src, pin))
    remaining = set(instances)
    while remaining:
        progressed = False
        for name in sorted(remaining):
            sources = incoming.get(name, [])
            if all(src in value for src, _ in sources):
                value[name] = cell_output(instances[name], {p: value[s] for s, p in sources})
                remaining.discard(name)
                progressed = True
        assert progressed, "graph is not evaluable"
    return {
        p.name: value[incoming[f"port:{p.name}"][0][0]]
        for p in module.ports if p.direction == eda.PortDirection.OUTPUT
    }


def test_full_adder_graph_computes_addition(tmp_path, mode, library):
    module = build(tmp_path, FULL_ADDER, mode, library)
    dot = module.dump_graph()
    for a, b, cin in itertools.product((0, 1), repeat=3):
        out = evaluate_graph(dot, module, {"A": a, "B": b, "Cin": cin})
        total = a + b + cin
        assert out == {"Sum": total & 1, "Cout": total >> 1}, (a, b, cin)


def test_combinational_graph_matches_reference_model(tmp_path, mode, library):
    module = build(tmp_path, COMBINATIONAL, mode, library)
    dot = module.dump_graph()
    for a, b, c, s in itertools.product((0, 1), repeat=4):
        out = evaluate_graph(dot, module, {"a": a, "b": b, "c": c, "s": s})
        y = (a ^ b) if s else 1 - (a & c)
        z = a | b
        assert out == {"y": y, "z": z, "w": (1 - z) & c}, (a, b, c, s)


def test_graph_logic_agrees_with_iverilog(tmp_path, mode, library):
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("iverilog not installed")
    module = build(tmp_path, COMBINATIONAL, mode, library)
    dot = module.dump_graph()
    source = write(tmp_path, COMBINATIONAL, "mix.v")
    bench = tmp_path / "tb.v"
    bench.write_text(
        "module tb; reg a,b,c,s; wire y,z,w; integer i;\n"
        "mix dut(a,b,c,s,y,z,w);\n"
        "initial begin for (i=0;i<16;i=i+1) begin {a,b,c,s}=i; #1;\n"
        '$display("%b%b%b%b %b%b%b",a,b,c,s,y,z,w); end $finish; end endmodule\n'
    )
    exe = tmp_path / "sim"
    subprocess.run([iverilog, "-o", str(exe), source, str(bench)], check=True)
    lines = subprocess.run([vvp, str(exe)], capture_output=True, text=True,
                           check=True).stdout.split("\n")
    checked = 0
    for line in lines:
        if not re.fullmatch(r"[01]{4} [01]{3}", line):
            continue
        bits, result = line.split()
        a, b, c, s = (int(x) for x in bits)
        out = evaluate_graph(dot, module, {"a": a, "b": b, "c": c, "s": s})
        assert f"{out['y']}{out['z']}{out['w']}" == result, line
        checked += 1
    assert checked == 16


def test_counter_graph_alone_computes_next_state(tmp_path, mode, library):
    """Cut the flops in the DOT: edges into D are the next state, edges out of Q the state."""
    module = build(tmp_path, COUNTER, mode, library)
    nodes, edges = parse_dot(module.dump_graph())
    instances = {i.name: i for i in module.instances}
    flops = {n: i for n, i in instances.items()
             if any(c.pin == "D" for c in i.connections)}
    assert len(flops) == 4
    # Flop -> state bit index, read from the labels of the edges that leave the flop.
    bit_of = {}
    for src, dst, net, _ in edges:
        if src in flops and dst == "port:count":
            bit_of[src] = int(re.fullmatch(r"count\[(\d)\]", net).group(1))
    assert sorted(bit_of.values()) == [0, 1, 2, 3]

    incoming = {}
    for src, dst, net, pin in edges:
        incoming.setdefault(dst, []).append((src, net, pin))

    def next_state(count, rst):
        net_value = {f"count[{i}]": (count >> i) & 1 for i in range(4)}
        net_value["rst"] = rst
        logic = [n for n in instances if n not in flops]
        pending = list(logic)
        while pending:
            progressed = False
            for name in list(pending):
                feeds = incoming.get(name, [])
                if not all(net in net_value for _, net, _ in feeds):
                    continue
                inst = instances[name]
                out_net = next(c.net for c in inst.connections
                               if pin_is_output(inst, c.pin))
                net_value[out_net] = cell_output(inst, {p: net_value[n] for _, n, p in feeds})
                pending.remove(name)
                progressed = True
            assert progressed, "next-state logic is not evaluable from the graph"
        result = 0
        for flop, bit in bit_of.items():
            (_, d_net, _), = [e for e in incoming[flop] if e[2] == "D"]
            result |= net_value[d_net] << bit
        return result

    for count in range(16):
        assert next_state(count, 0) == (count + 1) & 0xF
        assert next_state(count, 1) == 0
