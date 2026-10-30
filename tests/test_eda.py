import json
from functools import lru_cache
from pathlib import Path

import pytest

import labneura.eda as eda


VERILOG = """
// A small two-gate combinational design.
module logic_example (a, b, c, f);
    input wire a, b, c;
    output wire f;
    wire w;

    and AND1 (w, a, b);
    or OR1 (f, w, c);
endmodule
"""

LIBERTY = """
library (demo_cells) {
  time_unit : "1ns";
  capacitive_load_unit (1, pf);

  lu_table_template (delay_template) {
    variable_1 : input_net_transition;
    variable_2 : total_output_net_capacitance;
    index_1 ("0.1, 0.2");
    index_2 ("1, 2");
  }

  cell (AND2_X1) {
    area : 1.25;
    pin (A) {
      direction : input;
      capacitance : 0.003;
    }
    pin (B) {
      direction : input;
      capacitance : 0.004;
    }
    pin (Y) {
      direction : output;
      function : "A & B";
      timing () {
        related_pin : "A";
        timing_type : combinational;
        timing_sense : positive_unate;
        cell_rise (delay_template) {
          values ("0.11, 0.12", "0.21, 0.22");
        }
        cell_fall (delay_template) {
          values ("0.31, 0.32", "0.41, 0.42");
        }
        rise_constraint (delay_template) {
          values ("0.51, 0.52", "0.61, 0.62");
        }
      }
    }
  }
}
"""

DEFAULT_LIBERTY = (
    Path(__file__).resolve().parents[1]
    / "assets/sky130_fd_sc_hd__tt_025C_1v80.lib"
)


@lru_cache(maxsize=1)
def default_cell_library():
    return eda.parse_liberty(str(DEFAULT_LIBERTY))


def map_verilog(path):
    rtl_module = eda.parse_verilog(str(path))
    generic_module = eda.generic_map(rtl_module)
    return eda.tech_map(generic_module, default_cell_library())


def test_parse_verilog_builds_module_netlist(tmp_path):
    verilog_path = tmp_path / "logic_example.v"
    verilog_path.write_text(VERILOG, encoding="utf-8")
    module = eda.parse_verilog(str(verilog_path))

    assert module.name == "logic_example"
    assert module.assignments == []
    assert module.get_name() == module.name
    assert module.get_ports()[0].get_name() == module.ports[0].name
    assert module.get_nets()[0].get_kind() == module.nets[0].kind
    assert module.get_instances()[0].get_cell_type() == module.instances[0].cell_type
    assert (
        module.instances[0].get_connections()[0].get_pin()
        == module.instances[0].connections[0].pin
    )
    assert [(port.name, port.direction) for port in module.ports] == [
        ("a", eda.PortDirection.INPUT),
        ("b", eda.PortDirection.INPUT),
        ("c", eda.PortDirection.INPUT),
        ("f", eda.PortDirection.OUTPUT),
    ]
    assert [(net.name, net.kind) for net in module.nets] == [
        ("a", eda.NetType.INPUT),
        ("b", eda.NetType.INPUT),
        ("c", eda.NetType.INPUT),
        ("f", eda.NetType.OUTPUT),
        ("w", eda.NetType.INTERNAL),
    ]
    assert [
        (instance.name, instance.cell_type,
         {connection.pin: connection.net for connection in instance.connections})
        for instance in module.instances
    ] == [
        ("AND1", "and", {"Y": "w", "A": "a", "B": "b"}),
        ("OR1", "or", {"Y": "f", "A": "w", "B": "c"}),
    ]
    with pytest.raises(AttributeError):
        module.name = "changed"


def test_parse_verilog_exposes_hierarchy_in_python_module_api(tmp_path):
    verilog_path = tmp_path / "hierarchy.v"
    verilog_path.write_text(
        """
        module half_adder(input wire a, b, output wire s, c);
            assign s = a ^ b;
            assign c = a & b;
        endmodule

        module top_module(input wire a, b, output wire s1, c1, s2, c2);
            half_adder u_ha1(a, b, s1, c1);
            half_adder u_ha2(a, b, s2, c2);
        endmodule
        """,
        encoding="utf-8",
    )

    top = eda.parse_verilog(str(verilog_path))

    assert top.hierarchy_instance_name == ""
    assert top.get_hierarchy_instance_name() == ""
    assert len(top.child_modules) == 2
    assert top.get_child_modules()[0].name == "half_adder"
    assert top.child_modules[0].hierarchy_instance_name == "u_ha1"
    assert top.child_modules[0].get_hierarchy_instance_name() == "u_ha1"
    assert top.child_modules[1].hierarchy_instance_name == "u_ha2"
    assert [assignment.lhs for assignment in top.child_modules[0].assignments] == [
        "s",
        "c",
    ]
    generic = eda.generic_map(top)
    generic = eda.generic_map(top)

    assert [module.hierarchy_instance_name for module in generic.child_modules] == [
        "u_ha1",
        "u_ha2",
    ]
    assert [instance.cell_type for instance in generic.instances] == [
        "half_adder",
        "half_adder",
    ]
    for child in generic.child_modules:
        assert child.assignments == []
        assert {instance.cell_type for instance in child.instances} == {
            "XOR2",
            "AND2",
        }


def test_parse_file_and_write_json(tmp_path):
    verilog_path = tmp_path / "design.v"
    json_path = tmp_path / "design.json"
    verilog_path.write_text(VERILOG, encoding="utf-8")

    module = map_verilog(verilog_path)
    eda.write_json(module, str(json_path))
    serialized = json.loads(json_path.read_text(encoding="utf-8"))

    assert serialized["format"] == "eda-netlist"
    assert serialized["version"] == 1
    assert serialized["modules"][0]["name"] == "logic_example"
    assert json.loads(eda.to_json(module)) == serialized


def test_parse_verilog_resolves_instances_to_bundled_cells(tmp_path):
    verilog_path = tmp_path / "cells.v"
    verilog_path.write_text(VERILOG, encoding="utf-8")

    rtl_module = eda.parse_verilog(str(verilog_path))
    module = eda.tech_map(eda.generic_map(rtl_module), default_cell_library())

    assert [(instance.cell_type, instance.cell.name) for instance in module.instances] == [
        ("sky130_fd_sc_hd__and2_1", "sky130_fd_sc_hd__and2_1"),
        ("sky130_fd_sc_hd__or2_1", "sky130_fd_sc_hd__or2_1"),
    ]
    assert module.instances[0].cell.area == pytest.approx(6.256)
    assert {
        connection.pin: connection.net
        for connection in module.instances[0].connections
    } == {"X": "w", "A": "a", "B": "b"}


def test_parse_half_adder_verilog_resolves_cells():
    verilog_path = Path(__file__).resolve().parents[1] / "examples/verilog/half_adder.v"

    module = map_verilog(verilog_path)
    source_module = eda.parse_verilog(str(verilog_path))

    assert module.name == "half_adder"
    assert [(instance.name, instance.cell.name) for instance in module.instances] == [
        ("sum", "sky130_fd_sc_hd__xor2_1"),
        ("carry", "sky130_fd_sc_hd__and2_1"),
    ]
    assert [
        {connection.pin: connection.net for connection in instance.connections}
        for instance in module.instances
    ] == [
        {"X": "s", "A": "a", "B": "b"},
        {"X": "c", "A": "a", "B": "b"},
    ]
    assert [(instance.cell_type, instance.cell) for instance in source_module.instances] == [
        ("xor", None),
        ("and", None),
    ]
    assert [
        instance.cell_type
        for instance in eda.generic_map(source_module).instances
    ] == ["XOR2", "AND2"]


def test_parse_verilog_reads_ansi_ports_and_assignments():
    verilog_path = Path(__file__).resolve().parents[1] / "examples/verilog/mux.v"

    module = eda.parse_verilog(str(verilog_path))

    assert module.name == "mux_4to1"
    assert [(port.name, port.direction) for port in module.ports] == [
        ("I", eda.PortDirection.INPUT),
        ("S", eda.PortDirection.INPUT),
        ("Y", eda.PortDirection.OUTPUT),
    ]
    assert len(module.assignments) == 1
    assert module.assignments[0].lhs == "Y"
    assert module.assignments[0].rhs == "I[S]"
    assert module.instances == []
    serialized = json.loads(eda.to_json(module))
    serialized_module = serialized["modules"][0]
    assert len(serialized_module["assignments"]) == 1
    assert serialized_module["instances"] == []


def test_staged_mapping_accepts_an_explicit_liberty_library(tmp_path):
    verilog_path = tmp_path / "custom_library.v"
    liberty_path = tmp_path / "custom_library.lib"
    verilog_path.write_text(
        "module top(input wire a, b, output wire y); "
        "assign y = a & b; endmodule",
        encoding="utf-8",
    )
    liberty_path.write_text(LIBERTY, encoding="utf-8")

    rtl_module = eda.parse_verilog(str(verilog_path))
    generic_module = eda.generic_map(rtl_module)
    mapped_module = eda.tech_map(
        generic_module, eda.parse_liberty(str(liberty_path))
    )

    assert len(rtl_module.assignments) == 1
    assert [(instance.cell_type, instance.cell) for instance in generic_module.instances] == [
        ("AND2", None)
    ]
    assert mapped_module.instances[0].cell.name == "AND2_X1"
    assert mapped_module.instances[0].cell.area == pytest.approx(1.25)


def test_tech_map_errors_when_library_lacks_a_generic_cell(tmp_path):
    verilog_path = tmp_path / "missing_generic_cell.v"
    liberty_path = tmp_path / "and_only.lib"
    verilog_path.write_text(
        "module top(input wire a, b, output wire y); "
        "assign y = a ^ b; endmodule",
        encoding="utf-8",
    )
    liberty_path.write_text(LIBERTY, encoding="utf-8")

    generic_module = eda.generic_map(
        eda.parse_verilog(str(verilog_path))
    )
    with pytest.raises(RuntimeError, match="no standard cell matching generic cell 'XOR2'"):
        eda.tech_map(generic_module, eda.parse_liberty(str(liberty_path)))


def test_parse_verilog_lowers_binary_assignment_expression(tmp_path):
    verilog_path = tmp_path / "binary.v"
    verilog_path.write_text(
        "module binary(input wire a, b, output wire y); "
        "assign y = a & b; endmodule",
        encoding="utf-8",
    )

    module = map_verilog(verilog_path)

    assert module.assignments == []
    assert len(module.instances) == 1
    instance = module.instances[0]
    assert instance.cell.name == "sky130_fd_sc_hd__and2_1"
    assert {
        connection.pin: connection.net for connection in instance.connections
    } == {"A": "a", "B": "b", "X": "y"}


def test_parse_verilog_lowers_scalar_not_and_conditional_cells(tmp_path):
    verilog_path = tmp_path / "logic_cells.v"
    verilog_path.write_text(
        "module logic_cells(input wire a, b, s, output wire y, z); "
        "assign y = ~a; assign z = s ? a : b; endmodule",
        encoding="utf-8",
    )

    module = map_verilog(verilog_path)

    assert module.assignments == []
    assert [(instance.cell.name, {
        connection.pin: connection.net for connection in instance.connections
    }) for instance in module.instances] == [
        ("sky130_fd_sc_hd__inv_1", {"A": "a", "Y": "y"}),
        (
            "sky130_fd_sc_hd__mux2_1",
            {"A0": "b", "A1": "a", "S": "s", "X": "z"},
        ),
    ]


def test_parse_verilog_lowers_direct_scalar_assignment_to_buffer(tmp_path):
    verilog_path = tmp_path / "buffer.v"
    verilog_path.write_text(
        "module buffer_example(input wire a, output wire y); "
        "assign y = a; endmodule",
        encoding="utf-8",
    )

    module = map_verilog(verilog_path)

    assert module.assignments == []
    assert module.instances[0].cell.name == "sky130_fd_sc_hd__buf_1"
    assert {
        connection.pin: connection.net
        for connection in module.instances[0].connections
    } == {"A": "a", "X": "y"}


@pytest.mark.parametrize(
    ("literal", "expected_cell"),
    [
        ("1'b0", "LO"),
        ("1'b1", "HI"),
        ("0", "LO"),
        ("1", "HI"),
        ("2", "LO"),
    ],
)
def test_parse_verilog_lowers_scalar_constants_to_tie_cells(
    tmp_path, literal, expected_cell
):
    verilog_path = tmp_path / "tie.v"
    verilog_path.write_text(
        f"module tie_example(output wire y); assign y = {literal}; endmodule",
        encoding="utf-8",
    )

    module = map_verilog(verilog_path)

    assert module.assignments == []
    assert len(module.instances) == 1
    instance = module.instances[0]
    assert instance.cell.name == "sky130_fd_sc_hd__conb_1"
    assert {
        connection.pin: connection.net for connection in instance.connections
    } == {expected_cell: "y"}


def test_parse_verilog_rejects_unknown_scalar_constants(tmp_path):
    verilog_path = tmp_path / "unknown_tie.v"
    verilog_path.write_text(
        "module tie_example(output wire y); assign y = 1'bx; endmodule",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="X/Z constants cannot be mapped"):
        eda.generic_map(eda.parse_verilog(str(verilog_path)))


def test_parse_verilog_errors_when_assignment_has_no_cell_mapping(tmp_path):
    verilog_path = tmp_path / "unsupported_operator.v"
    verilog_path.write_text(
        "module top(input wire a, b, output wire y); "
        "assign y = a + b; endmodule",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="no supported generic mapping"):
        eda.generic_map(eda.parse_verilog(str(verilog_path)))


def test_parse_full_adder_lowers_nested_boolean_expressions():
    verilog_path = (
        Path(__file__).resolve().parents[1] / "examples/verilog/full_adder.v"
    )

    module = map_verilog(verilog_path)

    assert module.assignments == []
    assert len(module.instances) == 5
    assert len(module.nets) == 8
    cell_names = [instance.cell.name for instance in module.instances]
    assert cell_names.count("sky130_fd_sc_hd__xor3_1") == 1
    assert cell_names.count("sky130_fd_sc_hd__and2_1") == 3
    assert cell_names.count("sky130_fd_sc_hd__or3_1") == 1
    driven_nets = {
        connection.net
        for instance in module.instances
        for connection in instance.connections
        if connection.pin
        in {"X", "Y"}
    }
    assert {"Sum", "Cout"} <= driven_nets


def test_generic_map_uses_library_independent_multi_input_cells():
    verilog_path = (
        Path(__file__).resolve().parents[1] / "examples/verilog/full_adder.v"
    )

    module = eda.generic_map(eda.parse_verilog(str(verilog_path)))

    assert module.assignments == []
    assert all(instance.cell is None for instance in module.instances)
    assert [instance.cell_type for instance in module.instances].count("XOR3") == 1
    assert [instance.cell_type for instance in module.instances].count("AND2") == 3
    assert [instance.cell_type for instance in module.instances].count("OR3") == 1


def test_generic_map_renames_all_boolean_gate_primitives(tmp_path):
    verilog_path = tmp_path / "boolean_gate_primitives.v"
    verilog_path.write_text(
        """
        module top(input wire a, b, c, enable,
                   output wire y_and, y_nand, y_or, y_nor,
                   y_xor, y_xnor, y_buf, y_not, y_bufif0, y_bufif1,
                   y_notif0, y_notif1, y_pullup, y_pulldown);
            and u_and(y_and, a, b, c);
            nand u_nand(y_nand, a, b, c);
            or u_or(y_or, a, b, c);
            nor u_nor(y_nor, a, b, c);
            xor u_xor(y_xor, a, b, c);
            xnor u_xnor(y_xnor, a, b, c);
            buf u_buf(y_buf, a);
            not u_not(y_not, a);
            bufif0 u_bufif0(y_bufif0, a, enable);
            bufif1 u_bufif1(y_bufif1, a, enable);
            notif0 u_notif0(y_notif0, a, enable);
            notif1 u_notif1(y_notif1, a, enable);
            pullup u_pullup(y_pullup);
            pulldown u_pulldown(y_pulldown);
        endmodule
        """,
        encoding="utf-8",
    )

    generic = eda.generic_map(eda.parse_verilog(str(verilog_path)))

    assert [
        (instance.name, instance.cell_type)
        for instance in generic.instances
        if instance.name.startswith("u_")
    ] == [
        ("u_and", "AND3"), ("u_nand", "NAND3"), ("u_or", "OR3"),
        ("u_nor", "NOR3"), ("u_xor", "XOR3"), ("u_xnor", "XNOR3"),
        ("u_buf", "BUF"), ("u_not", "INV"),
        ("u_bufif0", "TRIBUF"), ("u_bufif1", "TRIBUF"),
        ("u_notif0", "TRIBUF"), ("u_notif1", "TRIBUF"),
        ("u_pullup", "TIEHI"), ("u_pulldown", "TIELO"),
    ]
    assert [instance.cell_type for instance in generic.instances].count("INV") == 5
    mapped = eda.tech_map(generic, default_cell_library())
    assert all(instance.cell is not None for instance in mapped.instances)


def test_tech_map_matches_liberty_logic_by_function_and_pin_names(tmp_path):
    verilog_path = tmp_path / "custom_nand.v"
    liberty_path = tmp_path / "custom_logic.lib"
    verilog_path.write_text(
        "module top(input wire a, b, output wire y); "
        "nand u_nand(y, a, b); endmodule",
        encoding="utf-8",
    )
    liberty_path.write_text(
        """
        library (custom) {
            cell (mystery_logic) {
                area : 1.0;
                pin (P) { direction : input; }
                pin (R) { direction : input; }
                pin (Q) { direction : output; function : "!(P & R)"; }
            }
        }
        """,
        encoding="utf-8",
    )

    generic = eda.generic_map(eda.parse_verilog(str(verilog_path)))
    assert generic.instances[0].cell_type == "NAND2"
    mapped = eda.tech_map(generic, eda.parse_liberty(str(liberty_path)))

    assert mapped.instances[0].cell.name == "mystery_logic"
    assert {
        connection.pin: connection.net
        for connection in mapped.instances[0].connections
    } == {"Q": "y", "P": "a", "R": "b"}


@pytest.mark.parametrize(
    ("primitive", "connections"),
    [
        ("tran", "y, a"),
        ("rtran", "y, a"),
        ("tranif0", "y, a, b"),
        ("tranif1", "y, a, b"),
        ("rtranif0", "y, a, b"),
        ("rtranif1", "y, a, b"),
        ("cmos", "y, a, b, c"),
        ("nmos", "y, a, b"),
        ("pmos", "y, a, b"),
        ("rnmos", "y, a, b"),
        ("rpmos", "y, a, b"),
    ],
)
def test_generic_map_rejects_switch_level_primitives(
    tmp_path, primitive, connections
):
    verilog_path = tmp_path / f"{primitive}.v"
    verilog_path.write_text(
        "module top(input wire a, b, c, output wire y); "
        f"{primitive} u_switch({connections}); endmodule",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="does not support switch-level"):
        eda.generic_map(eda.parse_verilog(str(verilog_path)))


def test_generic_map_lowers_combinational_user_defined_primitives(tmp_path):
    verilog_path = tmp_path / "custom_udp.v"
    verilog_path.write_text(
        """
        primitive custom_xor (y, a, b);
            output y;
            input a, b;
            table
                0 0 : 0;
                0 1 : 1;
                1 0 : 1;
                1 1 : 0;
            endtable
        endprimitive
        module top(input wire a, b, output wire y);
            custom_xor u_xor(y, a, b);
        endmodule
        """,
        encoding="utf-8",
    )

    generic = eda.generic_map(eda.parse_verilog(str(verilog_path)))

    assert {instance.cell_type for instance in generic.instances} == {
        "INV",
        "AND2",
        "OR2",
    }
    mapped = eda.tech_map(generic, default_cell_library())
    assert all(instance.cell is not None for instance in mapped.instances)


def test_generated_logic_instance_names_do_not_collide_with_nets(tmp_path):
    verilog_path = tmp_path / "generated_name_collision.v"
    verilog_path.write_text(
        "module top(input wire a, b, output wire y); "
        "wire __generic_inst0; assign y = a ^ b; endmodule",
        encoding="utf-8",
    )

    module = eda.generic_map(eda.parse_verilog(str(verilog_path)))

    assert {net.name for net in module.nets}.isdisjoint(
        instance.name for instance in module.instances
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "and_gate.v",
        "or_gate.v",
        "mux.v",
        "alarm_logic.v",
        "majority3.v",
        "at_least_two_of_four.v",
        "half_adder.v",
    ],
)
def test_parse_combinational_examples_resolves_every_instance(relative_path):
    verilog_path = Path(__file__).resolve().parents[1] / "examples/verilog" / relative_path

    module = map_verilog(verilog_path)

    assert module.instances
    assert module.assignments == []
    assert all(instance.cell is not None for instance in module.instances)


def test_parse_verilog_rejects_unmapped_procedural_blocks(tmp_path):
    verilog_path = tmp_path / "testbench.v"
    verilog_path.write_text(
        'module testbench; initial begin $display("simulation only"); end endmodule',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="procedural block 'initial'.*simulation-only"):
        eda.parse_verilog(str(verilog_path))


def test_parse_liberty_exposes_cells_pins_and_timing_tables(tmp_path):
    liberty_path = tmp_path / "demo.lib"
    liberty_path.write_text(LIBERTY, encoding="utf-8")

    library = eda.parse_liberty(str(liberty_path))

    assert library.name == "demo_cells"
    assert library.time_unit == "1ns"
    assert library.capacitance_unit == "1, pf"
    assert library.cell_count == 1
    cell = library.cells[0]
    assert cell.name == "AND2_X1"
    assert cell.area == pytest.approx(1.25)
    assert [(pin.name, pin.direction) for pin in cell.pins] == [
        ("A", "input"),
        ("B", "input"),
        ("Y", "output"),
    ]
    assert cell.pins[0].capacitance == pytest.approx(0.003)
    output_pin = cell.pins[2]
    assert output_pin.function == "A & B"
    assert len(output_pin.timing_arcs) == 1
    arc = output_pin.timing_arcs[0]
    assert arc.related_pin == "A"
    assert arc.timing_type == "combinational"
    assert arc.timing_sense == "positive_unate"
    assert arc.cell_rise.template_name == "delay_template"
    assert arc.cell_rise.index_1 == pytest.approx([0.1, 0.2])
    assert arc.cell_rise.index_2 == pytest.approx([1.0, 2.0])
    assert arc.cell_rise.values == [
        pytest.approx([0.11, 0.12]),
        pytest.approx([0.21, 0.22]),
    ]
    tensor = arc.cell_rise.to_tensor()
    assert tensor.shape == (2, 2)
    assert tensor.size() == 4
    assert list(tensor.data_fp32()) == pytest.approx([0.11, 0.12, 0.21, 0.22])
    assert arc.cell_fall.values == [
        pytest.approx([0.31, 0.32]),
        pytest.approx([0.41, 0.42]),
    ]
    assert arc.rise_constraint.values == [
        pytest.approx([0.51, 0.52]),
        pytest.approx([0.61, 0.62]),
    ]


def test_parse_liberty_accepts_backslash_line_continuations(tmp_path):
    source = """
    library (continued_values) {
      cell (BUF_X1) {
        pin (Y) {
          direction : output;
          timing () {
            cell_rise (delay_template) {
              values ("0.11, 0.12", \\
                      "0.21, 0.22");
            }
          }
        }
      }
    }
    """

    liberty_path = tmp_path / "continued.lib"
    liberty_path.write_text(source, encoding="utf-8")
    library = eda.parse_liberty(str(liberty_path))
    table = library.cells[0].pins[0].timing_arcs[0].cell_rise

    assert table.values == [
        pytest.approx([0.11, 0.12]),
        pytest.approx([0.21, 0.22]),
    ]


def test_parse_educational_liberty_asset():
    liberty_path = Path(__file__).resolve().parents[1] / "assets/cells.lib"

    library = eda.parse_liberty(str(liberty_path))

    assert library.name == "labneura_educational_cells"
    assert {cell.name for cell in library.cells} == {
        "INV_X1",
        "BUF_X1",
        "TIELO_X1",
        "TIEHI_X1",
        "AND2_X1",
        "XOR2_X1",
        "OR2_X1",
        "MUX2_X1",
        "MUX4_X1",
    }
    mux = next(cell for cell in library.cells if cell.name == "MUX2_X1")
    output = next(pin for pin in mux.pins if pin.name == "Y")
    assert output.function == "(!S & A0) | (S & A1)"
    assert {arc.related_pin for arc in output.timing_arcs} == {"A0", "A1", "S"}
    assert output.timing_arcs[0].cell_rise.values


def test_parse_sky130_typical_liberty_asset():
    liberty_path = (
        Path(__file__).resolve().parents[1]
        / "assets/sky130_fd_sc_hd__tt_025C_1v80.lib"
    )

    library = eda.parse_liberty(str(liberty_path))

    assert library.name == "sky130_fd_sc_hd__tt_025C_1v80"
    assert library.cell_count == 428


def test_parse_liberty_rejects_missing_cells(tmp_path):
    liberty_path = tmp_path / "empty.lib"
    liberty_path.write_text(
        'library (empty) { time_unit : "1ns"; }', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="contains no cell"):
        eda.parse_liberty(str(liberty_path))


def test_parse_verilog_directory_recursively_collects_modules(tmp_path):
    project_dir = tmp_path / "project"
    nested_dir = project_dir / "nested"
    nested_dir.mkdir(parents=True)
    (project_dir / "design.v").write_text(
        """
        module first (a, y);
          input a;
          output y;
          wire w;
          and U1 (w, a, a);
        endmodule

        module second (b, z);
          input b;
          output z;
          and U2 (z, b, b);
        endmodule
        """,
        encoding="utf-8",
    )
    (nested_dir / "third.v").write_text(
        """
        module third (c, f);
          input c;
          output f;
          wire q;
          and U3 (q, c, c);
          or U4 (f, q, c);
        endmodule
        """,
        encoding="utf-8",
    )
    (project_dir / "notes.txt").write_text("ignored", encoding="utf-8")

    project = eda.parse_verilog(str(project_dir))

    assert project.module_count == 3
    assert [module.name for module in project.modules] == [
        "first",
        "second",
        "third",
    ]
    assert [module.port_count for module in project.modules] == [2, 2, 2]
    assert [module.wire_count for module in project.modules] == [1, 0, 1]
    assert [module.instance_count for module in project.modules] == [1, 1, 2]
    assert project.total_port_count == 6
    assert project.total_wire_count == 2
    assert project.total_instance_count == 4
    assert project.get_module_count() == project.module_count


def test_parse_behavioral_verilog_project():
    project_path = Path(__file__).resolve().parents[1] / "examples/verilog/8080_subset"
    project = eda.parse_verilog(str(project_path))
    ast = json.loads(project.ast_json)

    assert project.module_count >= 1
    assert project.source_files
    assert any(
        module.name == "cpu8080_subset" and module.port_count == 10
        for module in project.modules
    )
    assert project.total_port_count == 10
    assert project.total_instance_count == 1
    assert "ProceduralBlock" in project.ast_json
    assert isinstance(ast, dict)
    assert len(ast["definitions"]) == project.module_count
    assert "elaborated_root" in ast


def test_parse_full_8080_core_project():
    project_path = Path(__file__).resolve().parents[1] / "examples/verilog/8080_core"
    project = eda.parse_verilog(str(project_path))

    assert project.module_count == 3
    assert {module.name for module in project.modules} == {
        "alu",
        "i8080",
        "tb_i8080",
    }
    assert project.source_files
    testbench = next(module for module in project.modules if module.name == "tb_i8080")
    cpu = next(module for module in project.modules if module.name == "i8080")
    assert [
        (instance.name, instance.cell_type) for instance in testbench.instances
    ] == [("cpu", "i8080")]
    assert {
        connection.pin: connection.net
        for connection in testbench.instances[0].connections
    } == {
        "i_rst": "reset",
        "i_clk": "clk",
        "i_ready": "ready",
        "o_io": "io",
        "o_addr": "address",
        "i_dat_r": "data_to_cpu",
        "o_read": "read",
        "o_dat_w": "data_from_cpu",
        "o_write": "write",
        "i_int": "1'b0",
        "o_inta": "interrupt_ack",
    }
    alu_connections = {
        connection.pin: connection.net for connection in cpu.instances[0].connections
    }
    assert alu_connections["in_a"] == "r_a"
    assert alu_connections["in_c"] == "r_f_c"
    assert alu_connections["out"] == "alu_out"
    assert alu_connections["out_p"] == "alu_out_p"


def test_parse_modular_traffic_signal_project():
    project_path = Path(__file__).resolve().parents[1] / "examples/verilog/traffic_signal"
    project = eda.parse_verilog(str(project_path))

    assert project.module_count == 5
    assert {module.name for module in project.modules} == {
        "tick_generator",
        "phase_timer",
        "traffic_fsm",
        "traffic_signal",
        "tb_traffic_signal",
    }
    assert project.total_instance_count == 4


def test_parse_verilog_directory_resolves_nested_include_directories(tmp_path):
    project_dir = tmp_path / "project"
    source_dir = project_dir / "rtl"
    include_dir = project_dir / "include"
    source_dir.mkdir(parents=True)
    include_dir.mkdir()
    (include_dir / "width.vh").write_text("`define DATA_WIDTH 8\n", encoding="utf-8")
    (source_dir / "top.v").write_text(
        """
        `include "width.vh"
        module top(input wire [`DATA_WIDTH-1:0] data, output wire ready);
          assign ready = &data;
        endmodule
        """,
        encoding="utf-8",
    )

    project = eda.parse_verilog(str(project_dir))

    assert project.module_count == 1
    assert project.modules[0].name == "top"
    assert project.modules[0].port_count == 2


def test_parse_verilog_directory_reports_syntax_errors(tmp_path):
    project_dir = tmp_path / "broken"
    project_dir.mkdir()
    (project_dir / "broken.v").write_text("module broken ( ;", encoding="utf-8")

    with pytest.raises(RuntimeError, match="slang failed to parse|compile"):
        eda.parse_verilog(str(project_dir))


def test_tensor_supports_two_dimensions():
    import labneura

    matrix = labneura.Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert matrix.shape == (2, 2)
    assert (matrix.rows(), matrix.cols(), matrix.ndim()) == (2, 2, 2)
    assert list(matrix.add(matrix).data_fp32()) == [2.0, 4.0, 6.0, 8.0]
    assert labneura.Tensor([1.0, 2.0, 3.0]).shape == (3,)
    with pytest.raises(Exception):
        labneura.Tensor([[1.0], [2.0, 3.0]])
    with pytest.raises(Exception):
        matrix.add(labneura.Tensor([1.0, 2.0, 3.0, 4.0]))


BEHAVIORAL = """
module beh (input clk, input rst, input [1:0] op, input [3:0] a, b,
            output reg [3:0] y, output reg [3:0] cnt);
    reg [3:0] mem [0:1];
    always @* begin
        y = 4'd0;
        case (op)
            2'd0: y = a + b;
            2'd1: y = a & b;
            2'd2: y = mem[a[0]];
            default: y = a ^ b;
        endcase
    end
    always @(posedge clk) begin
        if (rst) cnt <= 0; else cnt <= cnt + 1;
        mem[b[0]] <= a;
    end
endmodule
"""


def _simulate(module, inputs, state):
    """Evaluate a generic-gate module for one cycle; returns (outputs, next_state)."""
    values = {"__tie0": 0, "__tie1": 1, **inputs}
    flops = [i for i in module.instances if i.cell_type == "DFF"]
    for flop in flops:
        values[{c.pin: c.net for c in flop.connections}["Y"]] = state.get(flop.name, 0)
    pending = [i for i in module.instances if i.cell_type != "DFF"]
    while pending:
        remaining = []
        for gate in pending:
            pins = {c.pin: c.net for c in gate.connections}
            ins = [n for p, n in pins.items() if p != "Y"]
            if any(n not in values for n in ins):
                remaining.append(gate)
                continue
            v = {p: values[n] for p, n in pins.items() if p != "Y"}
            kind = gate.cell_type
            out = {
                "INV": lambda: 1 - v["A"], "BUF": lambda: v["A"],
                "AND2": lambda: v["A"] & v["B"], "OR2": lambda: v["A"] | v["B"],
                "XOR2": lambda: v["A"] ^ v["B"],
                "MUX2": lambda: v["A1"] if v["S"] else v["A0"],
                "TIEHI": lambda: 1, "TIELO": lambda: 0,
            }[kind]()
            values[pins["Y"]] = out
        assert len(remaining) < len(pending), "combinational loop"
        pending = remaining
    nxt = {f.name: values[{c.pin: c.net for c in f.connections}["D"]] for f in flops}
    return values, nxt


def _bits(name, value, width):
    return {f"{name}[{i}]": (value >> i) & 1 for i in range(width)}


def _word(values, name, width):
    return sum(values[f"{name}[{i}]"] << i for i in range(width))


def test_generic_map_behavioral_matches_reference_model(tmp_path):
    source = tmp_path / "beh.v"
    source.write_text(BEHAVIORAL)
    module = eda.generic_map(eda.parse_verilog(str(source)))
    assert {i.cell_type for i in module.instances} <= {
        "AND2", "OR2", "XOR2", "INV", "MUX2", "DFF", "BUF", "TIEHI", "TIELO"}
    assert sum(i.cell_type == "DFF" for i in module.instances) == 4 + 8

    import random
    rng = random.Random(3)
    state, cnt, mem = {}, 0, [0, 0]
    flop_by_net = {}
    for flop in (i for i in module.instances if i.cell_type == "DFF"):
        flop_by_net[{c.pin: c.net for c in flop.connections}["Y"]] = flop.name
    for _ in range(60):
        rst, op = int(rng.random() < 0.1), rng.randrange(4)
        a, b = rng.randrange(16), rng.randrange(16)
        inputs = {"clk": 0, "rst": rst, **_bits("op", op, 2),
                  **_bits("a", a, 4), **_bits("b", b, 4)}
        values, state_next = _simulate(module, inputs, state)
        expected = [(a + b) & 15, a & b, mem[a & 1], a ^ b][op]
        assert _word(values, "y", 4) == expected
        assert _word(values, "cnt", 4) == cnt
        cnt = 0 if rst else (cnt + 1) & 15
        mem[b & 1] = a
        state = state_next


def test_generic_map_behavioral_infers_latch(tmp_path):
    source = tmp_path / "latch.v"
    source.write_text("module latch(input a, input d, output reg y);"
                      " always @* if (a) y = d; endmodule")
    module = eda.generic_map(eda.parse_verilog(str(source)))
    assert [i.cell_type for i in module.instances].count("LATCH") == 1


def test_generic_map_behavioral_async_reset_and_set_flops(tmp_path):
    source = tmp_path / "async.v"
    source.write_text("""
module asy(input clk, input rn, input s, input d, output reg q, output reg p);
  always @(posedge clk or negedge rn) if (!rn) q <= 1'b0; else q <= d;
  always @(posedge clk or posedge s) if (s) p <= 1'b1; else p <= d;
endmodule
""")
    module = eda.generic_map(eda.parse_verilog(str(source)))
    kinds = [i.cell_type for i in module.instances]
    assert kinds.count("DFFR") == 1 and kinds.count("DFFS") == 1
    library = eda.parse_liberty(str(Path(__file__).parent.parent / "assets" /
                                    "sky130_fd_sc_hd__tt_025C_1v80.lib"))
    mapped = [i.cell_type for i in eda.tech_map(module, library).instances]
    assert "sky130_fd_sc_hd__dfrtp_1" in mapped and "sky130_fd_sc_hd__dfstp_1" in mapped
    assert "sky130_fd_sc_hd__dlxtp_1" not in mapped


def test_generic_map_behavioral_functions_tasks_and_division(tmp_path):
    source = tmp_path / "arith.v"
    source.write_text("""
module arith(input [7:0] a, input [7:0] b, output [7:0] q, output [7:0] r,
             output [7:0] f, output [7:0] t, output [7:0] p);
  function [7:0] twice; input [7:0] x; begin twice = x + x; end endfunction
  task bump; input [7:0] x; output [7:0] y; begin y = x + 1; end endtask
  reg [7:0] tmp;
  always @* bump(a, tmp);
  assign q = a / b;
  assign r = a % b;
  assign f = twice(a);
  assign t = tmp;
  assign p = b ** 2;
endmodule
""")
    module = eda.generic_map(eda.parse_verilog(str(source)))
    import random
    rng = random.Random(5)
    for _ in range(40):
        a, b = rng.randrange(256), rng.randrange(1, 256)
        values, _ = _simulate(module, {**_bits("a", a, 8), **_bits("b", b, 8)}, {})
        assert _word(values, "q", 8) == a // b
        assert _word(values, "r", 8) == a % b
        assert _word(values, "f", 8) == (2 * a) & 255
        assert _word(values, "t", 8) == (a + 1) & 255
        assert _word(values, "p", 8) == (b * b) & 255


def test_synthesized_behavioral_design_maps_to_liberty_flops(tmp_path):
    source = tmp_path / "beh.v"
    source.write_text(BEHAVIORAL)
    library = eda.parse_liberty(str(Path(__file__).parent.parent / "assets" /
                                    "sky130_fd_sc_hd__tt_025C_1v80.lib"))
    mapped = eda.tech_map(eda.generic_map(eda.parse_verilog(str(source))), library)
    names = [i.cell_type for i in mapped.instances]
    assert names.count("sky130_fd_sc_hd__dfxtp_1") == 12
    assert all(n.startswith("sky130_fd_sc_hd__") for n in names)


def test_generic_map_behavioral_dual_edge_set_reset_and_tristate(tmp_path):
    source = tmp_path / "misc.v"
    source.write_text("""
module misc(input clk, input rn, input s, input d, input en, input a,
            output reg q, output reg both, output y);
  always @(posedge clk or negedge rn or posedge s)
    if (!rn) q <= 1'b0; else if (s) q <= 1'b1; else q <= d;
  always @(posedge clk or negedge clk) both <= d;
  assign y = en ? a : 1'bz;
endmodule
""")
    module = eda.generic_map(eda.parse_verilog(str(source)))
    kinds = [i.cell_type for i in module.instances]
    assert kinds.count("DFFSR") == 1
    assert kinds.count("TRIBUF") == 1
    assert kinds.count("DFF") == 2  # one rising + one falling flop for the dual-edge bit
    library = eda.parse_liberty(str(Path(__file__).parent.parent / "assets" /
                                    "sky130_fd_sc_hd__tt_025C_1v80.lib"))
    mapped = [i.cell_type for i in eda.tech_map(module, library).instances]
    assert "sky130_fd_sc_hd__dfbbp_1" in mapped
    assert "sky130_fd_sc_hd__ebufn_1" in mapped


def test_parse_verilog_flags_behavioral_modules_and_generic_map_lowers_them(tmp_path):
    source = tmp_path / "cnt.v"
    source.write_text("module cnt(input clk, input rst, output reg [3:0] q);"
                      " always @(posedge clk) if (rst) q <= 0; else q <= q + 1; endmodule")
    module = eda.parse_verilog(str(source))
    assert module.is_behavioral and not module.instances
    source.unlink()
    generic = eda.generic_map(module)
    assert sum(i.cell_type == "DFF" for i in generic.instances) == 4


def test_generic_and_tech_map_accept_a_project_with_top_and_parameters(tmp_path):
    (tmp_path / "leaf.v").write_text(
        "module leaf #(parameter W = 2)(input clk, input [W-1:0] d, output reg [W-1:0] q);"
        " always @(posedge clk) q <= d; endmodule")
    (tmp_path / "top.v").write_text(
        "module top #(parameter W = 2)(input clk, input [W-1:0] d, output [W-1:0] q);"
        " leaf #(.W(W)) u(.clk(clk), .d(d), .q(q)); endmodule")
    project = eda.parse_verilog(str(tmp_path))
    for source in tmp_path.glob("*.v"):
        source.unlink()
    library = eda.parse_liberty(str(Path(__file__).parent.parent / "assets" /
                                    "sky130_fd_sc_hd__tt_025C_1v80.lib"))
    assert sum(i.cell_type == "DFF" for i in eda.generic_map(project, "top").instances) == 2
    generic = eda.generic_map(project, "top", {"W": "5"})
    assert sum(i.cell_type == "DFF" for i in generic.instances) == 5
    mapped = eda.tech_map(project, library, "top", {"W": "5"})
    assert [i.cell_type for i in mapped.instances].count("sky130_fd_sc_hd__dfxtp_1") == 5
    with pytest.raises(RuntimeError, match="not found"):
        eda.generic_map(project, "missing")


def test_module_dump_graph_emits_dot(tmp_path):
    path = tmp_path / "t.v"
    path.write_text(
        "module t(input a, input b, output y); assign y = a & b; endmodule",
        encoding="utf-8",
    )
    module = eda.generic_map(eda.parse_verilog(str(path)))
    dot = module.dump_graph()
    assert dot.startswith('digraph "t"') and "->" in dot
    out = tmp_path / "t.dot"
    assert module.dump_graph(str(out)) == "" and out.read_text() == dot
