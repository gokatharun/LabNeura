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


def test_parse_verilog_builds_module_netlist():
    module = eda.parse_verilog_source(VERILOG)

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


def test_parse_verilog_resolves_explicit_liberty_cell_instances(tmp_path):
    verilog_path = tmp_path / "explicit_cell.v"
    verilog_path.write_text(
        "module top (a, b, y); input a, b; output y; "
        "AND2_X1 gate (.A(a), .B(b), .Y(y)); endmodule",
        encoding="utf-8",
    )

    rtl_module = eda.parse_verilog(str(verilog_path))
    assert rtl_module.instances[0].cell is None
    module = eda.tech_map(
        eda.generic_map(rtl_module), default_cell_library()
    )

    instance = module.instances[0]
    assert instance.cell.name == "sky130_fd_sc_hd__and2_1"
    assert instance.cell.pins[0].direction == "input"
    assert {
        connection.pin: connection.net for connection in instance.connections
    } == {"A": "a", "B": "b", "X": "y"}


def test_parse_verilog_resolves_legacy_mux2_cell_alias(tmp_path):
    verilog_path = tmp_path / "mux2_cell.v"
    verilog_path.write_text(
        "module top (a, b, s, y); input a, b, s; output y; "
        "MUX2_X1 mux (.A0(a), .A1(b), .S(s), .Y(y)); endmodule",
        encoding="utf-8",
    )

    instance = map_verilog(verilog_path).instances[0]

    assert instance.cell.name == "sky130_fd_sc_hd__mux2_1"
    assert {
        connection.pin: connection.net for connection in instance.connections
    } == {
        "A0": "a",
        "A1": "b",
        "S": "s",
        "X": "y",
    }


def test_parse_half_adder_verilog_resolves_cells():
    verilog_path = Path(__file__).resolve().parents[1] / "examples/verilog/half_adder.v"

    module = map_verilog(verilog_path)
    source_module = eda.parse_verilog_source(
        verilog_path.read_text(encoding="utf-8")
    )

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


def test_parse_verilog_reports_cells_missing_from_bundled_library(tmp_path):
    verilog_path = tmp_path / "missing_cell.v"
    verilog_path.write_text(
        "module top (a); input a; MISSING_X1 u1 (.A(a)); endmodule",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as error:
        eda.parse_verilog(str(verilog_path))
    assert "MISSING_X1" in str(error.value)


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

    with pytest.raises(RuntimeError, match="procedural block 'initial'.*not yet representable"):
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


def test_parse_liberty_accepts_backslash_line_continuations():
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

    library = eda.parse_liberty_source(source)
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


def test_parse_liberty_source_rejects_missing_cells():
    with pytest.raises(RuntimeError, match="contains no cell"):
        eda.parse_liberty_source('library (empty) { time_unit : "1ns"; }')


def test_parse_verilog_project_recursively_collects_modules(tmp_path):
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

    project = eda.parse_verilog_project(str(project_dir))

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


def test_rejects_undeclared_instance_net():
    source = """
    module bad (a, f);
      input a;
      output f;
      and U1 (f, a, missing);
    endmodule
    """

    with pytest.raises(RuntimeError, match="undeclared net"):
        eda.parse_verilog_source(source)


def test_rejects_unsupported_verilog():
    with pytest.raises(RuntimeError, match="unsupported Verilog statement"):
        eda.parse_verilog_source("module bad (a); input a; assign a = 1'b0; endmodule")


def test_parse_behavioral_verilog_project():
    project_path = Path(__file__).resolve().parents[1] / "examples/verilog/8080_subset"
    project = eda.parse_verilog_project(str(project_path))
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
    project = eda.parse_verilog_project(str(project_path))

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
    project = eda.parse_verilog_project(str(project_path))

    assert project.module_count == 5
    assert {module.name for module in project.modules} == {
        "tick_generator",
        "phase_timer",
        "traffic_fsm",
        "traffic_signal",
        "tb_traffic_signal",
    }
    assert project.total_instance_count == 4


def test_parse_verilog_project_resolves_nested_include_directories(tmp_path):
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

    project = eda.parse_verilog_project(str(project_dir))

    assert project.module_count == 1
    assert project.modules[0].name == "top"
    assert project.modules[0].port_count == 2


def test_parse_verilog_project_reports_syntax_errors(tmp_path):
    project_dir = tmp_path / "broken"
    project_dir.mkdir()
    (project_dir / "broken.v").write_text("module broken ( ;", encoding="utf-8")

    with pytest.raises(RuntimeError, match="slang failed to parse|compile"):
        eda.parse_verilog_project(str(project_dir))


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
