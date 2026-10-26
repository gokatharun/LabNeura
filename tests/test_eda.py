import json
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
        ("AND1", "AND", {"Y": "w", "A": "a", "B": "b"}),
        ("OR1", "OR", {"Y": "f", "A": "w", "B": "c"}),
    ]
    with pytest.raises(AttributeError):
        module.name = "changed"


def test_parse_file_and_write_json(tmp_path):
    verilog_path = tmp_path / "design.v"
    json_path = tmp_path / "design.json"
    verilog_path.write_text(VERILOG, encoding="utf-8")

    module = eda.parse_verilog(str(verilog_path))
    eda.write_json(module, str(json_path))
    serialized = json.loads(json_path.read_text(encoding="utf-8"))

    assert serialized["format"] == "eda-netlist"
    assert serialized["version"] == 1
    assert serialized["modules"][0]["name"] == "logic_example"
    assert json.loads(eda.to_json(module)) == serialized


def test_parse_verilog_resolves_instances_to_bundled_cells(tmp_path):
    verilog_path = tmp_path / "cells.v"
    verilog_path.write_text(VERILOG, encoding="utf-8")

    module = eda.parse_verilog(str(verilog_path))

    assert [(instance.cell_type, instance.cell.name) for instance in module.instances] == [
        ("and", "AND2_X1"),
        ("or", "OR2_X1"),
    ]
    assert module.instances[0].cell.area == pytest.approx(0.90)
    assert {
        connection.pin: connection.net
        for connection in module.instances[0].connections
    } == {"Y": "w", "A": "a", "B": "b"}


def test_parse_verilog_resolves_explicit_liberty_cell_instances(tmp_path):
    verilog_path = tmp_path / "explicit_cell.v"
    verilog_path.write_text(
        "module top (a, b, y); input a, b; output y; "
        "AND2_X1 gate (.A(a), .B(b), .Y(y)); endmodule",
        encoding="utf-8",
    )

    module = eda.parse_verilog(str(verilog_path))

    instance = module.instances[0]
    assert instance.cell.name == "AND2_X1"
    assert instance.cell.pins[0].direction == "input"
    assert {
        connection.pin: connection.net for connection in instance.connections
    } == {"A": "a", "B": "b", "Y": "y"}


def test_parse_verilog_reports_cells_missing_from_bundled_library(tmp_path):
    verilog_path = tmp_path / "missing_cell.v"
    verilog_path.write_text(
        "module top (a); input a; MISSING_X1 u1 (.A(a)); endmodule",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="bundled standard-cell library") as error:
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
    assert [(assignment.lhs, assignment.rhs) for assignment in module.assignments] == [
        ("Y", "I[S]")
    ]
    assert module.get_assignments()[0].get_lhs() == "Y"
    assert module.get_assignments()[0].get_rhs() == "I[S]"
    rhs = module.assignments[0].rhs_expression
    assert rhs.kind == eda.ExpressionKind.BIT_SELECT
    assert rhs.source == "I[S]"
    assert [(operand.kind, operand.value) for operand in rhs.operands] == [
        (eda.ExpressionKind.IDENTIFIER, "I"),
        (eda.ExpressionKind.IDENTIFIER, "S"),
    ]
    serialized = json.loads(eda.to_json(module))
    serialized_assignment = serialized["modules"][0]["assignments"][0]
    assert serialized_assignment["lhs"] == "Y"
    assert serialized_assignment["rhs"] == "I[S]"
    assert serialized_assignment["rhs_expression"]["kind"] == "bit_select"
    assert [
        operand["value"]
        for operand in serialized_assignment["rhs_expression"]["operands"]
    ] == ["I", "S"]


def test_parse_verilog_lowers_binary_assignment_expression(tmp_path):
    verilog_path = tmp_path / "binary.v"
    verilog_path.write_text(
        "module binary(input wire a, b, output wire y); "
        "assign y = a & b; endmodule",
        encoding="utf-8",
    )

    module = eda.parse_verilog(str(verilog_path))
    rhs = module.assignments[0].rhs_expression

    assert rhs.kind == eda.ExpressionKind.BINARY
    assert rhs.value == "&"
    assert [operand.value for operand in rhs.operands] == ["a", "b"]


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
    assert arc.cell_fall.values == [
        pytest.approx([0.31, 0.32]),
        pytest.approx([0.41, 0.42]),
    ]
    assert arc.rise_constraint.values == [
        pytest.approx([0.51, 0.52]),
        pytest.approx([0.61, 0.62]),
    ]


def test_parse_educational_liberty_asset():
    liberty_path = Path(__file__).resolve().parents[1] / "assets/cells.lib"

    library = eda.parse_liberty(str(liberty_path))

    assert library.name == "labneura_educational_cells"
    assert {cell.name for cell in library.cells} == {
        "INV_X1",
        "AND2_X1",
        "OR2_X1",
        "MUX2_X1",
    }
    mux = next(cell for cell in library.cells if cell.name == "MUX2_X1")
    output = next(pin for pin in mux.pins if pin.name == "Y")
    assert output.function == "(!S & A0) | (S & A1)"
    assert {arc.related_pin for arc in output.timing_arcs} == {"A0", "A1", "S"}
    assert output.timing_arcs[0].cell_rise.values


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
