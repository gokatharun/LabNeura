"""Portable public-contract tests for the native production EDA frontend."""

from pathlib import Path

import pytest

import labneura.eda as eda


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _connections(instance):
    return {connection.pin: connection.net for connection in instance.connections}


def test_native_multimodule_project_preserves_order_and_hierarchy(tmp_path):
    _write(
        tmp_path / "design.v",
        """
module leaf #(parameter WIDTH = 2)(input [WIDTH-1:0] a,
                                    output [WIDTH-1:0] y);
  assign y = a;
endmodule
module top(input clk, input [1:0] data, output [1:0] result);
  leaf #(.WIDTH(2)) u_leaf(.a(data), .y(result));
endmodule
""",
    )
    project = eda.parse_verilog(str(tmp_path))
    assert [module.name for module in project.modules] == ["top", "leaf"]
    top = next(module for module in project.modules if module.name == "top")
    assert [port.name for port in top.ports] == ["clk", "data", "result"]
    assert [(port.width, port.left_index, port.right_index)
            for port in top.ports] == [(1, None, None), (2, 1, 0), (2, 1, 0)]
    assert [(instance.name, instance.cell_type)
            for instance in top.instances] == [("u_leaf", "leaf")]
    assert _connections(top.instances[0]) == {"a": "data", "y": "result"}


@pytest.mark.parametrize(
    ("a", "b", "expected_and", "expected_xor"),
    [(0, 0, 0, 0), (0, 1, 0, 1), (1, 0, 0, 1), (1, 1, 1, 0)],
)
def test_native_combinational_mapping_matches_truth_table(
    tmp_path, a, b, expected_and, expected_xor
):
    source = _write(
        tmp_path / "logic.v",
        "module logic2(input a, input b, output y, output z);"
        " assign y = a & b; assign z = a ^ b; endmodule\n",
    )
    mapped = eda.generic_map(eda.parse_verilog(str(source)))
    cells = {instance.cell_type: _connections(instance)
             for instance in mapped.instances}
    assert cells["AND2"] == {"A": "a", "B": "b", "Y": "y"}
    assert cells["XOR2"] == {"A": "a", "B": "b", "Y": "z"}
    values = {"a": a, "b": b}
    functions = {
        "AND2": lambda pins: values[pins["A"]] & values[pins["B"]],
        "XOR2": lambda pins: values[pins["A"]] ^ values[pins["B"]],
    }
    for instance in mapped.instances:
        pins = _connections(instance)
        values[pins["Y"]] = functions[instance.cell_type](pins)
    assert values["y"] == expected_and
    assert values["z"] == expected_xor


def test_native_supported_sequential_mapping_has_resettable_state(tmp_path):
    source = _write(
        tmp_path / "register.v",
        """
module register(input clk, input rst, input d, output reg q);
  always @(posedge clk or posedge rst)
    if (rst) q <= 1'b0; else q <= d;
endmodule
""",
    )
    mapped = eda.generic_map(eda.parse_verilog(str(source)))
    reset_flops = [instance for instance in mapped.instances
                   if instance.cell_type == "DFFR"]
    assert len(reset_flops) == 1
    pins = _connections(reset_flops[0])
    assert pins["CLK"] == "clk" and pins["RESET_B"].startswith("__native_not")
    assert "D" in pins and "Y" in pins


def test_native_unsupported_construct_is_rejected_explicitly(tmp_path):
    source = _write(
        tmp_path / "simulation_only.v",
        'module simulation_only; initial $display("unsupported"); endmodule\n',
    )
    with pytest.raises(RuntimeError, match="UnsupportedSimulationOnlyProcess"):
        eda.parse_verilog(str(source))
