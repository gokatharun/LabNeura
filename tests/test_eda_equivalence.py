from itertools import product
from pathlib import Path
import shutil
import subprocess
import json

import pytest

import labneura.eda as eda


class Design:
    def __init__(self, name, source, inputs, outputs, reference):
        self.name = name
        self.source = source
        self.inputs = inputs
        self.outputs = outputs
        self.reference = reference


DESIGNS = [
    Design(
        "and_gate",
        "module and_gate(input wire a, input wire b, output wire y);\n"
        "  assign y = a & b;\nendmodule\n",
        ("a", "b"), ("y",), lambda a, b: (a & b,),
    ),
    Design(
        "or_gate",
        "module or_gate(input wire a, input wire b, output wire y);\n"
        "  assign y = a | b;\nendmodule\n",
        ("a", "b"), ("y",), lambda a, b: (a | b,),
    ),
    Design(
        "mux2",
        "module mux2(input wire a, input wire b, input wire sel, output wire y);\n"
        "  assign y = sel ? b : a;\nendmodule\n",
        ("a", "b", "sel"), ("y",), lambda a, b, sel: (b if sel else a,),
    ),
    Design(
        "half_adder",
        "module half_adder(input wire a, input wire b, output wire sum, output wire carry);\n"
        "  assign sum = a ^ b;\n"
        "  assign carry = a & b;\nendmodule\n",
        ("a", "b"), ("sum", "carry"),
        lambda a, b: (a ^ b, a & b),
    ),
    Design(
        "full_adder",
        "module full_adder(input wire a, input wire b, input wire cin, output wire sum, output wire cout);\n"
        "  assign sum = a ^ b ^ cin;\n"
        "  assign cout = (a & b) | (a & cin) | (b & cin);\nendmodule\n",
        ("a", "b", "cin"), ("sum", "cout"),
        lambda a, b, cin: ((a + b + cin) & 1, (a + b + cin) >> 1),
    ),
    Design(
        "comparator2",
        "module comparator2(input wire a1, input wire a0, input wire b1, input wire b0, "
        "output reg less, output reg equal, output reg greater);\n"
        "  always @* begin\n"
        "    less = {a1, a0} < {b1, b0};\n"
        "    equal = {a1, a0} == {b1, b0};\n"
        "    greater = {a1, a0} > {b1, b0};\n"
        "  end\nendmodule\n",
        ("a1", "a0", "b1", "b0"), ("less", "equal", "greater"),
        lambda a1, a0, b1, b0: (
            int((a1 << 1 | a0) < (b1 << 1 | b0)),
            int((a1 << 1 | a0) == (b1 << 1 | b0)),
            int((a1 << 1 | a0) > (b1 << 1 | b0)),
        ),
    ),
]


def _build_blif(design, tmp_path):
    rtl_path = tmp_path / f"{design.name}.v"
    rtl_path.write_text(design.source, encoding="utf-8")

    parsed = eda.parse_verilog(str(rtl_path))
    assert parsed.name == design.name
    assert parsed.child_modules == []
    assert tuple(port.name for port in parsed.ports
                 if port.direction == eda.PortDirection.INPUT) == design.inputs
    assert tuple(port.name for port in parsed.ports
                 if port.direction == eda.PortDirection.OUTPUT) == design.outputs

    generic = eda.generic_map(parsed)
    mapped = eda.fpga_map(generic, lut_size=4)
    assert mapped.child_modules == []
    assert not mapped.is_behavioral
    unsupported_for_combinational_proof = {
        "DFF", "DFFR", "DFFS", "DFFSR", "LATCH", "TRIBUF"
    }
    assert all(
        instance.cell_type not in unsupported_for_combinational_proof
        for module in (generic, mapped)
        for instance in module.instances
    )
    blif = eda.to_blif(mapped)
    (tmp_path / f"{design.name}.blif").write_text(blif, encoding="utf-8")
    return blif


def _parse_blif(blif):
    lines = [line.strip() for line in blif.splitlines() if line.strip()]
    model = None
    inputs = []
    outputs = []
    functions = []
    position = 0

    while position < len(lines):
        fields = lines[position].split()
        directive = fields[0]
        if directive == ".model":
            model = fields[1]
        elif directive == ".inputs":
            inputs = fields[1:]
        elif directive == ".outputs":
            outputs = fields[1:]
        elif directive == ".names":
            if len(fields) < 2:
                raise AssertionError(f"malformed BLIF directive: {lines[position]}")
            function_inputs = fields[1:-1]
            output = fields[-1]
            rows = []
            position += 1
            while position < len(lines) and not lines[position].startswith("."):
                rows.append(lines[position].split())
                position += 1
            functions.append((function_inputs, output, rows))
            continue
        position += 1

    return model, inputs, outputs, functions


def _evaluate_blif(functions, inputs):
    values = dict(inputs)
    pending = list(functions)

    while pending:
        next_pending = []
        progressed = False
        for function_inputs, output, rows in pending:
            if not all(name in values for name in function_inputs):
                next_pending.append((function_inputs, output, rows))
                continue

            result = 0
            for row in rows:
                if not function_inputs:
                    result |= int(row[-1] == "1")
                    continue
                pattern, bit = row
                if bit == "1" and all(
                    expected == "-" or int(expected) == values[name]
                    for name, expected in zip(function_inputs, pattern)
                ):
                    result = 1
                    break
            values[output] = result
            progressed = True

        if not progressed:
            unresolved = [output for _, output, _ in next_pending]
            raise AssertionError(f"BLIF has unresolved or cyclic nets: {unresolved}")
        pending = next_pending

    return values


@pytest.mark.parametrize("design", DESIGNS, ids=lambda design: design.name)
def test_blif_truth_tables_match_exhaustive_reference(tmp_path, design):
    blif = _build_blif(design, tmp_path)
    model, inputs, outputs, functions = _parse_blif(blif)

    assert model == design.name
    assert tuple(inputs) == design.inputs
    assert tuple(outputs) == design.outputs

    for vector in product((0, 1), repeat=len(design.inputs)):
        actual_values = _evaluate_blif(
            functions, dict(zip(design.inputs, vector))
        )
        actual = tuple(actual_values[name] for name in design.outputs)
        expected = design.reference(*vector)
        assert actual == expected, (
            f"{design.name}: inputs={dict(zip(design.inputs, vector))}, "
            f"expected={dict(zip(design.outputs, expected))}, "
            f"actual={dict(zip(design.outputs, actual))}"
        )

@pytest.fixture(scope="module")
def yosys():
    executable = shutil.which("yosys")
    if executable is None:
        pytest.skip("Yosys is required for independent BLIF equivalence proofs")

    version = subprocess.run(
        [executable, "-V"], capture_output=True, text=True, check=False,
        timeout=10,
    )
    if version.returncode != 0:
        pytest.fail(
            f"Unable to query Yosys version: exit={version.returncode}\n"
            f"stdout:\n{version.stdout}\nstderr:\n{version.stderr}"
        )
    return executable, (version.stdout + version.stderr).strip()


def _yosys_quote(path):
    return json.dumps(str(path))


def _run_equivalence(yosys, version, rtl_path, blif_path, gold, candidate,
                     tmp_path, design_name):
    script_path = tmp_path / f"{design_name}_equivalence.ys"
    script_path.write_text(
        "\n".join((
            f"read_verilog -formal {_yosys_quote(rtl_path)}",
            f"read_blif {_yosys_quote(blif_path)}",
            f"equiv_make {gold} {candidate} equiv",
            "prep -top equiv",
            "equiv_simple",
            "equiv_status -assert",
            "",
        )),
        encoding="utf-8",
    )
    command = [yosys, "-s", str(script_path)]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=60,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        pytest.fail(
            f"{design_name}: Yosys equivalence timed out\n"
            f"Yosys version: {version}\nCommand: {command!r}\n"
            f"Proof script: {script_path.read_text(encoding='utf-8')}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return result, command, script_path


def _proof_diagnostic(design_name, version, command, script_path, result):
    return (
        f"Design: {design_name}\nYosys version: {version}\n"
        f"Command: {command!r}\n"
        f"Proof script:\n{script_path.read_text(encoding='utf-8')}\n"
        f"exit: {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("design", DESIGNS, ids=lambda design: design.name)
def test_yosys_proves_rtl_equivalent_to_exported_blif(tmp_path, design, yosys):
    executable, version = yosys
    rtl_path = tmp_path / f"{design.name}.v"
    rtl_path.write_text(design.source, encoding="utf-8")
    blif = _build_blif(design, tmp_path)
    model, inputs, outputs, _ = _parse_blif(blif)
    candidate = f"{design.name}_candidate"
    assert model == design.name
    assert tuple(inputs) == design.inputs
    assert tuple(outputs) == design.outputs

    blif_path = tmp_path / f"{design.name}_candidate.blif"
    original_model = f".model {design.name}\n"
    assert blif.count(original_model) == 1
    blif_path.write_text(
        blif.replace(original_model, f".model {candidate}\n", 1),
        encoding="utf-8",
    )

    result, command, script_path = _run_equivalence(
        executable, version, rtl_path, blif_path,
        design.name, candidate, tmp_path, design.name,
    )
    assert result.returncode == 0, _proof_diagnostic(
        design.name, version, command, script_path, result
    )


def test_yosys_equivalence_rejects_perturbed_candidate(tmp_path, yosys):
    executable, version = yosys
    design = DESIGNS[0]
    rtl_path = tmp_path / "and_gate.v"
    rtl_path.write_text(design.source, encoding="utf-8")
    wrong_blif = tmp_path / "and_gate_wrong.blif"
    wrong_blif.write_text(
        ".model and_gate_candidate\n"
        ".inputs a b\n"
        ".outputs y\n"
        ".names a b y\n"
        "00 1\n"
        "01 1\n"
        "10 1\n"
        "11 1\n"
        ".end\n",
        encoding="utf-8",
    )

    result, command, script_path = _run_equivalence(
        executable, version, rtl_path, wrong_blif,
        "and_gate", "and_gate_candidate", tmp_path, "and_gate_negative",
    )
    diagnostic = _proof_diagnostic(
        "and_gate (deliberately perturbed candidate)", version,
        command, script_path, result,
    )
    assert result.returncode != 0, (
        "Yosys accepted a candidate that drives y high for every input\n" +
        diagnostic
    )
    assert "unproven" in (result.stdout + result.stderr).lower(), diagnostic
