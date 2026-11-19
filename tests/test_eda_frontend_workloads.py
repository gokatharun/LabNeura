import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "benchmarks" / "eda" / "generate_frontend_workloads.py"
SPEC = importlib.util.spec_from_file_location("frontend_workloads", GENERATOR_PATH)
GENERATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GENERATOR)


def test_all_families_are_deterministic_and_scale_real_constructs():
    for family in GENERATOR.FAMILIES:
        first, first_units = GENERATOR.generate_source(family, 2048)
        second, second_units = GENERATOR.generate_source(family, 2048)
        larger, larger_units = GENERATOR.generate_source(family, 4096)
        assert first == second
        assert first_units == second_units
        assert len(first.encode()) >= 2048
        assert len(larger.encode()) >= 4096
        assert larger_units > first_units
        assert "module frontend_" in first
        assert first.rstrip().endswith("endmodule")
        assert "assign" in first or "always_" in first


def test_representative_family_constructs_are_present():
    expected = {
        "boolean": " & ",
        "arithmetic": " + ",
        "wide": " ^ ",
        "continuous": "assign y[",
        "conditional": " ? ",
        "case": "case ({",
        "sequential": "always_ff @(posedge clk or negedge rst_n)",
        "parameter": "#(parameter int WIDTH =",
        "identifier": "deterministic_intermediate_signal_",
        "comment": "deterministic Boolean cone",
        "literal": "32'h",
    }
    for family, marker in expected.items():
        source, _ = GENERATOR.generate_source(family, 1024)
        assert marker in source
    parameter_source, _ = GENERATOR.generate_source("parameter", 1024)
    assert "localparam logic P_" in parameter_source


def test_manifest_hashes_sizes_and_order_are_reproducible(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    families = ("boolean", "case", "sequential", "comment")
    first = GENERATOR.generate_corpus(first_dir, (1024, 4096), families)
    second = GENERATOR.generate_corpus(second_dir, (1024, 4096), families)
    assert first == second
    assert json.loads((first_dir / "manifest.json").read_text()) == first

    assert len(first["workloads"]) == 8
    for record in first["workloads"]:
        payload = (first_dir / record["file"]).read_bytes()
        assert len(payload) == record["source_bytes"]
        assert len(payload) >= record["target_bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        assert f"module {record['top_module']}" in payload.decode()
        assert payload == (second_dir / record["file"]).read_bytes()
        assert record["complexity_units"] > 0
        assert record["native_end_to_end_supported"] is True


def test_invalid_family_and_size_are_rejected():
    for family, size in (("unknown", 1024), ("boolean", 0), ("boolean", -1)):
        try:
            GENERATOR.generate_source(family, size)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid workload request was accepted")
