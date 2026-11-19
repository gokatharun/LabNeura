#!/usr/bin/env python3
"""Generate deterministic SystemVerilog frontend profiling workloads.

The generated sources deliberately scale real declarations, expressions, or
procedural statements.  Target sizes are lower bounds, not exact byte counts;
no whitespace-only padding is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_SIZES = (1024, 10 * 1024, 100 * 1024, 1024 * 1024, 4 * 1024 * 1024)


def _module(family: str, count: int) -> str:
    width = max(1, count)
    name = f"frontend_{family}_{count}"
    width_expression = "WIDTH" if family == "parameter" else str(width)
    parameter_clause = (
        f" #(parameter int WIDTH = {width})" if family == "parameter" else ""
    )
    header = (
        f"module {name}{parameter_clause}(\n"
        f"    input logic [{width_expression} - 1:0] a, b,\n"
        f"    input logic clk, rst_n, sel,\n"
        f"    output logic [{width_expression} - 1:0] y\n"
        ");\n"
    )

    if family == "boolean":
        body = "".join(
            f"  assign y[{i}] = ((a[{i}] & b[{i}]) ^ (~a[{i}] | b[{i}])) & "
            f"(a[{i}] ^ (b[{i}] | (a[{i}] & ~b[{i}])));\n"
            for i in range(count)
        )
    elif family == "arithmetic":
        body = "".join(
            f"  assign y[{i}] = (a[{i}] + b[{i}]) - (a[{i}] & b[{i}]);\n"
            for i in range(count)
        )
    elif family == "wide":
        body = "".join(
            f"  assign y[{i}] = a[{i}] ^ b[{(i * 17 + 3) % width}];\n"
            for i in range(count)
        )
    elif family == "continuous":
        body = "".join(
            f"  assign y[{i}] = a[{i}] & b[{i}];\n" for i in range(count)
        )
    elif family == "conditional":
        body = "".join(
            f"  assign y[{i}] = sel ? (a[{i}] ? b[{i}] : ~b[{i}]) : "
            f"(b[{i}] ? a[{i}] : 1'b0);\n"
            for i in range(count)
        )
    elif family == "case":
        zeros = ", ".join("1'b0" for _ in range(count))
        selected = ", ".join(
            f"(sel ? a[{i}] : b[{i}])" for i in reversed(range(count)))
        inverted = ", ".join(
            f"~(a[{i}] ^ b[{i}])" for i in reversed(range(count)))
        ones = ", ".join("1'b1" for _ in range(count))
        body = (
            "  always_comb begin\n"
            "    case ({a[0], b[0]})\n"
            f"      2'b00: y = {{{zeros}}};\n"
            f"      2'b01: y = {{{selected}}};\n"
            f"      2'b10: y = {{{inverted}}};\n"
            f"      default: y = {{{ones}}};\n"
            "    endcase\n  end\n"
        )
    elif family == "sequential":
        statements = "".join(
            f"      y[{i}] <= a[{i}] ^ b[{i}];\n" for i in range(count)
        )
        body = (
            "  always_ff @(posedge clk or negedge rst_n) begin\n"
            "    if (!rst_n) y <= '0;\n"
            "    else begin\n"
            + statements
            + "    end\n  end\n"
        )
    elif family == "parameter":
        # Local parameters exercise constant resolution without changing the
        # externally comparable flat module interface.
        declarations = "".join(
            f"  localparam logic P_{i} = 1'b{(i * 13 + 1) & 1};\n"
            for i in range(count)
        )
        assigns = "".join(
            f"  assign y[{i}] = a[{i}] ^ P_{i};\n" for i in range(count)
        )
        body = declarations + assigns
    elif family == "identifier":
        declarations = "".join(
            f"  logic deterministic_intermediate_signal_{i:07d};\n"
            for i in range(count)
        )
        assigns = "".join(
            f"  assign deterministic_intermediate_signal_{i:07d} = a[{i}] ^ b[{i}];\n"
            f"  assign y[{i}] = deterministic_intermediate_signal_{i:07d};\n"
            for i in range(count)
        )
        body = declarations + assigns
    elif family == "comment":
        body = "".join(
            f"  // signal {i}: deterministic Boolean cone for profiling source trivia\n"
            f"  assign y[{i}] = a[{i}] ^ b[{i}];\n"
            for i in range(count)
        )
    elif family == "literal":
        body = "".join(
            f"  assign y[{i}] = a[{i}] ^ 32'h{((i * 2654435761) & 0xFFFFFFFF):08x};\n"
            for i in range(count)
        )
    else:
        raise ValueError(f"unknown workload family: {family}")
    return header + body + "endmodule\n"


FAMILIES = (
    "boolean",
    "arithmetic",
    "wide",
    "continuous",
    "conditional",
    "case",
    "sequential",
    "parameter",
    "identifier",
    "comment",
    "literal",
)


def generate_source(family: str, target_bytes: int) -> tuple[str, int]:
    """Return the smallest generated workload meeting ``target_bytes``."""
    if family not in FAMILIES:
        raise ValueError(f"unknown workload family: {family}")
    if target_bytes <= 0:
        raise ValueError("target size must be positive")

    low, high = 1, 1
    while len(_module(family, high).encode("utf-8")) < target_bytes:
        low = high + 1
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if len(_module(family, middle).encode("utf-8")) < target_bytes:
            low = middle + 1
        else:
            high = middle
    return _module(family, low), low


def _size_label(size: int) -> str:
    for divisor, suffix in ((1024 * 1024, "mib"), (1024, "kib")):
        if size % divisor == 0:
            return f"{size // divisor}{suffix}"
    return f"{size}b"


def generate_corpus(output: Path, sizes: tuple[int, ...], families: tuple[str, ...]) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for target in sizes:
        for family in families:
            source, complexity = generate_source(family, target)
            encoded = source.encode("utf-8")
            filename = f"{family}_{_size_label(target)}.sv"
            (output / filename).write_bytes(encoded)
            records.append(
                {
                    "file": filename,
                    "family": family,
                    "target_bytes": target,
                    "source_bytes": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "top_module": f"frontend_{family}_{complexity}",
                    "module_count": 1,
                    "process_count": 1 if family in ("case", "sequential") else 0,
                    "signal_width": complexity,
                    "complexity_units": complexity,
                    "native_end_to_end_supported": True,
                }
            )
    manifest = {
        "schema_version": 1,
        "generator": "generate_frontend_workloads.py",
        "deterministic": True,
        "workloads": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _positive_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers") from error
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must contain positive integers")
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sizes", type=_positive_sizes, default=DEFAULT_SIZES)
    parser.add_argument("--families", default=",".join(FAMILIES))
    arguments = parser.parse_args()
    families = tuple(item for item in arguments.families.split(",") if item)
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        parser.error("unknown families: " + ", ".join(unknown))
    manifest = generate_corpus(arguments.output, arguments.sizes, families)
    print(json.dumps({"output": str(arguments.output), "workloads": len(manifest["workloads"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
