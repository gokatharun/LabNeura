#!/usr/bin/env python3
"""Generate the compact root performance summary from benchmark artifacts.

The script intentionally reports ``N/C`` until a compatible persisted baseline
is supplied; it never treats an unrelated runner or compiler as a comparison.
"""
from __future__ import annotations

import argparse
import csv
import platform
import statistics
import subprocess
from pathlib import Path


def revision(ref: str) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
    except subprocess.CalledProcessError:
        return "unavailable"


def tensor_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[dict[str, str]] = list(csv.DictReader(path.open(encoding="utf-8")))
    selected = [row for row in rows if row.get("size") == "1048576" and
                row.get("operation") == "ADD"]
    result = []
    for row in selected:
        ns = float(row["median_ns"])
        size = int(row["size"])
        throughput = size / ns * 1000.0  # decimal million elements per second
        result.append((f"Tensor {row['backend']} {row['dtype']} ADD", f"{throughput:.2f} M elem/s"))
    return result


def lexer_rows(path: Path) -> list[tuple[str, str]]:
    from lexer_benchmark_report import fields, number
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = {key: number(value) for key, value in fields(line).items()}
        if row.get("case") == "lexer_end_to_end" and "input" not in row:
            value = row.get("median_MB_per_second")
            if isinstance(value, (int, float)):
                result.append((f"EDA lexer {row.get('backend', 'unknown')}", f"{value:.2f} MB/s"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--lexer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = tensor_rows(args.tensor) + lexer_rows(args.lexer)
    if not rows:
        raise SystemExit("no stable representative benchmark measurements found")
    lines = [
        "# Performance",
        "",
        f"Current measured source revision: `{revision('HEAD')}`.",
        f"Immediate previous source revision: `{revision('HEAD~1')}` (N/C until a compatible persisted measurement exists).",
        f"Measurement host: `{platform.platform()}`; uninstrumented Release binaries; benchmark-specific warmups/samples are retained in artifacts.",
        "",
        "Higher throughput is better. `N/C` means no compatible baseline is available; it is not zero change.",
        "",
        "| Workload | Previous | Current | Change |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(f"| {name} | N/C | {value} | N/C |" for name, value in rows)
    lines.extend(["", "Unsupported backends are reported by their benchmark artifacts as `N/A`; no fallback is labelled as SIMD."])
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
