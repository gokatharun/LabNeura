#!/usr/bin/env python3
"""Create the concise, versioned root coverage summary from measured inputs."""
from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def percent(covered: int, total: int) -> str:
    return "N/A" if not total else f"{100.0 * covered / total:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--python-xml", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = json.loads(args.native.read_text(encoding="utf-8"))["totals"]
    if args.python_xml is None or not args.python_xml.is_file():
        python_rate = "N/A"
    else:
        root = ET.parse(args.python_xml).getroot()
        python_lines = root.find(".//coverage")
        python_rate = ("N/A" if python_lines is None else
                       f"{100.0 * float(python_lines.attrib.get('line-rate', 0)):.2f}%")
    lines = [
        "# Coverage",
        "",
        f"Current measured revision: `{revision()}`.",
        "Previous compatible measurement: N/C (baseline not yet persisted).",
        "",
        "| Component | Previous line coverage | Current line coverage | Change |",
        "|---|---:|---:|---:|",
        f"| Native C++ | N/C | {percent(native['lines']['covered'], native['lines']['count'])} | N/C |",
        f"| Python API | N/C | {python_rate} | N/C |",
        "",
        "Detailed function, branch, file, and HTML reports are CI coverage artifacts. "
        "Native exclusions: tests, build output, generated code, and third-party code. "
        "Python API coverage is N/A when no importable Python source package participates in the run. "
        "Architecture-specific kernels that cannot execute on the measuring host remain explicitly unmeasured.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
