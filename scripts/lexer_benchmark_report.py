#!/usr/bin/env python3
"""Convert native lexer benchmark logs to JSON and a cross-platform summary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FIELD = re.compile(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=")


def fields(line: str) -> dict[str, str]:
    matches = list(FIELD.finditer(line))
    result: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        result[match.group(1)] = line[start:end].strip()
    return result


def number(value: str):
    if value == "unavailable":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_log(path: Path, platform: str) -> dict:
    metadata: dict[str, object] = {"platform": platform}
    measurements: list[dict[str, object]] = []
    unavailable: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        values = fields(raw)
        if not values:
            continue
        if "case" in values:
            measurements.append({key: number(value) for key, value in values.items()})
        elif "backend" in values and values.get("status") == "unavailable_at_runtime":
            unavailable.append(values["backend"])
        else:
            for key in ("architecture", "cpu_model", "os", "build_type", "compiler",
                        "sources", "input_bytes", "corpus_tokens", "warmups",
                        "repetitions", "hardware_threads"):
                if key in values:
                    metadata[key] = number(values[key])

    aggregate = [
        item for item in measurements
        if item.get("case") == "lexer_end_to_end" and "input" not in item
    ]
    if not aggregate:
        raise ValueError(f"{path} contains no aggregate complete-lexer measurement")
    if "scalar" not in {item.get("backend") for item in aggregate}:
        raise ValueError(f"{path} contains no scalar complete-lexer measurement")
    grouped: dict[tuple, dict[str, dict[str, object]]] = {}
    for item in measurements:
        key = (item.get("case"), item.get("input"), item.get("input_bytes"))
        grouped.setdefault(key, {})[str(item.get("backend"))] = item
    for group in grouped.values():
        scalar = group.get("scalar", {}).get("median_MB_per_second")
        slang = group.get("slang", {}).get("median_MB_per_second")
        for item in group.values():
            throughput = item.get("median_MB_per_second")
            item["speedup_vs_scalar"] = (
                throughput / scalar
                if isinstance(throughput, (int, float)) and isinstance(scalar, (int, float))
                else None
            )
            item["ratio_vs_slang"] = (
                throughput / slang
                if isinstance(throughput, (int, float)) and isinstance(slang, (int, float))
                else None
            )
    return {
        "schema_version": 1,
        "correctness": "passed",
        "metadata": metadata,
        "unavailable_backends": sorted(set(unavailable)),
        "measurements": measurements,
    }


def aggregate_rows(reports: list[dict]) -> tuple[list[dict], str]:
    rows: list[dict] = []
    detail: list[dict] = []
    for report in reports:
        metadata = report["metadata"]
        aggregate = {
            item["backend"]: item
            for item in report["measurements"]
            if item.get("case") == "lexer_end_to_end" and "input" not in item
        }
        scalar = aggregate.get("scalar", {}).get("median_MB_per_second")
        slang = aggregate.get("slang", {}).get("median_MB_per_second")
        native_values = [
            aggregate.get(name, {}).get("median_MB_per_second")
            for name in ("scalar", "neon", "avx2")
        ]
        native_values = [value for value in native_values if isinstance(value, (int, float))]
        ratio = max(native_values) / slang if native_values and isinstance(slang, (int, float)) else None
        rows.append({
            "platform": metadata.get("platform", "unknown"),
            "cpu": metadata.get("cpu_model", "unknown"),
            "scalar_MB_per_second": scalar,
            "neon_MB_per_second": aggregate.get("neon", {}).get("median_MB_per_second"),
            "avx2_MB_per_second": aggregate.get("avx2", {}).get("median_MB_per_second"),
            "slang_MB_per_second": slang,
            "best_native_over_slang": ratio,
            "compiler": metadata.get("compiler", "unknown"),
            "build_type": metadata.get("build_type", "unknown"),
        })
        detail.extend(report["measurements"])

    def display(value, suffix=""):
        return "N/A" if value is None else f"{value:.2f}{suffix}"

    lines = [
        "# Native lexer performance summary",
        "",
        "Performance is informational; correctness failures remain fatal.",
        "",
        "| Platform | CPU | Scalar MB/s | NEON MB/s | AVX2 MB/s | Slang MB/s | Best native / Slang |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['platform']} | {row['cpu']} | "
            f"{display(row['scalar_MB_per_second'])} | "
            f"{display(row['neon_MB_per_second'])} | "
            f"{display(row['avx2_MB_per_second'])} | "
            f"{display(row['slang_MB_per_second'])} | "
            f"{display(row['best_native_over_slang'], 'x')} |"
        )
    lines.extend([
        "",
        "LabNeura emits comments as tokens; Slang attaches trivia and can split literals.",
        "All measurements use identical in-memory bytes, but token counts are not directly comparable.",
    ])
    return rows, "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    parse = subparsers.add_parser("parse")
    parse.add_argument("--input", type=Path, required=True)
    parse.add_argument("--output", type=Path, required=True)
    parse.add_argument("--platform", required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--input-dir", type=Path, required=True)
    aggregate.add_argument("--json", type=Path, required=True)
    aggregate.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "parse":
        report = parse_log(args.input, args.platform)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
        return 0

    paths = sorted(args.input_dir.rglob("lexer-benchmark-*.json"))
    if not paths:
        raise SystemExit("no lexer benchmark JSON artifacts found")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rows, markdown = aggregate_rows(reports)
    args.json.write_text(json.dumps({"schema_version": 1, "platforms": rows},
                                    indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
