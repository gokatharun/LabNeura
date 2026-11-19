#!/usr/bin/env python3
"""Validate and aggregate native-versus-Slang frontend benchmark JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ACCEPTED_CORRECTNESS = {"pass", "passed", "accepted", "agreed", "proven", "lowered"}
FRONTENDS = ("native", "slang")


class ComparisonError(ValueError):
    pass


def load_jsonl(paths: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ComparisonError(f"{path}:{number}: result must be an object")
            rows.append(value)
    return rows


def _correctness(row: Mapping[str, object]) -> str:
    value = row.get("correctness_verification_status",
                    row.get("correctness_status", row.get("correctness", "")))
    if isinstance(value, Mapping):
        value = value.get("status", "")
    return str(value).lower()


def _median(row: Mapping[str, object]) -> float | None:
    value = row.get("median_seconds")
    if value is None and isinstance(row.get("timing"), Mapping):
        value = row["timing"].get("median_seconds")  # type: ignore[index]
    if value is None:
        return None
    result = float(value)
    if result <= 0:
        raise ComparisonError("median_seconds must be positive when available")
    return result


def comparison_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    try:
        digest = row.get("workload_content_hash", row.get("content_hash_fnv1a64"))
        if digest is None:
            raise KeyError("workload_content_hash")
        return (str(row["category"]), str(digest),
                str(row.get("execution_backend", "cpu")))
    except KeyError as error:
        raise ComparisonError(f"missing comparison identity field: {error.args[0]}") from error


def _structure(row: Mapping[str, object]) -> dict[str, object | None]:
    return {
        "port_count": row.get("port_count", row.get("generated_port_count")),
        "net_count": row.get("net_count", row.get("generated_net_count")),
        "instance_count": row.get("instance_count", row.get("generated_instance_count")),
        "cell_types": row.get("cell_types"),
    }


def compare(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], dict[str, Mapping[str, object]]] = {}
    for row in rows:
        frontend = str(row.get("frontend", "")).lower()
        if frontend not in FRONTENDS:
            raise ComparisonError(f"unknown frontend: {frontend or '<missing>'}")
        key = comparison_key(row)
        if frontend in groups.setdefault(key, {}):
            raise ComparisonError(f"duplicate {frontend} result for {key}")
        groups[key][frontend] = row

    comparisons: list[dict[str, object]] = []
    for key, pair in sorted(groups.items()):
        missing = [frontend for frontend in FRONTENDS if frontend not in pair]
        if missing:
            raise ComparisonError(f"missing {', '.join(missing)} result for {key}")
        native, slang = pair["native"], pair["slang"]
        invalid = [frontend for frontend, row in pair.items()
                   if _correctness(row) not in ACCEPTED_CORRECTNESS]
        if invalid:
            raise ComparisonError(
                f"comparison {key} has unaccepted correctness for {', '.join(invalid)}")
        native_median, slang_median = _median(native), _median(slang)
        ratio = None if native_median is None or slang_median is None else (
            slang_median / native_median)
        native_structure, slang_structure = _structure(native), _structure(slang)
        comparisons.append({
            "category": key[0],
            "workload_content_hash": key[1],
            "execution_backend": key[2],
            "workload_identity": native.get("workload_identity",
                                               native.get("workload", key[1])),
            "native_median_seconds": native_median,
            "slang_median_seconds": slang_median,
            "native_speedup_ratio": ratio,
            "native_metrics": native.get("metrics"),
            "slang_metrics": slang.get("metrics"),
            "native_structure": native_structure,
            "slang_structure": slang_structure,
            "structurally_equal": native_structure == slang_structure,
        })
    return comparisons


def ranked_phase_contributions(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    measured: list[tuple[str, str, float]] = []
    for row in rows:
        phases = row.get("phase_timings_seconds")
        if not isinstance(phases, Mapping):
            continue
        frontend = str(row.get("frontend", "unknown"))
        for phase, duration in phases.items():
            if duration is not None:
                measured.append((frontend, str(phase), float(duration)))
    totals: dict[str, float] = {}
    for frontend, _phase, duration in measured:
        totals[frontend] = totals.get(frontend, 0.0) + duration
    result = [{
        "frontend": frontend,
        "phase": phase,
        "measured_seconds": duration,
        "percent_of_measured_phases": (duration / totals[frontend] * 100.0)
        if totals[frontend] else None,
        "evidence": "measured",
    } for frontend, phase, duration in measured]
    return sorted(result, key=lambda item: float(item["measured_seconds"]), reverse=True)


def _number(value: object, digits: int = 6) -> str:
    return "unavailable" if value is None else f"{float(value):.{digits}g}"


def markdown_report(comparisons: Sequence[Mapping[str, object]],
                    phases: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Native versus Slang frontend profiling",
        "",
        "Ratios are reported only for matched, correctness-accepted workload hashes.",
        "A ratio above 1 means native was faster. Unavailable metrics remain unavailable.",
        "",
        "## Comparative timing and scaling",
        "",
        "| Workload | Category | Backend | Native median (s) | Slang median (s) | Native speedup | Structure |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in comparisons:
        structure = "same" if row["structurally_equal"] else "different (see JSON)"
        lines.append(
            f"| {row['workload_identity']} | {row['category']} | {row['execution_backend']} "
            f"| {_number(row['native_median_seconds'])} | {_number(row['slang_median_seconds'])} "
            f"| {_number(row['native_speedup_ratio'], 4)}x | {structure} |")
    lines.extend(["", "## Ranked measured phase contributions", ""])
    if not phases:
        lines.append("No independently measured phase timings were available; no hotspot ranking is claimed.")
    else:
        lines.extend([
            "| Rank | Frontend | Phase | Measured seconds | Share of measured phases |",
            "|---:|---|---|---:|---:|",
        ])
        for rank, phase in enumerate(phases, 1):
            share = phase["percent_of_measured_phases"]
            lines.append(
                f"| {rank} | {phase['frontend']} | {phase['phase']} "
                f"| {_number(phase['measured_seconds'])} | {_number(share, 4)}% |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_jsonl(args.inputs)
    comparisons = compare(rows)
    phases = ranked_phase_contributions(rows)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"comparisons": comparisons, "phases": phases},
                                    indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(markdown_report(comparisons, phases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
