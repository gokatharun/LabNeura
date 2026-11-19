#!/usr/bin/env python3
"""Collect optional OS profiling evidence for the EDA frontend benchmark.

The benchmark executable remains responsible for steady-state timing and
correctness.  This wrapper deliberately runs timing, memory, counters, and
sampling as separate processes so profiler overhead cannot contaminate timing
results.  Missing tools and inaccessible counters are recorded as unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence


PERF_EVENTS = (
    "cycles",
    "instructions",
    "branches",
    "branch-misses",
    "cache-references",
    "cache-misses",
    "page-faults",
    "context-switches",
    "task-clock",
)


def unavailable(reason: str) -> dict[str, object]:
    return {"status": "unavailable", "value": None, "reason": reason}


def available(value: object, unit: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {"status": "available", "value": value}
    if unit:
        result["unit"] = unit
    return result


def peak_rss_command(system: str, command: Sequence[str]) -> list[str] | None:
    time_tool = "/usr/bin/time"
    if not Path(time_tool).exists():
        return None
    if system == "Darwin":
        return [time_tool, "-l", *command]
    if system == "Linux":
        return [time_tool, "-v", *command]
    return None


def perf_stat_command(command: Sequence[str], output: Path) -> list[str]:
    return [
        "perf", "stat", "-x", ",", "--no-big-num", "-o", str(output),
        "-e", ",".join(PERF_EVENTS), "--", *command,
    ]


def perf_record_command(command: Sequence[str], output: Path) -> list[str]:
    return ["perf", "record", "-g", "-o", str(output), "--", *command]


def xctrace_command(command: Sequence[str], output: Path,
                    template: str = "Time Profiler") -> list[str]:
    return [
        "xcrun", "xctrace", "record", "--template", template,
        "--output", str(output), "--launch", "--", *command,
    ]


def sample_command(pid: int, output: Path, duration_seconds: int = 1) -> list[str]:
    return ["sample", str(pid), str(duration_seconds), "-file", str(output)]


def parse_perf_stat(text: str) -> dict[str, dict[str, object]]:
    metrics = {event: unavailable("counter was not reported") for event in PERF_EVENTS}
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        raw_value, _unit, event = fields[:3]
        if event not in metrics:
            continue
        if not raw_value or "not supported" in raw_value or "not counted" in raw_value:
            metrics[event] = unavailable(raw_value or "counter unavailable")
            continue
        try:
            value: object = float(raw_value) if "." in raw_value else int(raw_value)
        except ValueError:
            metrics[event] = unavailable(f"unparseable counter value: {raw_value}")
        else:
            metrics[event] = available(value, "msec" if event == "task-clock" else "count")
    return metrics


def parse_peak_rss(system: str, stderr: str) -> dict[str, object]:
    if system == "Darwin":
        match = re.search(r"^\s*(\d+)\s+maximum resident set size\s*$", stderr, re.MULTILINE)
        return available(int(match.group(1)), "bytes") if match else unavailable(
            "macOS time output did not contain maximum resident set size")
    if system == "Linux":
        match = re.search(
            r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr)
        return available(int(match.group(1)), "KiB") if match else unavailable(
            "GNU time output did not contain maximum resident set size")
    return unavailable(f"peak RSS collection is unsupported on {system}")


def _run(command: Sequence[str], stdout_path: Path, stderr_path: Path) -> dict[str, object]:
    started = time.monotonic_ns()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "wall_time_ns": time.monotonic_ns() - started,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def _run_sample(command: Sequence[str], output_dir: Path) -> dict[str, object]:
    stdout_path = output_dir / "sample-target.stdout.txt"
    stderr_path = output_dir / "sample-target.stderr.txt"
    artifact = output_dir / "sample.txt"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        sampler = subprocess.run(sample_command(process.pid, artifact),
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.PIPE, check=False)
        target_returncode = process.wait()
    return {
        "status": "available" if sampler.returncode == 0 and artifact.exists()
        else "unavailable",
        "artifact": str(artifact) if artifact.exists() else None,
        "target_returncode": target_returncode,
        "sampler_returncode": sampler.returncode,
        "sampler_stderr": sampler.stderr.decode(errors="replace"),
    }


def _tool(name: str) -> str | None:
    return shutil.which(name)


def collect(command: Sequence[str], output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    system = platform.system()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "platform": {"system": system, "machine": platform.machine()},
        "benchmark_command": list(command),
        "separate_runs": True,
        "timing": _run(command, output_dir / "timing.stdout.jsonl",
                       output_dir / "timing.stderr.txt"),
    }

    rss_command = peak_rss_command(system, command)
    if rss_command is None:
        manifest["peak_rss"] = unavailable("/usr/bin/time or platform support is unavailable")
    else:
        run = _run(rss_command, output_dir / "rss.stdout.txt", output_dir / "rss.stderr.txt")
        stderr = (output_dir / "rss.stderr.txt").read_text(errors="replace")
        manifest["peak_rss"] = {"run": run, "metric": parse_peak_rss(system, stderr)}

    if system == "Linux":
        if not _tool("perf"):
            manifest["hardware_counters"] = unavailable("perf is not installed")
            manifest["sampling_profile"] = unavailable("perf is not installed")
        else:
            stat_file = output_dir / "perf-stat.csv"
            run = _run(perf_stat_command(command, stat_file),
                       output_dir / "perf-stat.stdout.txt",
                       output_dir / "perf-stat.stderr.txt")
            if stat_file.exists():
                counters = parse_perf_stat(stat_file.read_text(errors="replace"))
            else:
                counters = {event: unavailable("perf stat produced no output")
                            for event in PERF_EVENTS}
            manifest["hardware_counters"] = {"run": run, "metrics": counters}
            profile = output_dir / "perf.data"
            run = _run(perf_record_command(command, profile),
                       output_dir / "perf-record.stdout.txt",
                       output_dir / "perf-record.stderr.txt")
            manifest["sampling_profile"] = {
                "status": "available" if run["returncode"] == 0 and profile.exists()
                else "unavailable",
                "artifact": str(profile) if profile.exists() else None,
                "run": run,
            }
    elif system == "Darwin":
        if not _tool("xcrun"):
            manifest["sampling_profile"] = unavailable("xcrun/xctrace is not installed")
        else:
            trace = output_dir / "time-profile.trace"
            run = _run(xctrace_command(command, trace),
                       output_dir / "xctrace.stdout.txt",
                       output_dir / "xctrace.stderr.txt")
            time_profile = {
                "status": "available" if run["returncode"] == 0 and trace.exists()
                else "unavailable",
                "artifact": str(trace) if trace.exists() else None,
                "run": run,
            }
            allocations = output_dir / "allocations.trace"
            allocation_run = _run(xctrace_command(command, allocations, "Allocations"),
                                  output_dir / "allocations.stdout.txt",
                                  output_dir / "allocations.stderr.txt")
            manifest["sampling_profile"] = {
                "xctrace_time_profiler": time_profile,
                "xctrace_allocations": {
                    "status": "available" if allocation_run["returncode"] == 0
                    and allocations.exists() else "unavailable",
                    "artifact": str(allocations) if allocations.exists() else None,
                    "run": allocation_run,
                },
                "sample": _run_sample(command, output_dir) if _tool("sample")
                else unavailable("sample is not installed"),
            }
        manifest["hardware_counters"] = unavailable(
            "portable unprivileged hardware counters are not exposed by this script on macOS")
    else:
        manifest["hardware_counters"] = unavailable(f"unsupported platform: {system}")
        manifest["sampling_profile"] = unavailable(f"unsupported platform: {system}")

    manifest_path = output_dir / "profile-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="benchmark executable and arguments, after --")
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a benchmark command is required after --")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = collect(args.command, args.output_dir)
    timing = manifest["timing"]
    return int(timing["returncode"])


if __name__ == "__main__":
    sys.exit(main())
