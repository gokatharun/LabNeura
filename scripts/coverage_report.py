#!/usr/bin/env python3
"""Collect native LabNeura coverage without mixing it with Release artifacts.

The input build must have ``LABNEURA_ENABLE_COVERAGE=ON``. LLVM profiles use
per-process filenames supplied by the caller, so parallel CTest execution
cannot overwrite a previous process's data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

EXCLUDE = r"/(tests|build|generated|third_party)/"


def run(command: list[str]) -> str:
    return subprocess.check_output(command, text=True)


def llvm_tools() -> tuple[str, str]:
    profdata = shutil.which("llvm-profdata")
    cov = shutil.which("llvm-cov")
    if not profdata and __import__("platform").system() == "Darwin":
        profdata = subprocess.check_output(["xcrun", "--find", "llvm-profdata"], text=True).strip()
    if not cov and __import__("platform").system() == "Darwin":
        cov = subprocess.check_output(["xcrun", "--find", "llvm-cov"], text=True).strip()
    if not profdata or not cov:
        raise RuntimeError("llvm-profdata and llvm-cov are required for Clang coverage")
    return profdata, cov


def native_binaries(build_dir: Path) -> list[Path]:
    suffix = ".exe" if __import__("os").name == "nt" else ""
    binaries = sorted(path for path in build_dir.rglob(f"labneura_*_test{suffix}")
                      if path.is_file())
    if not binaries:
        raise RuntimeError(f"no native CTest executables found under {build_dir}")
    return binaries


def llvm_coverage(build_dir: Path, profiles: Path, output: Path) -> dict:
    profdata, cov = llvm_tools()
    raw = sorted(profiles.glob("*.profraw"))
    if not raw:
        raise RuntimeError(f"no LLVM profile data was produced in {profiles}")
    merged = output / "native.profdata"
    subprocess.run([profdata, "merge", "-sparse", *map(str, raw), "-o", str(merged)], check=True)
    binaries = native_binaries(build_dir)
    objects = [item for binary in binaries for item in ("-object", str(binary))]
    exported = run([cov, "export", "-format=text", f"-instr-profile={merged}",
                    f"-ignore-filename-regex={EXCLUDE}", str(binaries[0]), *objects])
    payload = json.loads(exported)
    totals = payload.get("data", [{}])[0].get("totals", {})
    output.joinpath("native-coverage.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html = output / "native-html"
    subprocess.run([cov, "show", "-format=html", f"-instr-profile={merged}",
                    f"-output-dir={html}", f"-ignore-filename-regex={EXCLUDE}",
                    str(binaries[0]), *objects], check=True)
    return {name: value for name, value in totals.items() if name in {"lines", "functions", "branches"}}


def gcov_coverage(build_dir: Path, output: Path) -> dict:
    """Collect GCC coverage with gcovr, retaining the common summary schema."""
    gcovr = shutil.which("gcovr")
    if not gcovr:
        raise RuntimeError(
            "gcovr is required for GCC coverage; install it or configure with Clang"
        )
    root = Path(__file__).resolve().parents[1]
    summary = output / "gcovr-summary.json"
    (output / "native-html").mkdir(parents=True, exist_ok=True)
    subprocess.run([
        gcovr, "--root", str(root), "--filter", str(root / "src"),
        "--exclude", EXCLUDE, "--json-summary-pretty", "--output", str(summary),
        str(build_dir),
    ], check=True)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    subprocess.run([
        gcovr, "--root", str(root), "--filter", str(root / "src"),
        "--exclude", EXCLUDE, "--html-details", "--output",
        str(output / "native-html" / "index.html"), str(build_dir),
    ], check=True)
    return {
        "lines": {"covered": payload.get("line_covered", 0),
                  "count": payload.get("line_total", 0)},
        "functions": {"covered": payload.get("function_covered", 0),
                      "count": payload.get("function_total", 0)},
        "branches": {"covered": payload.get("branch_covered", 0),
                     "count": payload.get("branch_total", 0)},
    }


def percent(item: object) -> str:
    if not isinstance(item, dict) or not item.get("count"):
        return "N/A"
    return f"{100.0 * float(item.get('covered', 0)) / float(item['count']):.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--toolchain", choices=("auto", "llvm", "gcov"), default="auto")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    toolchain = args.toolchain
    if toolchain == "auto":
        toolchain = "llvm" if any(args.profiles.glob("*.profraw")) else "gcov"
    totals = (llvm_coverage(args.build_dir, args.profiles, args.output)
              if toolchain == "llvm" else gcov_coverage(args.build_dir, args.output))
    summary = {"schema_version": 1, "language": "C++", "totals": totals,
               "exclusions": ["tests", "build", "generated", "third_party"]}
    args.output.joinpath("native-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.joinpath("native-summary.md").write_text(
        "# Native C++ coverage\n\n"
        "| Metric | Coverage | Covered | Total |\n|---|---:|---:|---:|\n" +
        "\n".join(f"| {name} | {percent(value)} | {value.get('covered', 0)} | {value.get('count', 0)} |"
                  for name, value in sorted(totals.items())) + "\n",
        encoding="utf-8")
    print(args.output / "native-summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
