#!/usr/bin/env python3
"""Export benchmark results to CSV with separated ratios and percentages."""

import csv
import json
from pathlib import Path
from typing import Dict, Tuple

RESULTS_FILE = Path(__file__).parent / "benchmark_results.json"
OUTPUT_DIR = Path(__file__).parent / "csv"


def load_results() -> Dict:
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)


def ensure_outdir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def ratio_and_pct(num: float, den: float, higher_is_better: bool = True) -> Tuple[float, float]:
    if den <= 0 or num <= 0:
        return 0.0, 0.0
    r = (num / den) if higher_is_better else (den / num)
    pct = (r - 1.0) * 100.0
    return r, pct


def export_flops(results: Dict):
    out = ensure_outdir() / "flops_summary.csv"
    fields = [
        "operation",
        "size",
        "labneura_flops",
        "numpy_flops",
        "pytorch_flops",
        "ln_np_x",
        "ln_np_pct",
        "ln_pt_x",
        "ln_pt_pct",
        "winner",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for size, ops in results.items():
            for op, data in ops.items():
                ln = data["labneura"]["flops"]
                npv = data["numpy"]["flops"]
                pt = data["pytorch"]["flops"]
                ln_np_x, ln_np_pct = ratio_and_pct(ln, npv, True)
                ln_pt_x, ln_pt_pct = ratio_and_pct(ln, pt, True)
                winner = max({"LabNeura": ln, "NumPy": npv, "PyTorch": pt}, key=lambda k: {"LabNeura": ln, "NumPy": npv, "PyTorch": pt}[k])
                writer.writerow({
                    "operation": op,
                    "size": int(size),
                    "labneura_flops": ln,
                    "numpy_flops": npv,
                    "pytorch_flops": pt,
                    "ln_np_x": round(ln_np_x, 4),
                    "ln_np_pct": round(ln_np_pct, 2),
                    "ln_pt_x": round(ln_pt_x, 4),
                    "ln_pt_pct": round(ln_pt_pct, 2),
                    "winner": winner,
                })
    print(f"✓ FLOPS summary -> {out}")


def export_latency(results: Dict):
    out = ensure_outdir() / "latency_summary.csv"
    fields = [
        "operation",
        "size",
        "labneura_ms",
        "numpy_ms",
        "pytorch_ms",
        "ln_np_x",
        "ln_np_pct",
        "ln_pt_x",
        "ln_pt_pct",
        "winner",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for size, ops in results.items():
            for op, data in ops.items():
                ln = data["labneura"].get("total_time_ms")
                npv = data["numpy"].get("total_time_ms")
                pt = data["pytorch"].get("total_time_ms")
                if ln is None or npv is None or pt is None:
                    continue
                ln_np_x, ln_np_pct = ratio_and_pct(ln, npv, higher_is_better=False)
                ln_pt_x, ln_pt_pct = ratio_and_pct(ln, pt, higher_is_better=False)
                winner = min({"LabNeura": ln, "NumPy": npv, "PyTorch": pt}, key=lambda k: {"LabNeura": ln, "NumPy": npv, "PyTorch": pt}[k])
                writer.writerow({
                    "operation": op,
                    "size": int(size),
                    "labneura_ms": ln,
                    "numpy_ms": npv,
                    "pytorch_ms": pt,
                    "ln_np_x": round(ln_np_x, 4),
                    "ln_np_pct": round(ln_np_pct, 2),
                    "ln_pt_x": round(ln_pt_x, 4),
                    "ln_pt_pct": round(ln_pt_pct, 2),
                    "winner": winner,
                })
    print(f"✓ Latency summary -> {out}")


def export_throughput(results: Dict):
    out = ensure_outdir() / "throughput_summary.csv"
    fields = [
        "operation",
        "size",
        "labneura_tput_els_per_s",
        "numpy_tput_els_per_s",
        "pytorch_tput_els_per_s",
        "ln_np_x",
        "ln_np_pct",
        "ln_pt_x",
        "ln_pt_pct",
        "winner",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for size, ops in results.items():
            for op, data in ops.items():
                ln_ms = data["labneura"].get("total_time_ms")
                np_ms = data["numpy"].get("total_time_ms")
                pt_ms = data["pytorch"].get("total_time_ms")
                if ln_ms is None or np_ms is None or pt_ms is None:
                    continue
                ln_time = ln_ms / 1000.0
                np_time = np_ms / 1000.0
                pt_time = pt_ms / 1000.0
                iterations = data.get("labneura", {}).get("iterations") or data.get("iterations") or 100
                n = int(size) * iterations
                ln_tput = n / ln_time if ln_time > 0 else 0
                np_tput = n / np_time if np_time > 0 else 0
                pt_tput = n / pt_time if pt_time > 0 else 0
                ln_np_x, ln_np_pct = ratio_and_pct(ln_tput, np_tput, higher_is_better=True)
                ln_pt_x, ln_pt_pct = ratio_and_pct(ln_tput, pt_tput, higher_is_better=True)
                winner = max({"LabNeura": ln_tput, "NumPy": np_tput, "PyTorch": pt_tput}, key=lambda k: {"LabNeura": ln_tput, "NumPy": np_tput, "PyTorch": pt_tput}[k])
                writer.writerow({
                    "operation": op,
                    "size": int(size),
                    "labneura_tput_els_per_s": ln_tput,
                    "numpy_tput_els_per_s": np_tput,
                    "pytorch_tput_els_per_s": pt_tput,
                    "ln_np_x": round(ln_np_x, 4),
                    "ln_np_pct": round(ln_np_pct, 2),
                    "ln_pt_x": round(ln_pt_x, 4),
                    "ln_pt_pct": round(ln_pt_pct, 2),
                    "winner": winner,
                })
    print(f"✓ Throughput summary -> {out}")


def main():
    results = load_results()
    export_flops(results)
    export_latency(results)
    export_throughput(results)


if __name__ == "__main__":
    main()
