import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_performance_report_marks_missing_baselines_noncomparable(tmp_path):
    tensor = tmp_path / "tensor.csv"
    tensor.write_text("architecture,backend,dtype,operation,size,median_ns,p95_ns\narm64,GENERIC,FP32,ADD,1048576,1000,1100\n")
    lexer = tmp_path / "lexer.log"
    lexer.write_text("case=lexer_end_to_end backend=scalar median_MB_per_second=12.5\n")
    output = tmp_path / "PERFORMANCE.md"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "performance_report.py"),
                    "--tensor", str(tensor), "--lexer", str(lexer), "--output", str(output)], check=True)
    text = output.read_text(encoding="utf-8")
    assert "Tensor GENERIC FP32 ADD" in text
    assert "EDA lexer scalar" in text
    assert "N/C" in text
