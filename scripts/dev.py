"""Portable developer commands used by Make and CI."""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = ROOT / "build" / "dev"
COMPONENTS = ("all", "core", "tensor", "eda")
TARGETS = ("native", "arm64", "x86_64", "windows-x86_64", "all")


@dataclass(frozen=True)
class Resources:
    cpus: int
    memory_bytes: int
    compile_jobs: int
    link_jobs: int


def available_memory() -> int:
    system = platform.system()
    if system == "Linux":
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
    elif system == "Darwin":
        try:
            return int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True,
                stderr=subprocess.DEVNULL,
            ))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    elif system == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                        ("available_extended", ctypes.c_ulonglong)]
        status = MemoryStatus(); status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.available)
    return 4 * 1024**3


def detect_resources(explicit_jobs: int | None = None) -> Resources:
    cpus = max(1, os.cpu_count() or 1)
    memory = max(1024**3, available_memory())
    override = explicit_jobs or int(os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "0") or 0)
    compile_jobs = override if override > 0 else min(cpus, max(1, memory // (1536 * 1024**2)))
    link_jobs = max(1, min(compile_jobs, memory // (3 * 1024**3)))
    return Resources(cpus, memory, max(1, compile_jobs), link_jobs)


def component_args(component: str) -> list[str]:
    accel, tensor, eda, compat = {
        "all": ("ON", "ON", "ON", "ON"), "core": ("ON", "OFF", "OFF", "OFF"),
        "tensor": ("ON", "ON", "OFF", "OFF"), "eda": ("OFF", "OFF", "ON", "OFF"),
    }[component]
    return [f"-DLABNEURA_BUILD_CORE={accel}", f"-DLABNEURA_BUILD_ACCEL_CORE={accel}", f"-DLABNEURA_BUILD_TENSOR={tensor}",
            f"-DLABNEURA_BUILD_EDA={eda}", f"-DLABNEURA_BUILD_COMPAT={compat}",
            "-DLABNEURA_ENABLE_SLANG_VALIDATION=OFF", "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
            "-DBUILD_TESTING=ON"]


def target_args(target: str) -> list[str]:
    system, machine = platform.system(), platform.machine().lower()
    if target == "native": return []
    if target in {"arm64", "x86_64"}:
        if system == "Darwin": return [f"-DCMAKE_OSX_ARCHITECTURES={target}"]
        native = "arm64" if machine in {"arm64", "aarch64"} else "x86_64"
        if target == native: return []
        variable = f"LABNEURA_{target.upper()}_TOOLCHAIN".replace("-", "_")
        if os.environ.get(variable): return [f"-DCMAKE_TOOLCHAIN_FILE={os.environ[variable]}"]
    if target == "windows-x86_64":
        if system == "Windows": return ["-A", "x64"]
        if os.environ.get("LABNEURA_WINDOWS_X86_64_TOOLCHAIN"):
            return [f"-DCMAKE_TOOLCHAIN_FILE={os.environ['LABNEURA_WINDOWS_X86_64_TOOLCHAIN']}"]
    raise RuntimeError(f"target {target!r} unavailable on {system}/{machine}; configure its toolchain environment variable")


class Commands:
    def __init__(self, args: argparse.Namespace): self.args, self.resources = args, detect_resources(args.jobs)
    @property
    def build_dir(self) -> Path: return BUILD_ROOT / self.args.target / self.args.component
    def run(self, command: list[str], env: dict[str, str] | None = None) -> None:
        print("+", subprocess.list2cmdline(command), flush=True)
        environment = os.environ.copy()
        cache = BUILD_ROOT / ".ccache"
        cache.mkdir(parents=True, exist_ok=True)
        environment.setdefault("CCACHE_DIR", str(cache))
        environment.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", str(self.resources.compile_jobs))
        environment.setdefault("PYTHONPYCACHEPREFIX", str(BUILD_ROOT / ".pycache"))
        if env:
            environment.update(env)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    def configure(self) -> None:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.run(["cmake", "-S", str(ROOT), "-B", str(self.build_dir),
                  f"-DCMAKE_BUILD_TYPE={self.args.build_type}", *component_args(self.args.component),
                  *target_args(self.args.target)])
    def compile(self) -> None:
        self.configure()
        print(f"resources: cpus={self.resources.cpus} memory_gib={self.resources.memory_bytes / 1024**3:.1f} compile_jobs={self.resources.compile_jobs} link_jobs={self.resources.link_jobs}")
        self.run(["cmake", "--build", str(self.build_dir), "--parallel", str(self.resources.compile_jobs)])
        help_text = subprocess.check_output(["cmake", "--build", str(self.build_dir), "--target", "help"], cwd=ROOT, text=True)
        targets = []
        if self.args.component in {"all", "eda"}: targets += ["labneura_lexer_benchmark", "labneura_preprocessor_benchmark"]
        if self.args.component in {"all", "tensor"}: targets += ["labneura_native_backend_benchmark"]
        for target in targets:
            if target in help_text:
                self.run(["cmake", "--build", str(self.build_dir), "--target", target, "--parallel", str(self.resources.compile_jobs)])
        if self.args.target == "native":
            wheelhouse = self.build_dir / "python-dist"
            wheelhouse.mkdir(exist_ok=True)
            packages = {
                "core": ("core",), "tensor": ("tensor",), "eda": ("eda",),
                "all": ("core", "tensor", "eda"),
            }[self.args.component]
            for package in packages:
                self.run([
                    sys.executable, "-m", "build", "--wheel", "--no-isolation",
                    "--outdir", str(wheelhouse), str(ROOT / "packaging" / package),
                ])
    def test(self) -> None:
        self.compile()
        self.run(["ctest", "--test-dir", str(self.build_dir), "--output-on-failure", "--no-tests=error"])
        selected = {"eda": ["tests/test_eda.py"], "tensor": ["tests/test_tensor.py"],
                    "core": ["tests/python_packaging/test_core_package.py", "tests/test_version.py"],
                    "all": ["tests"]}[self.args.component]
        self.run([sys.executable, "-m", "pytest", "-q", *selected])
        print("summary: PASS (native CTest and Python regressions)")
    def check(self) -> None:
        self.run(["git", "diff", "--check"]); self.run([sys.executable, "scripts/update_version.py", "--check"])
        self.run([sys.executable, "scripts/check_architecture.py"])
        self.run([sys.executable, "-m", "compileall", "-q", "scripts"]); print("summary: PASS (source integrity and architecture)")
    def benchmark(self) -> None:
        self.compile()
        if self.args.component in {"all", "eda"}:
            suffix = ".exe" if os.name == "nt" else ""
            self.run([str(self.build_dir / f"labneura_lexer_benchmark{suffix}"), "--synthetic",
                      "tests/data/verilog/lexer_corpus.sv", "examples/verilog/ibex/ibex_alu.sv",
                      "examples/verilog/8080_core/i8080.v"])
        print("summary: PASS (available benchmarks)")
    def coverage(self) -> None:
        coverage_dir = BUILD_ROOT / "coverage" / self.args.component
        profiles = coverage_dir / "profiles"
        profiles.mkdir(parents=True, exist_ok=True)
        self.run(["cmake", "-S", str(ROOT), "-B", str(coverage_dir / "native"),
                  "-DCMAKE_BUILD_TYPE=Debug", *component_args(self.args.component),
                  "-DLABNEURA_ENABLE_COVERAGE=ON"])
        self.run(["cmake", "--build", str(coverage_dir / "native"), "--parallel",
                  str(self.resources.compile_jobs)])
        self.run(["ctest", "--test-dir", str(coverage_dir / "native"),
                  "--output-on-failure", "--no-tests=error"],
                 {"LLVM_PROFILE_FILE": str(profiles / "%p-%m.profraw")})
        self.run([sys.executable, "scripts/coverage_report.py", "--build-dir",
                  str(coverage_dir / "native"), "--profiles", str(profiles),
                  "--output", str(coverage_dir / "native-report")])
        selected = {"eda": ["tests/test_eda.py"], "tensor": ["tests/test_tensor.py"],
                    "core": ["tests/test_version.py"], "all": ["tests"]}[self.args.component]
        self.run([sys.executable, "-m", "pytest", "-q", *selected])
        self.run([sys.executable, "scripts/write_coverage_summary.py", "--native",
                  str(coverage_dir / "native-report" / "native-summary.json"), "--output",
                  str(ROOT / "COVERAGE.md")])
    def format(self) -> None:
        formatter = shutil.which("clang-format")
        if not formatter: raise RuntimeError("clang-format is required for make format")
        changed = subprocess.check_output(["git", "diff", "--name-only", "--diff-filter=ACMRTUXB"], cwd=ROOT, text=True).splitlines()
        sources = [name for name in changed if Path(name).suffix in {".c", ".cc", ".cpp", ".h", ".hpp"}]
        if sources: self.run([formatter, "-i", *sources])
        print(f"summary: PASS ({len(sources)} changed C/C++ files formatted)")
    def clean(self) -> None:
        target = self.build_dir.resolve()
        if BUILD_ROOT.resolve() not in target.parents: raise RuntimeError(f"unsafe clean path: {target}")
        if target.exists(): shutil.rmtree(target)
        print(f"removed {target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LabNeura portable developer commands")
    parser.add_argument("--component", choices=COMPONENTS, default="all"); parser.add_argument("--target", choices=TARGETS, default="native")
    parser.add_argument("--build-type", choices=("Debug", "Release", "RelWithDebInfo"), default="Release"); parser.add_argument("--jobs", type=int)
    parser.add_argument("command", choices=("compile", "test", "benchmark", "coverage", "check", "format", "clean")); return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.jobs is not None and args.jobs < 1: raise RuntimeError("--jobs must be positive")
    targets = [args.target]
    if args.target == "all":
        targets = ["native", *filter(None, os.environ.get("LABNEURA_CROSS_TARGETS", "").split(","))]
        if targets == ["native"]: print("TARGET=all: no configured cross targets; building native only")
    for target in targets:
        child = argparse.Namespace(**vars(args)); child.target = target
        getattr(Commands(child), args.command)()
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr); raise SystemExit(1)
