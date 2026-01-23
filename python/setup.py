from setuptools import setup, Extension
import sys
import os
import platform
from pathlib import Path
import sysconfig

try:
    import pybind11
except ImportError:
    print("pybind11 is required to build the extension; please install it first.")
    sys.exit(1)

def detect_simd_flags():
    """Return a list of compiler flags for SIMD support.
    
    IMPORTANT: Does NOT use -march=native to ensure wheel compatibility.
    Instead, uses conditional compilation (-mavx2, -msse4.1) with runtime
    CPU detection via cpuid to select the best backend at runtime.
    
    - On x86_64: enables AVX2 and SSE4.1 compilation (if supported by compiler)
    - On arm64/aarch64: NEON is generally enabled by default; no extra flags.
    """
    # Allow disabling SIMD flags via env for older toolchains
    if os.environ.get("LABNEURA_DISABLE_SIMD") == "1":
        return []

    flags = []
    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64"):
        # Common SIMD flags for x86_64 compilers (Clang/GCC). If unsupported, the build will ignore/handle.
        # Note: Using -msse4.1 instead of -msse4.2 for better compatibility
        flags.extend(["-mavx2", "-msse4.1"])  # CMake gates source; runtime CPU detection selects backend safely.
    return flags

# --------------------------------------
# Common configuration
# --------------------------------------
# Resolve paths robustly regardless of invocation cwd or layout (repo vs sdist)
repo_dir = Path(__file__).resolve().parent

# Prefer the directory that actually contains src/labneura/backends/
root_dir_candidates = [repo_dir, repo_dir.parent]
root_dir = None
for candidate in root_dir_candidates:
    if (candidate / "src/labneura/backends/cpu/avx2").exists():
        root_dir = candidate
        break

if root_dir is None:
    # Fall back to repo_dir to avoid breaking the build; error will surface naturally if files are missing
    root_dir = repo_dir

include_dirs = [
    pybind11.get_include(),
    str(root_dir / "include"),      # Top-level include directory
]

extra_compile_args = [
    "-O3",
    "-std=c++17",
    "-fvisibility=default",  # Ensure all symbols are exported (critical for Linux)
    *detect_simd_flags(),  # Add SIMD flags if supported
]

# Add aggressive optimization flags for target architecture
# Only when NOT building universal binary and NOT in coverage mode
is_coverage = os.environ.get("LABNEURA_COVERAGE") == "1"
arch = platform.machine().lower()

if not is_coverage:  # Skip architecture-specific flags in coverage mode
    # Common portable optimizations across architectures
    extra_compile_args.extend([
        "-funroll-loops",            # Unroll loops for performance
        "-fomit-frame-pointer",      # Extra register for computations
        "-fno-math-errno",           # Faster math without errno
    ])

    # Prefer compiler-aware vectorization flags
    cc = (sysconfig.get_config_var("CC") or "").lower()
    is_gcc = ("gcc" in cc) or ("g++" in cc)
    is_clang = ("clang" in cc)

    if is_gcc:
        # GCC: tree vectorizer (usually implied by -O3, but flag is accepted)
        extra_compile_args.append("-ftree-vectorize")
    # Clang auto-vectorization is enabled by -O3; avoid non-portable -fvectorize

extra_link_args = []

# --------------------------------------
# Optional LLVM coverage (enabled via env)
# --------------------------------------
if os.environ.get("LABNEURA_COVERAGE") == "1":
    extra_compile_args += [
        "-g",
        "-fprofile-instr-generate",
        "-fcoverage-mapping",
        "-fno-lto"
    ]
    extra_link_args += [
        "-fprofile-instr-generate",
        "-fcoverage-mapping",
        "-fno-lto",
        "-Wl"
    ]
    print("✓ LLVM coverage ENABLED")
else:
    print("✓ LLVM coverage DISABLED")

# Add -ffast-math for all platforms (unless coverage is enabled)
if os.environ.get("LABNEURA_COVERAGE") != "1":
    extra_compile_args.append("-ffast-math")

# --------------------------------------
# Extension module
# --------------------------------------
ext_modules = [
    Extension(
        name="labneura",
        sources=[
            "labneura_py.cpp",
            str(root_dir / "src/labneura/tensor.cpp"),
            str(root_dir / "src/labneura/backends/common/base.cpp"),
            str(root_dir / "src/labneura/backends/common/cpu_features.cpp"),
            str(root_dir / "src/labneura/backends/common/backend_factory.cpp"),
            # CPU Generic backend (modular)
            str(root_dir / "src/labneura/backends/cpu/generic/generic_backend.cpp"),
            str(root_dir / "src/labneura/backends/cpu/generic/generic_ops_fp32.cpp"),
            str(root_dir / "src/labneura/backends/cpu/generic/generic_ops_int8.cpp"),
            # CPU NEON backend (modular)
            str(root_dir / "src/labneura/backends/cpu/neon/neon_backend.cpp"),
            str(root_dir / "src/labneura/backends/cpu/neon/neon_ops_fp32.cpp"),
            str(root_dir / "src/labneura/backends/cpu/neon/neon_ops_int8.cpp"),
            str(root_dir / "src/labneura/backends/cpu/neon/neon_ops_int16.cpp"),
            # CPU AVX2 backend (modular)
            str(root_dir / "src/labneura/backends/cpu/avx2/avx2_backend.cpp"),
            str(root_dir / "src/labneura/backends/cpu/avx2/avx2_ops_fp32.cpp"),
            str(root_dir / "src/labneura/backends/cpu/avx2/avx2_ops_int8.cpp"),
            str(root_dir / "src/labneura/backends/cpu/avx2/avx2_ops_fp16.cpp"),
            str(root_dir / "src/labneura/backends/cpu/avx2/avx2_ops_int16.cpp"),
        ],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

# --------------------------------------
# Setup
# --------------------------------------
# Read long description from top-level README.md
readme_path = Path(__file__).resolve().parent.parent / "README.md"
long_desc = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "LabNeura: Python bindings for a SIMD-accelerated tensor backend."

setup(
    name="labneura2",
    version="0.2.6",
    description="SIMD-accelerated tensor operations for neural networks",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    author="LabNeura Authors",
    author_email="",
    url="https://github.com/gokatharun/LabNeura",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=ext_modules,
    include_package_data=True,
    zip_safe=False,
)