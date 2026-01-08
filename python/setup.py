from setuptools import setup, Extension
import sys
import os
import platform
from pathlib import Path

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
include_dirs = [
    pybind11.get_include(),
    os.path.abspath("src"),
    os.path.abspath("include"),
]

extra_compile_args = [
    "-O3",
    "-std=c++17",
    *detect_simd_flags(),  # Add SIMD flags if supported
]

# Add aggressive optimization flags for target architecture
# Only when NOT building universal binary and NOT in coverage mode
is_coverage = os.environ.get("LABNEURA_COVERAGE") == "1"
arch = platform.machine().lower()

if not is_coverage:  # Skip architecture-specific flags in coverage mode
    if arch in ("arm64", "aarch64"):
        # ARM-specific optimizations (portable for PyPI wheels)
        extra_compile_args.extend([
            "-fvectorize",               # Enable auto-vectorization
            "-funroll-loops",            # Unroll loops for performance
            "-fomit-frame-pointer",      # Extra register for computations
            "-fno-math-errno",           # Faster math without errno
        ])
        # Note: -mcpu=apple-m1 and -mcpu=native are NOT used for wheel portability
        # Runtime NEON detection handles Apple Silicon optimization
    elif arch in ("x86_64", "amd64"):
        # Intel/AMD specific optimizations (portable for PyPI)
        extra_compile_args.extend([
            "-fvectorize",               # Enable auto-vectorization
            "-funroll-loops",            # Unroll loops
            "-fomit-frame-pointer",      # Extra register
            # Note: -march=native is NOT used for wheel distribution
            # Runtime CPU detection handles architecture selection
        ])

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
            "src/labneura/tensor.cpp",
            "src/labneura/backends/base.cpp",
            "src/labneura/backends/generic.cpp",
            "src/labneura/backends/neon.cpp",
            "src/labneura/backends/avx2.cpp",
            "src/labneura/backends/sse41.cpp",
            "src/labneura/backends/cpu_features.cpp",
            "src/labneura/backends/backend_factory.cpp",
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
    version="0.1.7",
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