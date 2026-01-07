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
    """Return a list of compiler flags based on build machine architecture.
    - On x86_64: attempt to enable AVX2 and SSE4.2.
    - On arm64/aarch64: NEON is generally enabled by default; no extra flags.
    """
    flags = []
    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64"):
        # Common SIMD flags for x86_64 compilers (Clang/GCC). If unsupported, the build will ignore/handle.
        flags.extend(["-mavx2", "-msse4.2"])  # CMake gates AVX2 source; runtime selects backend safely.
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

# --------------------------------------
# Platform-specific flags
# --------------------------------------
if sys.platform == "darwin":
    # macOS (Clang)
    extra_compile_args += [
        "-ffast-math",
    ]
elif sys.platform.startswith("linux"):
    # Linux (Clang/GCC)
    extra_compile_args += [
        "-ffast-math",
    ]

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
    version="0.1.6",
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