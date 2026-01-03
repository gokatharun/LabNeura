from setuptools import setup, Extension
import sys
import os

try:
    import pybind11
except ImportError:
    print("pybind11 is required to build the extension; please install it first.")
    sys.exit(1)

# --------------------------------------
# Common configuration
# --------------------------------------
include_dirs = [
    pybind11.get_include(),
    os.path.abspath("../include"),
]

extra_compile_args = [
    "-O3",
    "-std=c++17",
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
        os.popen(
                "ls $(xcrun --show-sdk-path)/usr/lib/clang/*/lib/darwin/libclang_rt.profile_osx.a"
            ).read().strip()
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
            "../src/labneura/tensor.cpp",
            "../src/labneura/backends/base.cpp",
            "../src/labneura/backends/generic.cpp",
            "../src/labneura/backends/neon.cpp",
            "../src/labneura/backends/avx2.cpp",
            "../src/labneura/backends/sse41.cpp",
            "../src/labneura/backends/cpu_features.cpp",
            "../src/labneura/backends/backend_factory.cpp",
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
setup(
    name="labneura",
    version="0.1",
    description="LabNeura Python bindings with SIMD-accelerated Tensor backend",
    ext_modules=ext_modules,
    zip_safe=False,
)