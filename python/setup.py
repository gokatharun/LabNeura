from setuptools import setup, Extension
import sys
import os

try:
    import pybind11
except Exception:
    print("pybind11 is required to build the extension; please install it first.")
    sys.exit(1)

# Configure OpenMP with libomp from Homebrew
include_dirs = [pybind11.get_include(), '../include']
extra_compile_args = ['-O3', '-std=c++17','-fprofile-instr-generate', '-fcoverage-mapping']
extra_link_args = ['-fprofile-instr-generate', '-fcoverage-mapping', '-Wl']
library_dirs = []

# Find libomp
libomp_path = None
if os.path.exists('/opt/homebrew/opt/libomp'):
    libomp_path = '/opt/homebrew/opt/libomp'
elif os.path.exists('/usr/local/opt/libomp'):
    libomp_path = '/usr/local/opt/libomp'

if libomp_path:
    print(f"Found libomp at: {libomp_path}")
    include_dirs.append(os.path.join(libomp_path, 'include'))
    library_dirs.append(os.path.join(libomp_path, 'lib'))
    extra_compile_args.extend(['-Xclang', '-fopenmp', '-ffast-math' ])
    extra_link_args.append('-lomp')
    print("✓ OpenMP (libomp) support ENABLED")
else:
    print("⚠ libomp not found - install with: brew install libomp")
    print("⚠ Building without OpenMP support (single-threaded)")

ext_modules = [
    Extension(
        'labneura',
        sources=['labneura_py.cpp', '../src/labneura/tensor.cpp', '../src/labneura/backends/generic.cpp', '../src/labneura/backends/neon.cpp'],
        include_dirs=include_dirs,
        library_dirs=library_dirs if library_dirs else None,
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name='labneura',
    version='0.1',
    description='LabNeura Python bindings with Tensor support',
    ext_modules=ext_modules,
)
