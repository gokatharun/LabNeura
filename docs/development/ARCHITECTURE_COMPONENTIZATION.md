# Native component architecture

## Pre-migration inventory

The ARCH-1B baseline, captured before physical moves, contained:

- 66 production C/C++ source files under `src/labneura`;
- 117 native/Python test source files;
- 8 C++/Python benchmark sources;
- 75 explicit CTest registrations and 62 test/benchmark executable declarations
  in the monolithic root CMake file;
- complete tracked native mirrors under `python/src` and `python/include`.

The post-migration inventory must retain or intentionally account for these
tests and benchmarks. A smaller source count is acceptable only when duplicate
mirrors or obsolete implementations are removed; registered tests must not be
silently dropped.

## Dependency direction

```text
labneura_core
    ^
    |
labneura_tensor_core -> labneura_tensor

labneura_eda (independent of Tensor; may use domain-independent Core)
```

Core owns CPU/platform/memory utilities. Tensor Core owns storage, runtime,
execution, dispatch, and numerical kernels. EDA owns its frontend, analysis,
lowering, netlist, and mapping implementations. Tensor and EDA cannot depend
on each other.

`python/src` and `python/include` are not authoritative trees. Python wheels and
source distributions stage `src` and `include`; generated staging copies are
temporary build products.

## Automated enforcement

Run:

```bash
python scripts/check_architecture.py
python -m pytest tests/architecture
```

The checks reject cross-domain includes, numerical kernels outside Tensor Core,
domain algorithms in Core, duplicate Python native trees, missing component
CMake source registration, and target-wide AVX2/F16C flags. Compatibility
forwarding headers are allowed because they contain declarations/includes but
no independently maintained implementation.
