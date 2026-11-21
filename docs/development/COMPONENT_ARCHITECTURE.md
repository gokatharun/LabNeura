# Native component architecture

LabNeura has three native ownership domains and three Python distributions.
They are related but not identical.

## Native dependency graph

```text
labneura_core
    └── labneura_tensor

labneura_eda  (independent of Core and Tensor)

labneura      (compatibility archive reusing the three object libraries)
```

Core owns CPU capability detection and domain-independent platform utilities.
Tensor owns storage, runtime, execution, dispatch, and numerical kernels under
`src/labneura/tensor/core`. EDA owns its parser, analysis, lowering, netlist,
and mapping pipeline under `src/labneura/eda` and does not link Tensor.

The canonical public include roots are `labneura/core`, `labneura/tensor`, and
`labneura/eda`. Historical Tensor/backend include paths remain forwarding
headers. `labneura_accel_core` and `LABNEURA_BUILD_ACCEL_CORE` remain build
compatibility aliases; new code should use `labneura_core` and
`LABNEURA_BUILD_CORE`.

Component source inventories live in `cmake/components`. The root project owns
global policies, component selection, shared optimization checks, packaging
bindings, installation, and export coordination. Object libraries are reused
by component archives and the compatibility archive, so a source is compiled
once per compatible configuration.

## Packaging

`labneura-tensor` and `labneura-eda` build directly from the authoritative
repository `src/` and `include/` trees. The pure `labneura` umbrella owns the
Python initializer and depends on both native distributions. There is no
Python Core distribution and no maintained `python/src` or `python/include`
mirror. Source distributions stage temporary copies from the authoritative
tree during construction.

## Enforcement

`python scripts/check_architecture.py` rejects cross-domain dependencies,
Tensor kernels outside Tensor Core, domain algorithms in Core, duplicate native
Python mirrors, missing CMake registration, and target-wide AVX/F16C flags.
It is part of `make check`.

Portable presets are available through `cmake --preset` for `core`, `tensor`,
`eda`, `development`, `release`, `benchmark`, `coverage`, and `sanitizer`.
Build parallelism remains a caller/build-tool choice; `scripts/dev.py` derives
it from available CPUs and memory while respecting explicit overrides.
