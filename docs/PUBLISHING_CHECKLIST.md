# LabNeura Publishing Checklist

## ✅ Modular Backend Integration Complete

All backend files have been successfully modularized and integrated into the Python packaging system.

## Package Configuration Status

## Wheel publishing

GitHub Actions publishes to PyPI when a GitHub Release is published. The
release tag (`X.Y.Z` or `vX.Y.Z`) determines the package version; after
publishing, the workflow opens a pull request to sync that version into the
default branch.

Local publishing automatically increments the patch version and pushes a
version-only commit before upload. To publish a specific version instead, pass
`VERSION=X.Y.Z` to the make target. Version declarations must not have local
uncommitted edits before running a publish target.

Source builds using GCC require GCC 13 or newer. CMake rejects older GCC
versions with an actionable error. Run the test suite manually before
publishing.

### Files Updated for Modularization

- ✅ **CMakeLists.txt** - C++ library and Python extension build configuration
- ✅ **pyproject.toml** - Root Python package metadata and scikit-build-core backend
- ✅ **scripts/sync_files.sh** - Automatically syncs all files to `python/`

### Verification Completed

```bash
# Source distribution includes all modular files
✓ labneura-0.2.5.tar.gz
  - 15 modular .cpp files (avx2/, generic/, neon/)
  - 3 internal .h files
  - All organized in proper subdirectories
  - twine check: PASSED

# Binary wheel compiles successfully
✓ labneura-0.2.5-cp39-cp39-macosx_10_9_universal2.whl
  - Contains labneura.cpython-39-darwin.so (979 KB)
  - twine check: PASSED

# All tests pass
✓ 557 tests passed
✓ 27 tests skipped (platform-specific)
✓ Zero regressions
```

## Publishing Commands

### 1. Build Distribution Packages

```bash
make build-dist
```

This will:
- Sync files from `src/` to `python/src/`
- Build both sdist (`.tar.gz`) and wheel (`.whl`)
- Run `twine check` to validate packages

### 2. Publish to TestPyPI (Recommended First)

```bash
export TEST_PYPI_TOKEN="your-test-pypi-token"
make publish-testpypi
```

Then test installation:
```bash
pip install --index-url https://test.pypi.org/simple/ labneura
```

### 3. Publish to PyPI (Production)

```bash
export PYPI_TOKEN="your-pypi-token"
make publish
```

This will:
- Run full test suite
- Build distributions
- Increment the patch version (or use `VERSION=X.Y.Z`) and push the version commit
- Upload to PyPI
- Display confirmation message

For GitHub releases, create and publish a release with a tag such as `v1.2.3`.
The workflow builds and publishes using `1.2.3`, then opens a version-sync PR
if the default branch does not already declare that version.

## What Changed in Modularization

### Before
```
src/labneura/backends/
├── avx2.cpp (1,666 lines)
├── neon.cpp (1,628 lines)
├── generic.cpp (129 lines)
```

### After
```
src/labneura/backends/
├── avx2/
│   ├── avx2_backend.cpp (183 lines)
│   ├── avx2_ops_fp32.cpp (607 lines)
│   ├── avx2_ops_int8.cpp (138 lines)
│   ├── avx2_ops_int16.cpp (400 lines)
│   ├── avx2_ops_fp16.cpp (367 lines)
│   └── avx2_internal.h (66 lines)
├── generic/
│   ├── generic_backend.cpp (75 lines)
│   ├── generic_ops_fp32.cpp (25 lines)
│   ├── generic_ops_int8.cpp (43 lines)
│   └── generic_internal.h (5 lines)
└── neon/
    ├── neon_backend.cpp (142 lines)
    ├── neon_ops_fp32.cpp (768 lines)
    ├── neon_ops_int8.cpp (376 lines)
    ├── neon_ops_int16.cpp (382 lines)
    └── neon_internal.h (10 lines)
```

## Benefits for Package Users

1. **No API Changes** - All existing code continues to work
2. **Same Binary Size** - Compiled `.so` file size unchanged
3. **Faster Compilation** - Only modified files recompile during development
4. **Better Debugging** - Stack traces show specific operation files

## Pre-Publishing Checklist

Before running `make publish`, ensure:

- [ ] All tests pass: `make test`
- [ ] Choose the intended version; local publish auto-bumps patch unless `VERSION=X.Y.Z` is provided
- [ ] CHANGELOG updated with new features
- [ ] README updated if API changed
- [ ] Clean build: `make clean install test`
- [ ] Token environment variable set: `PYPI_TOKEN` or `TEST_PYPI_TOKEN`

## Troubleshooting

### Issue: Files missing from sdist

**Solution:** Run `bash scripts/sync_files.sh` before building

### Issue: Compilation errors during wheel build

**Solution:** Check the CMake target sources and rebuild the root package.

### Issue: Import errors after installation

**Solution:** Rebuild cleanly: `make clean install test`

## Documentation

- Full architecture: [docs/BACKEND_ARCHITECTURE.md](docs/BACKEND_ARCHITECTURE.md)
- Development guide: [README.md](README.md)

---

✅ **Status: Ready to Publish**

The modular backend architecture is fully integrated and tested. All packaging configuration is complete.
