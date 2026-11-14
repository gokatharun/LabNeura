# ======================================
# LabNeura Portable Local CI/CD Makefile
# ======================================

# --------------------------------------
# Platform detection
# --------------------------------------
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# --------------------------------------
# Hardware-aware build parallelism (used by CMake/scikit-build-core)
# --------------------------------------
NPROC := $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
export CMAKE_BUILD_PARALLEL_LEVEL ?= $(NPROC)

# --------------------------------------
# Python
# --------------------------------------
PYTHON ?= python3
PKG_NAME := labneura
MODULE_NAME := labneura

# Prefer project virtualenv for publish steps
# Get absolute path to workspace root
WORKSPACE_ROOT := $(abspath $(CURDIR))
VENV_DIRS := $(WORKSPACE_ROOT)/.venv/bin $(WORKSPACE_ROOT)/env/bin \
	$(WORKSPACE_ROOT)/venv/bin $(WORKSPACE_ROOT)/.venv/Scripts \
	$(WORKSPACE_ROOT)/env/Scripts $(WORKSPACE_ROOT)/venv/Scripts
VENV_PYTHON := $(firstword $(foreach dir,$(VENV_DIRS),$(wildcard \
	$(dir)/python $(dir)/python3 $(dir)/python3.* $(dir)/python.exe \
	$(dir)/python3.exe)))

# Prefer venv if available to avoid PEP 668 system pip restrictions
ifneq (,$(VENV_PYTHON))
	PYTHON := $(VENV_PYTHON)
endif
PIP = $(PYTHON) -m pip
TWINE = $(PYTHON) -m twine
# Older pip versions copy local projects to a temporary directory, preventing
# scikit-build-core from reusing its configured CMake build directory.
PIP_IN_TREE_BUILD := $(shell $(PYTHON) -c \
	'import pip; version = tuple(int(part) for part in pip.__version__.split(".")[:2]); print("--use-feature=in-tree-build" if version < (21, 3) else "")')

# --------------------------------------
# Compiler detection
# --------------------------------------
# ifeq ($(UNAME_S),Darwin)
# 	CLANG := $(shell xcrun --find clang)
# 	CLANGXX := $(shell xcrun --find clang++)
# else
# 	CLANG ?= clang
# 	CLANGXX ?= clang++
# endif

# export CC := $(CLANG)
# export CXX := $(CLANGXX)

# --------------------------------------
# LLVM tools (optional)
# --------------------------------------
# LLVM_PROFDATA := $(shell command -v llvm-profdata 2>/dev/null)
# LLVM_COV := $(shell command -v llvm-cov 2>/dev/null)

# --------------------------------------
# Architecture (macOS only)
# --------------------------------------
ifeq ($(UNAME_S),Darwin)
	COV_ARCH := -arch $(UNAME_M)
else
	COV_ARCH :=
endif

# --------------------------------------
# Default target
# --------------------------------------
.PHONY: all
all: clean sync-src install test

# --------------------------------------
# Dependencies
# --------------------------------------
.PHONY: deps
deps:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install pytest pytest-cov numpy pybind11

# --------------------------------------
# Clean
# --------------------------------------
.PHONY: clean
clean:
	rm -rf build dist *.egg-info coverage_html
	rm -f labneura*.so python/labneura*.so
	find . -name "*.profraw" -delete || true
	find . -name "*.profdata" -delete || true
	find . -name "__pycache__" -delete || true

# --------------------------------------
# Build & install
# --------------------------------------
.PHONY: sync-src
sync-src:
	./scripts/sync_files.sh

.PHONY: install
install: sync-src
	$(PIP) install $(PIP_IN_TREE_BUILD) . --no-cache-dir

# --------------------------------------
# Smoke test
# --------------------------------------
.PHONY: smoke
smoke:
	$(PYTHON) -c "import $(MODULE_NAME)" || (echo "❌ Import failed" && exit 1)
	@echo "✅ Import test passed"

# --------------------------------------
# Tests
# --------------------------------------
.PHONY: test
test: smoke
	$(PYTHON) -m pytest -v

# --------------------------------------
# Python coverage
# --------------------------------------
.PHONY: coverage
coverage:
	$(PYTHON) -m pytest --cov=$(PKG_NAME) --cov-report=term-missing || true

# --------------------------------------
# C++ Coverage (optional, portable)
# --------------------------------------
.PHONY: cpp-coverage
cpp-coverage:
ifeq ($(LLVM_PROFDATA),)
	@echo "⚠️ llvm-profdata not found, skipping C++ coverage"
else ifeq ($(LLVM_COV),)
	@echo "⚠️ llvm-cov not found, skipping C++ coverage"
else
	@echo "⚠️ Running C++ coverage"
	set -e; \
	COVERAGE_BUILD="$(CURDIR)/build/cpp-coverage"; \
	rm -rf "$$COVERAGE_BUILD" coverage_html; \
	mkdir -p "$$COVERAGE_BUILD"; \
	export LLVM_PROFILE_FILE="$$COVERAGE_BUILD/labneura.profraw"; \
	PYTHON_EXECUTABLE=`$(PYTHON) -c "import sys; print(sys.executable)"`; \
	PYBIND11_DIR=`$(PYTHON) -m pybind11 --cmakedir`; \
	cmake -S . -B "$$COVERAGE_BUILD" \
		-DCMAKE_CXX_COMPILER="$(CXX)" \
		-DPython_EXECUTABLE="$$PYTHON_EXECUTABLE" \
		-Dpybind11_DIR="$$PYBIND11_DIR" \
		-DCMAKE_INSTALL_PREFIX="$$COVERAGE_BUILD/install" \
		-DSKBUILD=ON \
		-DLABNEURA_ENABLE_COVERAGE=ON; \
	cmake --build "$$COVERAGE_BUILD" --parallel "$(CMAKE_BUILD_PARALLEL_LEVEL)"; \
	cmake --install "$$COVERAGE_BUILD"; \
	SO_PATH=`PYTHONPATH="$$COVERAGE_BUILD/install" $(PYTHON) -c \
		"import $(MODULE_NAME); print($(MODULE_NAME).__file__)"`; \
	PYTHONPATH="$$COVERAGE_BUILD/install" $(PYTHON) -m pytest -v; \
	$(LLVM_PROFDATA) merge -sparse "$$COVERAGE_BUILD/labneura.profraw" \
		-o "$$COVERAGE_BUILD/labneura.profdata"; \
	$(LLVM_COV) show $$SO_PATH \
	$(COV_ARCH) \
	-instr-profile="$$COVERAGE_BUILD/labneura.profdata" \
	-path-equivalence=$(CURDIR)/src/labneura/backends/generic,$(CURDIR)/src/labneura/backends/cpu/generic \
	-path-equivalence=$(CURDIR)/src/labneura/backends/neon,$(CURDIR)/src/labneura/backends/cpu/neon \
	-path-equivalence=$(CURDIR)/src/labneura/backends/avx2,$(CURDIR)/src/labneura/backends/cpu/avx2 \
	-format=html \
	-output-dir=coverage_html \
	-ignore-filename-regex='site-packages|python|omp|libc\+\+|backends/base\.|backends/cpu_features\.|backends/backend_factory\.'; \
	echo "✅ HTML coverage generated at coverage_html/index.html"; \
	$(LLVM_COV) report $$SO_PATH \
    -instr-profile="$$COVERAGE_BUILD/labneura.profdata" \
	$(COV_ARCH) \
	-path-equivalence=$(CURDIR)/src/labneura/backends/generic,$(CURDIR)/src/labneura/backends/cpu/generic \
	-path-equivalence=$(CURDIR)/src/labneura/backends/neon,$(CURDIR)/src/labneura/backends/cpu/neon \
	-path-equivalence=$(CURDIR)/src/labneura/backends/avx2,$(CURDIR)/src/labneura/backends/cpu/avx2 \
	-ignore-filename-regex='site-packages|python3.9|omp|include|backends/base\.|backends/cpu_features\.|backends/backend_factory\.' > coverage.txt; \
	echo "✅ Coverage report generated at coverage.txt";

endif

# --------------------------------------
# CI (matches GitHub Actions)
# --------------------------------------
.PHONY: ci
ci: deps clean install smoke cpp-coverage

# --------------------------------------
# Build distribution (sdist + wheel)
# --------------------------------------
.PHONY: build-dist
build-dist: sync-src
	$(PIP) install --upgrade build twine
	LABNEURA_COVERAGE= MACOSX_DEPLOYMENT_TARGET=11.0 $(PYTHON) -m build
	$(TWINE) check dist/*

# --------------------------------------
# Update all package, native module, and CMake package versions
# --------------------------------------
VERSION_FILES = pyproject.toml CMakeLists.txt python/labneura_py.cpp \
	packaging/core/src/labneura/__init__.py \
	packaging/tensor/pyproject.toml packaging/eda/pyproject.toml \
	packaging/compat/pyproject.toml packaging/all/pyproject.toml
.PHONY: update-version
update-version:
	@if ! git diff --quiet -- $(VERSION_FILES) || \
	    ! git diff --cached --quiet -- $(VERSION_FILES); then \
		echo "Version files have existing edits; commit or stash them before publishing."; \
		exit 1; \
	fi
	$(PYTHON) scripts/update_version.py $(if $(VERSION),--version "$(VERSION)",--bump-patch)

# --------------------------------------
# Commit and push version changes to git
# --------------------------------------
.PHONY: git-commit-version
git-commit-version:
	@set -e; \
	NEW_VERSION=$$($(PYTHON) scripts/update_version.py --get-version); \
	CURRENT_BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
	if [ "$$CURRENT_BRANCH" = "HEAD" ]; then \
		echo "Cannot push a version bump from a detached HEAD."; \
		exit 1; \
	fi; \
	if git diff --quiet -- $(VERSION_FILES) && \
	   git diff --cached --quiet -- $(VERSION_FILES); then \
		echo "No version changes to commit."; \
		exit 1; \
	fi; \
	git add $(VERSION_FILES); \
	git commit -m "chore: bump version to $$NEW_VERSION" -- $(VERSION_FILES); \
	git push origin "$$CURRENT_BRANCH"

# --------------------------------------
# Publish to PyPI (requires PYPI_TOKEN env variable)
# --------------------------------------
.PHONY: publish
publish:
	@if [ -z "$(PYPI_TOKEN)" ]; then \
		echo "PYPI_TOKEN not set. Export PYPI_TOKEN before running 'make publish'."; \
		exit 1; \
	fi
	$(MAKE) sync-src
	$(MAKE) update-version VERSION="$(VERSION)"
	# Full local validation using venv
	$(MAKE) clean install smoke test
	$(MAKE) build-dist
	$(MAKE) git-commit-version
	# Upload artifacts to PyPI
	$(TWINE) upload --non-interactive dist/* --username __token__ --password $(PYPI_TOKEN)
	@echo "✅ Published to PyPI"

# --------------------------------------
# Publish to TestPyPI (requires TEST_PYPI_TOKEN env variable)
# --------------------------------------
.PHONY: publish-testpypi
publish-testpypi:
	@if [ -z "$(TEST_PYPI_TOKEN)" ]; then \
		echo "TEST_PYPI_TOKEN not set. Export TEST_PYPI_TOKEN before running 'make publish-testpypi'."; \
		exit 1; \
	fi
	$(MAKE) sync-src
	$(MAKE) update-version VERSION="$(VERSION)"
	$(MAKE) clean install smoke test
	$(MAKE) build-dist
	$(MAKE) git-commit-version
	$(TWINE) upload --repository testpypi --non-interactive dist/* --username __token__ --password $(TEST_PYPI_TOKEN)
	@echo "✅ Published to TestPyPI"
