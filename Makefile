# ======================================
# LabNeura Portable Local CI/CD Makefile
# ======================================

# --------------------------------------
# Platform detection
# --------------------------------------
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# --------------------------------------
# Python
# --------------------------------------
PYTHON ?= python3
PIP ?= pip3
PKG_NAME := labneura2
MODULE_NAME := labneura

# Prefer project virtualenv for publish steps
# Get absolute path to workspace root
WORKSPACE_ROOT := $(shell pwd)
VENV_PYTHON := $(WORKSPACE_ROOT)/env/bin/python
VENV_PIP := $(WORKSPACE_ROOT)/env/bin/pip
TWINE := $(VENV_PYTHON) -m twine
PYPI_TOKEN ?=
TEST_PYPI_TOKEN ?=

# Prefer venv if available to avoid PEP 668 system pip restrictions
ifneq (,$(wildcard $(VENV_PIP)))
	PIP := $(VENV_PIP)
	PYTHON := $(VENV_PYTHON)
endif

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
all: clean install test

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
	rm -rf python/build python/dist python/*.egg-info
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
	$(PIP) uninstall -y $(PKG_NAME) || true
	cd python && $(PIP) install . --no-cache-dir --no-build-isolation

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
	export LABNEURA_COVERAGE=1; \
	export LLVM_PROFILE_FILE=labneura.profraw; \
	$(MAKE) clean install test; \
	$(LLVM_PROFDATA) merge -sparse labneura.profraw -o labneura.profdata; \
	SO_PATH=`$(PYTHON) -c "import $(MODULE_NAME); import os; print(os.path.abspath($(MODULE_NAME).__file__))"`; \
	$(LLVM_COV) show $$SO_PATH \
	$(COV_ARCH) \
	-instr-profile=labneura.profdata \
	-path-equivalence=$(CURDIR)/src/labneura/backends/generic,$(CURDIR)/src/labneura/backends/cpu/generic \
	-path-equivalence=$(CURDIR)/src/labneura/backends/neon,$(CURDIR)/src/labneura/backends/cpu/neon \
	-path-equivalence=$(CURDIR)/src/labneura/backends/avx2,$(CURDIR)/src/labneura/backends/cpu/avx2 \
	-format=html \
	-output-dir=coverage_html \
	-ignore-filename-regex='site-packages|python|omp|libc\+\+'; \
	echo "✅ HTML coverage generated at coverage_html/index.html"; \
	$(LLVM_COV) report $$SO_PATH \
    -instr-profile=labneura.profdata \
	$(COV_ARCH) \
	-path-equivalence=$(CURDIR)/src/labneura/backends/generic,$(CURDIR)/src/labneura/backends/cpu/generic \
	-path-equivalence=$(CURDIR)/src/labneura/backends/neon,$(CURDIR)/src/labneura/backends/cpu/neon \
	-path-equivalence=$(CURDIR)/src/labneura/backends/avx2,$(CURDIR)/src/labneura/backends/cpu/avx2 \
	-ignore-filename-regex='site-packages|python3.9|omp|include' > coverage.txt; \
	echo "✅ Coverage report generated at coverage.txt";

endif

.PHONY: coverage
coverage:
	pytest --cov=$(PKG_NAME) --cov-report=term-missing || true

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
	# Ensure build tools are available in venv
	$(VENV_PIP) install --upgrade build twine pybind11 || true
	# Build inside python package directory (disable coverage for packaging)
	# Build wheel directly (not from sdist) to avoid path issues
	cd python && LABNEURA_COVERAGE= $(VENV_PYTHON) -m build --no-isolation --wheel
	# Build sdist separately
	cd python && LABNEURA_COVERAGE= $(VENV_PYTHON) -m build --no-isolation --sdist
	cd python && $(TWINE) check dist/*

# --------------------------------------
# Publish to PyPI (requires PYPI_TOKEN env variable)
# --------------------------------------
.PHONY: publish
publish: sync-src
	@if [ -z "$(PYPI_TOKEN)" ]; then \
		echo "PYPI_TOKEN not set. Export PYPI_TOKEN before running 'make publish'."; \
		exit 1; \
	fi
	# Full local validation using venv
	$(MAKE) clean install smoke test PIP=$(VENV_PIP) PYTHON=$(VENV_PYTHON)
	$(MAKE) build-dist
	# Upload artifacts to PyPI
	cd python && $(TWINE) upload --non-interactive dist/* --username __token__ --password $(PYPI_TOKEN)
	@echo "✅ Published to PyPI"

# --------------------------------------
# Publish to TestPyPI (requires TEST_PYPI_TOKEN env variable)
# --------------------------------------
.PHONY: publish-testpypi
publish-testpypi: sync-src
	@if [ -z "$(TEST_PYPI_TOKEN)" ]; then \
		echo "TEST_PYPI_TOKEN not set. Export TEST_PYPI_TOKEN before running 'make publish-testpypi'."; \
		exit 1; \
	fi
	$(MAKE) clean install smoke test PIP=$(VENV_PIP) PYTHON=$(VENV_PYTHON)
	$(MAKE) build-dist
	cd python && $(TWINE) upload --repository testpypi --non-interactive dist/* --username __token__ --password $(TEST_PYPI_TOKEN)
	@echo "✅ Published to TestPyPI"