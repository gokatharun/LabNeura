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
PIP ?= pip3
PKG_NAME := labneura
MODULE_NAME := labneura

# Prefer project virtualenv for publish steps
# Get absolute path to workspace root
WORKSPACE_ROOT := $(shell pwd)
VENV_PYTHON := $(WORKSPACE_ROOT)/.venv/bin/python
VENV_PIP := $(WORKSPACE_ROOT)/.venv/bin/pip
TWINE = $(PYTHON) -m twine
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
	$(PIP) install . --no-cache-dir

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
	-ignore-filename-regex='site-packages|python|omp|libc\+\+|backends/base\.|backends/cpu_features\.|backends/backend_factory\.'; \
	echo "✅ HTML coverage generated at coverage_html/index.html"; \
	$(LLVM_COV) report $$SO_PATH \
    -instr-profile=labneura.profdata \
	$(COV_ARCH) \
	-path-equivalence=$(CURDIR)/src/labneura/backends/generic,$(CURDIR)/src/labneura/backends/cpu/generic \
	-path-equivalence=$(CURDIR)/src/labneura/backends/neon,$(CURDIR)/src/labneura/backends/cpu/neon \
	-path-equivalence=$(CURDIR)/src/labneura/backends/avx2,$(CURDIR)/src/labneura/backends/cpu/avx2 \
	-ignore-filename-regex='site-packages|python3.9|omp|include|backends/base\.|backends/cpu_features\.|backends/backend_factory\.' > coverage.txt; \
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
	$(PIP) install --upgrade build twine
	LABNEURA_COVERAGE= MACOSX_DEPLOYMENT_TARGET=11.0 $(PYTHON) -m build
	$(TWINE) check dist/*

# Store the new version for use in git operations
NEW_VERSION_FILE := /tmp/labneura_new_version.txt

# --------------------------------------
# Update version in pyproject.toml and labneura_py.cpp
# --------------------------------------
.PHONY: update-version
update-version:
	@echo "==================================="
	@echo "LabNeura Version Update"
	@echo "==================================="
	@CURRENT_VERSION=$$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml); \
	echo "Current version: $$CURRENT_VERSION"; \
	echo ""; \
	read -p "Enter new version (or press Enter to keep current): " NEW_VERSION; \
	if [ -z "$$NEW_VERSION" ]; then \
		echo "Keeping current version: $$CURRENT_VERSION"; \
		echo "$$CURRENT_VERSION" > $(NEW_VERSION_FILE); \
	else \
		echo "Updating version from $$CURRENT_VERSION to $$NEW_VERSION..."; \
		sed -i '' "s/^version = \"$$CURRENT_VERSION\"/version = \"$$NEW_VERSION\"/" pyproject.toml; \
		echo "✅ Version updated to $$NEW_VERSION in pyproject.toml"; \
		CURRENT_CPP_VERSION=$$(grep '__version__' python/labneura_py.cpp | sed -E 's/.*"([^"]+)".*/\1/'); \
		sed -i '' "s/m.attr(\"__version__\") = \"$$CURRENT_CPP_VERSION\"/m.attr(\"__version__\") = \"$$NEW_VERSION\"/" python/labneura_py.cpp; \
		echo "✅ Version updated to $$NEW_VERSION in python/labneura_py.cpp"; \
		echo "$$NEW_VERSION" > $(NEW_VERSION_FILE); \
	fi
	@echo "==================================="

# --------------------------------------
# Commit and push version changes to git
# --------------------------------------
.PHONY: git-commit-version
git-commit-version:
	@if [ ! -f $(NEW_VERSION_FILE) ]; then \
		echo "❌ Version file not found. Run update-version first."; \
		exit 1; \
	fi
	@NEW_VERSION=$$(cat $(NEW_VERSION_FILE)); \
	CURRENT_BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
	echo ""; \
	echo "==================================="
	echo "Git Commit & Push"; \
	echo "==================================="
	echo "Branch: $$CURRENT_BRANCH"; \
	echo "New version: $$NEW_VERSION"; \
	echo ""; \
	echo "Files to commit:"; \
	git diff --name-only; \
	echo ""; \
	read -p "Continue with git commit and push? (y/n): " CONFIRM; \
	if [ "$$CONFIRM" = "y" ] || [ "$$CONFIRM" = "Y" ]; then \
		git add pyproject.toml python/labneura_py.cpp; \
		git commit -m "chore: bump version to $$NEW_VERSION"; \
		echo "✅ Committed: chore: bump version to $$NEW_VERSION"; \
		git push origin $$CURRENT_BRANCH; \
		echo "✅ Pushed to $$CURRENT_BRANCH"; \
	else \
		echo "⏭️  Skipped git commit and push"; \
	fi
	@echo "==================================="

# --------------------------------------
# Publish to PyPI (requires PYPI_TOKEN env variable)
# --------------------------------------
.PHONY: publish
publish: sync-src update-version git-commit-version
	@if [ -z "$(PYPI_TOKEN)" ]; then \
		echo "PYPI_TOKEN not set. Export PYPI_TOKEN before running 'make publish'."; \
		exit 1; \
	fi
	# Full local validation using venv
	$(MAKE) clean install smoke test PIP=$(VENV_PIP) PYTHON=$(VENV_PYTHON)
	$(MAKE) build-dist
	# Upload artifacts to PyPI
	$(TWINE) upload --non-interactive dist/* --username __token__ --password $(PYPI_TOKEN)
	@echo "✅ Published to PyPI"
	@rm -f $(NEW_VERSION_FILE)

# --------------------------------------
# Publish to TestPyPI (requires TEST_PYPI_TOKEN env variable)
# --------------------------------------
.PHONY: publish-testpypi
publish-testpypi: sync-src update-version git-commit-version
	@if [ -z "$(TEST_PYPI_TOKEN)" ]; then \
		echo "TEST_PYPI_TOKEN not set. Export TEST_PYPI_TOKEN before running 'make publish-testpypi'."; \
		exit 1; \
	fi
	$(MAKE) clean install smoke test PIP=$(VENV_PIP) PYTHON=$(VENV_PYTHON)
	$(MAKE) build-dist
	$(TWINE) upload --repository testpypi --non-interactive dist/* --username __token__ --password $(TEST_PYPI_TOKEN)
	@echo "✅ Published to TestPyPI"
	@rm -f $(NEW_VERSION_FILE)