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
PKG_NAME := labneura

# --------------------------------------
# Compiler detection
# --------------------------------------
ifeq ($(UNAME_S),Darwin)
	CLANG := $(shell xcrun --find clang)
	CLANGXX := $(shell xcrun --find clang++)
else
	CLANG ?= clang
	CLANGXX ?= clang++
endif

export CC := $(CLANG)
export CXX := $(CLANGXX)

# --------------------------------------
# LLVM tools (optional)
# --------------------------------------
LLVM_PROFDATA := $(shell command -v llvm-profdata 2>/dev/null)
LLVM_COV := $(shell command -v llvm-cov 2>/dev/null)

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
	find . -name "*.profraw" -delete || true
	find . -name "*.profdata" -delete || true
	find . -name "__pycache__" -delete || true

# --------------------------------------
# Build & install
# --------------------------------------
.PHONY: install
install:
	$(PIP) uninstall -y $(PKG_NAME) || true
	cd python && $(PIP) install . --no-cache-dir --no-build-isolation

# --------------------------------------
# Smoke test
# --------------------------------------
.PHONY: smoke
smoke:
	$(PYTHON) -c "import $(PKG_NAME)" || (echo "❌ Import failed" && exit 1)
	@echo "✅ Import test passed"

# --------------------------------------
# Tests
# --------------------------------------
.PHONY: test
test: smoke
	pytest -v

# --------------------------------------
# Python coverage
# --------------------------------------
.PHONY: coverage
coverage:
	pytest --cov=$(PKG_NAME) --cov-report=term-missing || true

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
	SO_PATH=`$(PYTHON) -c "import labneura; import os; print(os.path.abspath(labneura.__file__))"`; \
	$(LLVM_COV) show $$SO_PATH \
	$(COV_ARCH) \
	-instr-profile=labneura.profdata \
	-format=html \
	-output-dir=coverage_html \
	-ignore-filename-regex='site-packages|python|omp|libc\+\+'; \
	echo "✅ HTML coverage generated at coverage_html/index.html"
endif

# --------------------------------------
# CI (matches GitHub Actions)
# --------------------------------------
.PHONY: ci
ci: deps clean install test cpp-coverage