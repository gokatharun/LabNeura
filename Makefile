# ======================================
# LabNeura Local CI/CD Makefile
# ======================================

# Python
PYTHON ?= python3
PIP ?= pip3

# Apple Clang (force correct toolchain)
CLANG := $(shell xcrun --find clang)
CLANGXX := $(shell xcrun --find clang++)

# Export compiler for setuptools
export CC := $(CLANG)
export CXX := $(CLANGXX)

# Project
PKG_NAME = labneura

# ======================================
# Default target
# ======================================
.PHONY: all
all: clean install test

# ======================================
# Install dependencies
# ======================================
.PHONY: deps
deps:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install pytest pytest-cov numpy

# ======================================
# Clean build artifacts
# ======================================
.PHONY: clean
clean:
	rm -rf build dist *.egg-info
	find . -name "*.profraw" -delete
	find . -name "*.profdata" -delete
	find . -name "__pycache__" -delete

# ======================================
# Build & install package (CI-equivalent)
# ======================================
.PHONY: install
install:
	$(PIP) uninstall -y $(PKG_NAME) || true
	cd python && $(PIP) install . --no-cache-dir --no-build-isolation

# ======================================
# Import smoke test (CRITICAL)
# ======================================
.PHONY: smoke
smoke:
	$(PYTHON) -c "import $(PKG_NAME)" || (echo "❌  Import test failed!" && exit 1)
	@echo "✅  Import test passed!"

# ======================================
# Run unit tests
# ======================================
.PHONY: test
test: smoke
	pytest -v

# ======================================
# Python coverage
# ======================================
.PHONY: coverage
coverage:
	pytest --cov=$(PKG_NAME) --cov-report=term-missing || true

# ======================================
# CI (same as GitHub Actions)
# ======================================
.PHONY: ci
ci: deps clean install test coverage

# ======================================
# C++ Coverage (optional, advanced)
# ======================================
.PHONY: cpp-coverage
cpp-coverage:
	@echo "⚠️  Running C++ coverage build"
	export LABNEURA_COVERAGE=1; \
	export LLVM_PROFILE_FILE=labneura.profraw; \
	$(MAKE) clean install test; \
	llvm-profdata merge -sparse labneura.profraw -o labneura.profdata; \
	llvm-cov report src/$(PKG_NAME)/_core*.so \
	  -instr-profile=labneura.profdata