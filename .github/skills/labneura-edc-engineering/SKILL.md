---
name: labneura-edc-engineering
description: Use for LabNeura EDA Verilog/SystemVerilog frontend work: lexer, preprocessor, parser, semantic analysis, type inference, constant evaluation, elaboration, performance optimization, verification, and benchmarking.
---

# LabNeura EDA Compiler Engineering Skill

You are an engineering agent working on LabNeura's native Verilog/SystemVerilog frontend
(EDA component). The goal is an independently maintained, high-performance frontend that
progressively replaces Slang as the production parser. C++ is the implementation language.

## Mission

Build a native LabNeura frontend that is:
- Functionally correct.
- Progressively compatible with Verilog/SystemVerilog.
- Faster than established parsers on measured representative workloads.
- Memory-efficient.
- SIMD-accelerated where beneficial.
- Multithreaded where dependencies permit.
- Independently maintained without Slang as a production dependency.

Slang remains the temporary correctness oracle and production parser. Do not remove it prematurely.

## Read before substantial work

- `.github/copilot-instructions.md`: repository-wide rules and the 12 mandatory EDA engineering rules.
- `.github/skills/labneura-edc-engineering/references/eda-current-state.md`: verified EDA architecture, completed work,
  current development priorities, and unresolved validation.
- `.github/skills/labneura-edc-engineering/templates/feature-task.md`: template for splitting work into bounded tasks.
- The actual repository source, tests, CMake targets, and benchmarks. The reference can become outdated;
  inspect the current source before proposing changes.

## Mandatory EDA engineering rules

1. C++ remains the primary implementation language.
2. Preserve existing public APIs unless a justified migration is required.
3. Keep EDA independent of Tensor. Do not introduce Tensor dependencies into EDA.
4. Keep acceleration-core responsibilities separate from EDA language semantics.
5. Preserve existing CMake component isolation.
6. Preserve Python packaging boundaries.
7. Keep scalar implementations as correctness references for optimized backends.
8. Keep hardware backend selection separate from frontend selection.
9. Use source maps and preserve diagnostics across macro expansions and includes.
10. Never claim complete IEEE SystemVerilog compliance without evidence.
11. Use Slang as a temporary correctness oracle, not as the permanent native implementation.
12. Do not rewrite working subsystems unnecessarily.

## Mandatory workflow for every task

### 1. Inspect
- Check `git status --short` and relevant diffs.
- Locate the public API header, implementation, tests, CMake registration, and benchmark code.
- Read existing conventions. Do not invent parallel abstractions without a concrete need.
- Identify dependencies and assumptions.

### 2. Define the contract before coding
Specify:
- API and expected input/output.
- Supported language constructs and versions.
- Width, type, and signedness inference rules (semantic agent owns this).
- Allocation, mutation, aliasing and lifetime behavior.
- Invalid-input errors and diagnostic codes.
- Required tests and reference comparison (Slang differential where applicable).

### 3. Tests first or alongside implementation
- Add focused tests before or in the same change as implementation.
- Implement and test scalar/reference semantics first.
- When adding an optimized backend (NEON/AVX2), add explicit tests for that backend in the same change.
- Register native tests with the existing CMake/CTest setup.
- Add differential tests against Slang where workloads and output semantics are comparable.

### 4. Implement one layer at a time
- Do not implement several backends in one speculative patch.
- Do not duplicate logic across components.
- Coordinate API-contract changes with the architect before editing shared interfaces.

### 5. Validate
- Build the narrow CMake target.
- Run focused native tests.
- Run the complete relevant native suite.
- Run existing formatting/static checks.
- Use sanitizers when memory behavior changes.

### 6. Required completion report
1. Exact change and semantic contract.
2. Files changed.
3. Generic/scalar and optimized backend coverage.
4. Tests added and exact commands/results.
5. Tests not run and why.
6. Benchmarks and regressions, if relevant.
7. Known limitations.
8. Confirmation unrelated worktree changes were preserved.
9. Confirmation no commit/push was made unless requested.

## Component boundaries

| Component | Owning agent | Key files |
|-----------|-------------|-----------|
| Lexer (scalar + NEON/AVX2 classifier) | Agent 2 | `include/labneura/eda/verilog_lexer.h`, `src/labneura/eda/parser/verilog/lexer/*` |
| Preprocessor | Agent 2 (with Agent 4 for macro semantics) | `include/labneura/eda/verilog_preprocessor.h`, `src/labneura/eda/parser/verilog/preprocessor/*` |
| Parser / Syntax | Agent 3 | `include/labneura/eda/verilog_syntax.h`, `src/labneura/eda/parser/verilog/syntax/*` |
| Semantic analysis | Agent 4 | `include/labneura/eda/verilog_semantic.h`, `src/labneura/eda/parser/verilog/semantic/*` |
| Type system / width inference | Agent 4 | `include/labneura/eda/verilog_type.h`, `src/labneura/eda/parser/verilog/types/*` |
| Constant evaluation | Agent 4 | `include/labneura/eda/verilog_constant.h`, `src/labneura/eda/parser/verilog/evaluation/*` |
| Elaboration | Agent 5 | `include/labneura/eda/verilog_elaboration.h`, `src/labneura/eda/parser/verilog/elaboration/*` |
| Performance / SIMD | Agent 6 | `src/labneura/eda/parser/verilog/lexer/verilog_byte_classifier*`, `src/labneura/tensor/core/kernels/cpu/*` |
| Verification / differential | Agent 7 | `tests/eda_*_test.cpp`, `tests/data/verilog/*` |
| Benchmarks | Agent 8 | `benchmarks/eda/*`, `benchmarking/*` |

## Existing EDA test data

- `tests/data/verilog/lexer_corpus.sv` — lexer differential corpus.
- `tests/data/verilog/parser_foundation.sv` — parser fixture.
- `tests/data/verilog/behavioral_rtl.sv` — behavioral RTL fixture.
