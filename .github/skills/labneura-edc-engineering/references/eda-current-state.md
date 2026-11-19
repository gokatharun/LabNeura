# LabNeura EDA Current Project State

Last verified: 2026-11-15. Treat this as a continuity reference, not a substitute for
inspecting the current source, tests, worktree, and benchmark artifacts.

## Project identity

LabNeura is a C++ n-dimensional tensor/numerical-computing library with Python bindings,
plus an independently implemented Verilog/SystemVerilog frontend (EDA component) that
progressively replaces Slang as the production parser.

## EDA frontend architecture

The native frontend lives under `include/labneura/eda/` and `src/labneura/eda/`:

| Layer | Header | Implementation | Status |
|-------|--------|---------------|--------|
| Lexer | `verilog_lexer.h` | `verilog_lexer.cpp` | Scalar reference complete; NEON/AVX2 byte classification available |
| Byte classifier | `verilog_byte_classifier.h` | `verilog_byte_classifier.cpp`, `_avx2.cpp` | Scalar always compiled; NEON on ARM64; AVX2 on x86 |
| Preprocessor | `verilog_preprocessor.h` | `verilog_preprocessor.cpp` | Native preprocessing and macro expansion |
| Parser / Syntax | `verilog_syntax.h` | `verilog_parser.cpp` | Recursive-descent parser, owned AST, error recovery |
| Semantic | `verilog_semantic.h` | `verilog_semantic.cpp` | Scopes, symbol resolution, declaration/reference tracking |
| Types | `verilog_type.h` | `verilog_type.cpp` | Type categories, width, signedness, four-state |
| Constant | `verilog_constant.h` | `verilog_constant.cpp` | Four-state constant evaluation, 64K width limit |
| Elaboration | `verilog_elaboration.h` | `verilog_elaboration.cpp` | Hierarchy, generate, parameter specialization, cross-module binding |

Expression width, type, and signedness semantics are the next active implementation area.

Slang currently remains the production parser and differential reference. Do not remove it prematurely.

## Public API surface

- `labneura::eda::lex_verilog()` — tokenize Verilog/SystemVerilog source.
- `labneura::eda::preprocess_verilog()` — preprocess with source maps and macro expansion.
- `labneura::eda::parse_verilog_syntax()` — parse to owned AST (`VerilogCompilationUnit`).
- `labneura::eda::analyze_verilog_semantics()` — semantic analysis (scopes, symbols, references).
- `labneura::eda::infer_verilog_expression_type()` / `analyze_verilog_expression_types()` — type inference.
- `labneura::eda::evaluate_verilog_constant()` — four-state constant evaluation.
- `labneura::eda::elaborate_verilog_design()` — hierarchy elaboration, generate, parameter specialization.

## Key types

- `VerilogToken`, `VerilogTokenKind`, `VerilogLexResult`
- `VerilogPreprocessorResult`, `VerilogSourceMapSegment`
- `VerilogCompilationUnit`, `VerilogModuleDeclaration`, `VerilogExpression`, `VerilogStatement`
- `VerilogSemanticScope`, `VerilogSemanticSymbol`, `VerilogSemanticAnalysisResult`
- `VerilogIntegralType`, `VerilogTypeCategory`, `VerilogExpressionTypeResult`, `VerilogTypeAnalysisResult`
- `VerilogConstantValue`, `VerilogConstantEvaluationResult`, `VerilogConstantEnvironment`
- `VerilogElaboratedInstance`, `VerilogElaboratedGenerateScope`, `VerilogElaborationResult`

## Lexer backends

- `VerilogLexerBackend::Auto`, `Scalar`, `Neon`, `Avx2`
- `LABNEURA_VERILOG_LEXER_BACKEND=auto|scalar|neon` env var selects classifier backend.
- `LABNEURA_VERILOG_LEXER=slang|scalar` selects lexer (default: slang; production parser unchanged).

## Existing EDA tests

- `tests/eda_verilog_lexer_test.cpp` — lexer correctness, differential vs Slang, keyword tables.
- `tests/eda_verilog_syntax_test.cpp` — parser correctness, error recovery, Slang acceptance.
- `tests/eda_verilog_semantic_test.cpp` — symbol resolution, scope boundaries, diagnostics.
- `tests/eda_verilog_elaboration_test.cpp` — hierarchy, generate, parameters, top selection.
- `tests/eda_verilog_preprocessor_test.cpp` — preprocessing, source maps, includes.
- `tests/eda_verilog_type_test.cpp` — type inference (width, signedness).
- `tests/test_eda_equivalence.py` — Python-level EDA equivalence.
- `tests/test_eda.py` — Python EDA tests.
- `tests/test_eda_graph.py` — EDA graph tests.
- `tests/eda_verilog_fpga.py` — FPGA-related EDA tests.

## Existing EDA benchmarks

- `benchmarks/eda/verilog_lexer_benchmark.cpp` — lexer throughput benchmark.
- `benchmarks/eda/verilog_preprocessor_benchmark.cpp` — preprocessor benchmark.

## EDA test data

- `tests/data/verilog/lexer_corpus.sv` — lexer differential corpus.
- `tests/data/verilog/parser_foundation.sv` — parser fixture (2 modules, 1 interface, 1 package).
- `tests/data/verilog/behavioral_rtl.sv` — behavioral RTL fixture.

## CMake EDA targets

- `labneura_eda` — main EDA library (all EDA source files).
- `labneura_lexer_benchmark` — lexer benchmark (EXCLUDE_FROM_ALL).
- `labneura_preprocessor_benchmark` — preprocessor benchmark (EXCLUDE_FROM_ALL).
- `labneura_python` — Python module (when SKBUILD and LABNEURA_BUILD_COMPAT).

## Validation baseline

Apple Silicon: full native CTest had 22 passes, 8 expected AVX2 skips, 0 failures.
Lexer benchmark on two source files (69,727 bytes, 13,256 tokens): 6.534 ms median,
1.404 ms IQR, 16.214 ms p95, 10.67 MB/s, 2.03 M tokens/s (Apple M1, arm64, Apple Clang 21.0.0).
This is a local scalar baseline, not a SIMD or cross-frontend performance claim.
