# LabNeura repository instructions

LabNeura is a C++ n-dimensional tensor/numerical-computing library with Python bindings,
plus an independently implemented Verilog/SystemVerilog frontend (EDA component) that
progressively replaces Slang as the production parser.

## General rules (all components)

- Inspect `git status`, relevant diffs, public APIs, implementations, tests and build setup before editing.
- Preserve unrelated worktree changes. Never reset or overwrite user work.
- Implement one bounded feature or optimization per task. Avoid incidental refactors.
- Every new public feature must receive tests in the same change.
- Generic/scalar is the semantic reference. Optimized backends must match documented semantics.
- Test narrow targets first, then relevant full native and Python suites.
- Benchmark optimizations with correctness checks, equivalent inputs, median/IQR/p95,
  and recorded hardware/backend/dtype/shape/thread configuration.
- Do not claim support or performance without implementation and tests.
- Do not silently change broadcasting, overflow, saturation, aliasing, promotion, or error semantics.
- Do not commit or push unless explicitly requested.

## Windows portability (mandatory)

- Treat Windows x86_64/MSVC as a first-class build and runtime target alongside
  GCC, Clang, and AppleClang.
- Separate compiler guards from architecture guards, isolate SIMD translation
  units, and preserve runtime CPU dispatch; portable x86 wheels cannot assume
  AVX2 or F16C.
- Use fixed-width integer types, portable aligned allocation, standard
  filesystem paths, explicit DLL visibility, portable environment handling,
  and native Python extension initialization.
- Keep CMake and Python orchestration cross-platform. Do not embed POSIX shell
  assumptions in commands that execute on Windows.
- Previous regressions included empty `dumpbin` paths, absent CMake targets,
  POSIX quoting in PowerShell, and MSVC generated-source/string-literal limits.
  Validate paths and targets, use runner-native quoting, and bound generated
  translation units.
- AppleClang success is not evidence of MSVC success. Require remote Windows CI
  when local Windows compilation and runtime testing are unavailable.

## EDA compiler team rules (front end)

1. C++ remains the primary implementation language.
2. Preserve existing public APIs unless a justified migration is required.
3. Keep EDA independent of Tensor. Do not introduce Tensor dependencies into EDA language semantics.
4. Keep acceleration-core responsibilities separate from EDA language semantics.
5. Preserve existing CMake component isolation.
6. Preserve Python packaging boundaries.
7. Keep scalar implementations as correctness references for optimized backends.
8. Keep hardware backend selection separate from frontend selection.
9. Use source maps and preserve diagnostics across macro expansions and includes.
10. Never claim complete IEEE SystemVerilog compliance without evidence.
11. Use Slang as a temporary correctness oracle, not as the permanent native implementation.
12. Do not rewrite working subsystems unnecessarily.

## Eight specialist roles

The repository configures GitHub Copilot as an eight-role native compiler engineering team:

- **Agent 1 — Architect** (`.github/agents/labneura-architect.agent.md`): frontend architecture, API contracts, AST/type consistency, cross-component integration, work decomposition.
- **Agent 2 — Lexer Engineer** (`.github/agents/labneura-lexer.agent.md`): scalar reference lexer, NEON/AVX2/AVX-512 byte classification, runtime dispatch, token classification, source locations, throughput.
- **Agent 3 — Parser Engineer** (`.github/agents/labneura-parser.agent.md`): grammar coverage, parsing strategies, AST construction, error recovery, expression parsing, declarations, procedural syntax, functions, tasks, interfaces, packages.
- **Agent 4 — Semantic Engineer** (`.github/agents/labneura-semantics.agent.md`): symbol resolution, type system, width/signedness inference, four-state semantics, constant expressions, compatibility diagnostics.
- **Agent 5 — Elaboration Engineer** (`.github/agents/labneura-elaboration.agent.md`): module hierarchy, parameter specialization, generate constructs, package/interface visibility, cross-module binding, dependency-aware elaboration.
- **Agent 6 — CPU Performance Engineer** (`.github/agents/labneura-performance.agent.md`): SIMD/CPU optimization, NEON/AVX2/AVX-512, multithreading, cache locality, arena allocation, string interning, parallel semantic analysis.
- **Agent 7 — Verification Engineer** (`.github/agents/labneura-verification.agent.md`): differential testing vs Slang, language compatibility, regression, error recovery, randomized/generated tests, real-world RTL validation.
- **Agent 8 — Benchmark Engineer** (`.github/agents/labneura-benchmark.agent.md`): reproducible benchmark infrastructure, lexer/parser/semantic/elaboration benchmarks, memory/allocation measurements, scaling, regression detection.

## Work coordination

Agents must agree on API contracts before editing shared interfaces, own separate source files where practical,
communicate required interface changes, avoid concurrent conflicting edits, integrate frequently,
run relevant tests after implementation, and escalate genuine architectural conflicts.
The architect coordinates the work. Verification independently validates it. Performance independently measures it.

## Implementation-first policy

When assigned a feature: inspect only relevant implementation and dependencies → implement directly → add focused tests →
build the affected target → run relevant regression tests → run the full relevant suite when practical →
report changed files, test results and known limitations. Do not respond with another roadmap when implementation is requested.

## Worktree safety

Never execute git reset, git clean, checkout, stash, or destructive history operations.
Never discard existing user modifications or untracked files.
Never commit or push unless explicitly instructed.
Inspect existing changes before editing. Preserve unrelated work.

## Skill references

- Tensor/numerical work: `.github/skills/labneura-engineering/SKILL.md`
- EDA compiler work: `.github/skills/labneura-edc-engineering/SKILL.md`
