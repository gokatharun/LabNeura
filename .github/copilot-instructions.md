# LabNeura repository instructions

- LabNeura is a C++ n-dimensional tensor/numerical-computing library with Python bindings.
- Inspect `git status`, relevant diffs, public APIs, implementations, tests and build setup before editing.
- Preserve unrelated worktree changes. Never reset or overwrite user work.
- Implement one bounded feature or optimization per task. Avoid incidental refactors.
- Every new public feature must receive tests in the same change. Every new backend implementation must receive explicit backend-specific correctness tests immediately.
- Generic is the semantic reference. Optimized backends must match documented semantics.
- Test narrow targets first, then relevant full native and Python suites.
- Benchmark optimizations with correctness checks, equivalent inputs, median/IQR/p95, and recorded hardware/backend/dtype/shape/thread configuration.
- Do not claim support or performance without implementation and tests.
- Do not silently change broadcasting, overflow, saturation, aliasing, promotion, or error semantics.
- Do not commit or push unless explicitly requested.
- For feature, backend, runtime, test or performance work, follow `.github/skills/labneura-engineering/SKILL.md`.
