# CI validation responsibilities

ARCH-1C separates routine correctness from expensive and hardware-dependent
validation. Every workflow uses source builds for its own ABI/compiler target;
wheel artifacts are reused only by verification on that same runner family.

| Workflow | Trigger | Responsibility | Deliberately excluded |
|---|---|---|---|
| `ci.yml` | PR, push to main | Architecture, one Release C++ build per primary platform, CTest, three-wheel installation, Python regression | Sanitizers, formal tools, full wheel matrix, timing |
| `performance.yml` | Weekly/manual | Uninstrumented Release Tensor and lexer benchmarks after CTest | Coverage and publishing thresholds |
| `extended-validation.yml` | Weekly/manual | Sanitizers, Yosys, pinned external Slang validation, source-distribution rebuilds, detailed native coverage and lexer comparison | Required PR latency |
| `publish.yml` | Release/manual | Exact 60 native wheels plus one umbrella wheel, sdists, install/linkage/release qualification | Rebuilding already qualified artifacts |

The pre-ARCH-1C `ci.yml` performed all four responsibility classes in a
single large workflow. It is retained as `extended-validation.yml`, so no
existing validation category was silently removed. Routine CI now compiles one
complete native tree per supported primary hardware/compiler target and builds
component wheels once per runner. Release remains the sole publisher and is
not subject to ordinary-run cancellation.

Coverage builds are Debug/instrumented and never used for performance. LLVM
profiles use `%p-%m.profraw`, making parallel CTest profile collection safe.
Native report exclusions are test harnesses, generated/build files, and
third-party sources; architecture-specific paths unexecutable on a runner are
reported as unmeasured rather than averaged into another backend.
