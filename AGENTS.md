# LabNeura agent rules

Use `.github/skills/labneura-engineering/SKILL.md` for implementation and
validation policy. Preserve unrelated changes and never commit or push unless
the user explicitly requests it.

## Windows is a first-class target

Every portable change must consider GCC, Clang, AppleClang, and MSVC. Use
compiler and architecture guards independently, fixed-width integer types,
standard filesystem/path APIs, portable environment access, and CMake/Python
scripts that do not depend on POSIX shells. Keep SIMD intrinsics in isolated
translation units and select them only after runtime CPU capability checks.
Portable x86 wheels must not require AVX2 at process startup.

Use portable aligned allocation and explicit DLL import/export annotations.
Check Python extension initialization and generated-source/string-literal size
on MSVC. Never infer Windows correctness from AppleClang or GCC builds: when a
local MSVC runtime is unavailable, the matching Windows CI job is required and
must remain reported as pending until it executes.

Past failures included POSIX-only workflow quoting, passing an empty path to
`dumpbin`, unavailable CMake test targets, generated Liberty/string sources
exceeding MSVC limits, and architecture flags leaking into generic code.
Prevent them with native shell syntax per runner, validated paths, target
existence checks, bounded generated translation units, and per-source SIMD
compiler options.

## Development and release entry points

Use `make compile`, `make test`, and the equivalent `python scripts/dev.py`
commands. `make push MSG="..."` is user-authorized Git orchestration only; agents
must not invoke it automatically. Releases contain exactly `labneura`,
`labneura-tensor`, and `labneura-eda`; `scripts/release_inventory.py` is the
authoritative Python/platform inventory.
