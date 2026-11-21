"""Guarded `make push` implementation; never force-pushes or changes branches."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED_PARTS = {"build", "dist", "__pycache__", ".venv", "venv", "env"}
BANNED_SUFFIXES = {".so", ".dylib", ".dll", ".pyd", ".a", ".lib", ".o", ".obj", ".pyc"}
SECRET_PATTERNS = (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
                   re.compile(r"(?i)(?:api[_-]?key|token|password)\s*[:=]\s*['\"][^'\"]{8,}"))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                          capture_output=True).stdout.rstrip("\n")


def changed_files(status: str) -> list[str]:
    result = []
    for line in status.splitlines():
        if not line: continue
        path = line[3:].split(" -> ", 1)[-1]
        result.append(path)
    return result


def validate_paths(paths: list[str]) -> None:
    for name in paths:
        path = Path(name)
        if any(part in BANNED_PARTS for part in path.parts) or path.suffix.lower() in BANNED_SUFFIXES:
            raise RuntimeError(f"refusing to stage generated or binary artifact: {name}")
        candidate = ROOT / path
        if candidate.is_file() and candidate.stat().st_size <= 2 * 1024 * 1024:
            try: text = candidate.read_text(encoding="utf-8")
            except UnicodeDecodeError as error: raise RuntimeError(f"refusing unrecognized binary: {name}") from error
            if any(pattern.search(text) for pattern in SECRET_PATTERNS):
                raise RuntimeError(f"possible credential in {name}")


def validate_repository(message: str) -> list[str]:
    if not message.strip(): raise RuntimeError("MSG must contain a nonempty commit message")
    if git("branch", "--show-current") != "main": raise RuntimeError("make push requires branch main")
    if not git("remote", "get-url", "origin"): raise RuntimeError("origin remote is missing")
    if git("diff", "--name-only", "--diff-filter=U"): raise RuntimeError("unresolved Git conflicts exist")
    paths = changed_files(git("status", "--porcelain", "--untracked-files=all"))
    if not paths: raise RuntimeError("no changes exist; refusing an empty commit")
    validate_paths(paths)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--message", required=True)
    parser.add_argument("--component", choices=("all", "core", "tensor", "eda"), default="all")
    parser.add_argument("--target", default="native"); parser.add_argument("--jobs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); paths = validate_repository(args.message)
    print("Validated staging scope:"); [print(f"  {path}") for path in paths]
    common = [sys.executable, "scripts/dev.py", "--component", args.component, "--target", args.target]
    if args.jobs: common += ["--jobs", str(args.jobs)]
    for command in ("compile", "test", "check"):
        subprocess.run([*common, command], cwd=ROOT, check=True)
    if args.dry_run:
        print("dry-run: validation passed; repository was not modified"); return 0
    subprocess.run(["git", "add", "--", *paths], cwd=ROOT, check=True)
    subprocess.run(["git", "diff", "--cached", "--check"], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-m", args.message], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
    sha, origin = git("rev-parse", "HEAD"), git("remote", "get-url", "origin")
    url = origin.removesuffix(".git").replace("git@github.com:", "https://github.com/")
    print(f"commit: {sha}\nactions: {url}/actions"); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr); raise SystemExit(1)
