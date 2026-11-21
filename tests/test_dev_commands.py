import argparse
import subprocess
import sys

import pytest

from scripts import dev, git_push


def test_resource_detection_is_memory_and_cpu_bounded(monkeypatch):
    monkeypatch.setattr(dev.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(dev, "available_memory", lambda: 6 * 1024**3)
    monkeypatch.delenv("CMAKE_BUILD_PARALLEL_LEVEL", raising=False)
    resources = dev.detect_resources()
    assert resources.compile_jobs == 4
    assert resources.link_jobs == 2


def test_resource_detection_respects_explicit_override(monkeypatch):
    monkeypatch.setattr(dev.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(dev, "available_memory", lambda: 2 * 1024**3)
    assert dev.detect_resources(7).compile_jobs == 7


@pytest.mark.parametrize("component", dev.COMPONENTS)
def test_every_component_has_explicit_cmake_boundaries(component):
    arguments = dev.component_args(component)
    assert len([item for item in arguments if item.startswith("-DLABNEURA_BUILD_")]) >= 4
    assert "-DLABNEURA_ENABLE_SLANG_VALIDATION=OFF" in arguments


def test_push_rejects_empty_message():
    with pytest.raises(RuntimeError, match="nonempty"):
        git_push.validate_repository("")


def test_push_validation_uses_temporary_local_repository(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    remote = tmp_path / "origin.git"
    repository.mkdir()
    subprocess.run(["git", "init", "--bare", remote], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", "main", repository], check=True, capture_output=True)
    subprocess.run(["git", "-C", repository, "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", repository, "config", "user.name", "Test User"], check=True)
    (repository / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", repository, "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", repository, "commit", "-m", "base"], check=True, capture_output=True)
    subprocess.run(["git", "-C", repository, "remote", "add", "origin", str(remote)], check=True)
    (repository / "tracked.txt").write_text("changed\n", encoding="utf-8")
    monkeypatch.setattr(git_push, "ROOT", repository)
    assert git_push.validate_repository("meaningful message") == ["tracked.txt"]


def test_push_rejects_generated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(git_push, "ROOT", tmp_path)
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "artifact.o").write_bytes(b"object")
    with pytest.raises(RuntimeError, match="generated or binary"):
        git_push.validate_paths(["build/artifact.o"])


def test_push_dry_run_executes_checks_without_git_writes(monkeypatch):
    calls = []
    monkeypatch.setattr(git_push, "validate_repository", lambda message: ["source.cpp"])
    monkeypatch.setattr(git_push.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(
        sys, "argv",
        ["git_push.py", "--message", "test", "--component", "eda", "--dry-run"],
    )
    assert git_push.main() == 0
    assert [command[-1] for command in calls] == ["compile", "test", "check"]
    assert not any(command[:2] == ["git", "add"] for command in calls)
