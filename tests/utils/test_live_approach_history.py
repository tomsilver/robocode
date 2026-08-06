"""Tests for the background per-commit evaluation watcher."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from robocode.utils.live_approach_history import (
    _child_env,
    _CommitEvalWatcher,
    _run_eval_subprocess,
    live_commit_eval,
)

_OVERRIDES = ["approach=agentic", "environment=small_maze"]


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


def _init_sandbox(sandbox: Path) -> None:
    sandbox.mkdir(parents=True)
    _git(sandbox, "init")
    _git(sandbox, "config", "user.email", "t@t")
    _git(sandbox, "config", "user.name", "t")


def _commit_approach(sandbox: Path, body: str, message: str) -> None:
    (sandbox / "approach.py").write_text(body, encoding="utf-8")
    _git(sandbox, "add", "-A")
    _git(sandbox, "commit", "-m", message)


def _eval_recorder(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Intercept the child eval seam; real git/tar (get_snapshots, export) still run."""
    calls: list[list[str]] = []

    def fake_eval(cmd, _env):
        calls.append([str(part) for part in cmd])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "robocode.utils.live_approach_history._run_eval_subprocess", fake_eval
    )
    return calls


def test_disabled_is_a_noop(tmp_path: Path, monkeypatch) -> None:
    """With the flag off, nothing runs and no history dir appears."""
    calls = _eval_recorder(monkeypatch)
    with live_commit_eval(
        enabled=False,
        sandbox_dir=tmp_path / "sandbox",
        output_dir=tmp_path / "out",
        task_overrides=_OVERRIDES,
        eval_seed=42,
    ):
        pass
    assert not calls
    assert not (tmp_path / "out" / "approach_history").exists()


def test_scan_evaluates_each_new_commit_once(tmp_path: Path, monkeypatch) -> None:
    """A rescan picks up only commits it has not seen, and exports each version."""
    calls = _eval_recorder(monkeypatch)
    sandbox = tmp_path / "sandbox"
    _init_sandbox(sandbox)
    _commit_approach(sandbox, "v0", "first")
    output_dir = tmp_path / "out"

    watcher = _CommitEvalWatcher(sandbox, output_dir, _OVERRIDES, eval_seed=7)
    watcher._scan_and_eval()  # pylint: disable=protected-access
    assert len(calls) == 1
    watcher._scan_and_eval()  # nothing new  # pylint: disable=protected-access
    assert len(calls) == 1

    _commit_approach(sandbox, "v1", "second")
    watcher._scan_and_eval()  # pylint: disable=protected-access
    assert len(calls) == 2

    # Each version's tree is exported for the loader, and the command pins the suite
    # and points the loader/output at that version.
    for version in ("v000", "v001"):
        exported = output_dir / "approach_history" / version / "sandbox" / "approach.py"
        assert exported.exists()
    for cmd in calls:
        assert "eval_seed=7" in cmd
        assert "live_approach_history=false" in cmd
        assert "record_approach_history=false" in cmd
        assert "render_videos=false" in cmd
        assert any(a.startswith("approach.load_dir=") for a in cmd)
        assert any(a.startswith("hydra.run.dir=") for a in cmd)


def test_drain_on_exit_scores_all_commits(tmp_path: Path, monkeypatch) -> None:
    """The watcher thread drains the backlog when the block exits."""
    calls = _eval_recorder(monkeypatch)
    sandbox = tmp_path / "sandbox"
    _init_sandbox(sandbox)
    _commit_approach(sandbox, "v0", "first")
    _commit_approach(sandbox, "v1", "second")
    output_dir = tmp_path / "out"

    with live_commit_eval(
        enabled=True,
        sandbox_dir=sandbox,
        output_dir=output_dir,
        task_overrides=_OVERRIDES,
        eval_seed=42,
    ):
        pass  # stop() drains: every commit present at exit is scored

    assert len(calls) == 2


def test_missing_repo_is_tolerated(tmp_path: Path, monkeypatch) -> None:
    """A scan before the sandbox git repo exists does nothing, without raising."""
    calls = _eval_recorder(monkeypatch)
    watcher = _CommitEvalWatcher(
        tmp_path / "sandbox", tmp_path / "out", _OVERRIDES, eval_seed=42
    )
    watcher._scan_and_eval()  # pylint: disable=protected-access
    assert not calls


def test_child_env_caps_threads_and_drops_telemetry(monkeypatch) -> None:
    """Thread limits are forced to 1 even when inherited high; telemetry is dropped."""
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "32")
    monkeypatch.setenv("ROBOCODE_TELEMETRY", "/tmp/sink.jsonl")
    env = _child_env()
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["OMP_NUM_THREADS"] == "1"
    assert "ROBOCODE_TELEMETRY" not in env


def test_eval_subprocess_timeout_is_reported_not_raised(monkeypatch) -> None:
    """A child that exceeds the ceiling is killed and reported failed, not raised."""

    def _timeout(cmd, *_a, **_k):
        raise subprocess.TimeoutExpired(cmd, timeout=1)

    monkeypatch.setattr(subprocess, "run", _timeout)
    result = _run_eval_subprocess(["run_experiment.py"], {})
    assert result.returncode != 0
