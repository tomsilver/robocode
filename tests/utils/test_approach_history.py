"""Tests for approach history utilities."""

# pylint: disable=redefined-outer-name

from __future__ import annotations

import json
import subprocess
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from robocode.environments.base_env import BaseEnv
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.primitives.check_action_collision import check_action_collision
from robocode.primitives.render_state import render_state
from robocode.utils.approach_history import get_snapshots, record_episodes

_APPROACH_V1 = """\
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self._action_space = action_space
    def reset(self, state, info):
        pass
    def get_action(self, state):
        return self._action_space.sample()
"""

_APPROACH_V2 = """\
import numpy as np

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self._action_space = action_space
    def reset(self, state, info):
        pass
    def get_action(self, state):
        return np.zeros(self._action_space.shape)
"""

_APPROACH_BROKEN = """\
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        pass
    def reset(self, state, info):
        raise RuntimeError("intentionally broken")
    def get_action(self, state):
        pass
"""


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        check=True,
    )


@pytest.fixture()
def env() -> BaseEnv:
    """Create a KinderGeom2DEnv for testing."""
    e = KinderGeom2DEnv("kinder/Motion2D-p0-v0")
    e.reset(seed=0)
    return e


@pytest.fixture()
def primitives(env: BaseEnv) -> dict:
    """Build a minimal primitives dict."""
    return {
        "check_action_collision": partial(check_action_collision, env),
        "render_state": partial(render_state, env),
    }


@pytest.fixture()
def sandbox_with_history(tmp_path: Path) -> Path:
    """Create a sandbox dir with git history containing two approach.py versions."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    _git(sandbox, "init")
    _git(sandbox, "config", "user.email", "test@test.com")
    _git(sandbox, "config", "user.name", "test")

    (sandbox / "approach.py").write_text(_APPROACH_V1)
    _git(sandbox, "add", "-A")
    _git(sandbox, "commit", "-m", "v1")

    (sandbox / "approach.py").write_text(_APPROACH_V2)
    _git(sandbox, "add", "-A")
    _git(sandbox, "commit", "-m", "v2")

    return sandbox


def test_get_snapshots(sandbox_with_history: Path) -> None:
    """get_snapshots returns one entry per commit."""
    snapshots = get_snapshots(sandbox_with_history)
    assert len(snapshots) == 2
    assert snapshots[0].version == 0
    assert snapshots[1].version == 1
    assert snapshots[0].commit_hash != snapshots[1].commit_hash
    assert snapshots[0].message == "v1"
    assert snapshots[1].message == "v2"


def test_get_snapshots_skips_commits_without_approach(tmp_path: Path) -> None:
    """Commits that don't contain approach.py are excluded."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    _git(sandbox, "init")
    _git(sandbox, "config", "user.email", "test@test.com")
    _git(sandbox, "config", "user.name", "test")

    # Initial commit with no approach.py (like sandbox setup).
    (sandbox / "CLAUDE.md").write_text("instructions")
    _git(sandbox, "add", "-A")
    _git(sandbox, "commit", "-m", "initial setup")

    # Second commit adds approach.py.
    (sandbox / "approach.py").write_text(_APPROACH_V1)
    _git(sandbox, "add", "-A")
    _git(sandbox, "commit", "-m", "first approach")

    snapshots = get_snapshots(sandbox)
    assert len(snapshots) == 1
    assert snapshots[0].message == "first approach"
    assert snapshots[0].version == 0


def test_record_episodes(
    sandbox_with_history: Path,
    env: BaseEnv,
    primitives: dict,
    tmp_path: Path,
) -> None:
    """record_episodes produces GIFs and metrics for each snapshot."""
    snapshots = get_snapshots(sandbox_with_history)
    output_dir = tmp_path / "output"

    records = record_episodes(
        snapshots,
        sandbox_with_history,
        env,
        primitives,
        seed=42,
        max_steps=5,
        output_dir=output_dir,
    )

    assert len(records) == 2

    history_dir = output_dir / "approach_history"
    for i in range(2):
        version_dir = history_dir / f"v{i:03d}"
        assert (version_dir / "episode.gif").exists()
        assert (version_dir / "metrics.json").exists()

        with open(version_dir / "metrics.json", encoding="utf-8") as f:
            metrics = json.load(f)
        assert "commit_hash" in metrics
        assert "message" in metrics
        assert "total_reward" in metrics
        assert "solved" in metrics

    assert (history_dir / "summary.json").exists()


def test_record_episodes_preserves_env_state(
    sandbox_with_history: Path,
    env: BaseEnv,
    primitives: dict,
    tmp_path: Path,
) -> None:
    """Env state is restored after recording all episodes."""
    # record_episodes resets the env with its seed internally; verify it
    # restores the post-reset state after running all episodes.
    env.reset(seed=42)
    state_before = env.get_state().copy()
    snapshots = get_snapshots(sandbox_with_history)

    record_episodes(
        snapshots,
        sandbox_with_history,
        env,
        primitives,
        seed=42,
        max_steps=5,
        output_dir=tmp_path / "output",
    )

    np.testing.assert_array_equal(env.get_state(), state_before)


def test_record_episodes_restores_sandbox_head(
    sandbox_with_history: Path,
    env: BaseEnv,
    primitives: dict,
    tmp_path: Path,
) -> None:
    """Sandbox is restored to original HEAD after recording."""
    head_before = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(sandbox_with_history),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    snapshots = get_snapshots(sandbox_with_history)
    record_episodes(
        snapshots,
        sandbox_with_history,
        env,
        primitives,
        seed=42,
        max_steps=5,
        output_dir=tmp_path / "output",
    )

    head_after = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(sandbox_with_history),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert head_before == head_after

    # HEAD must still be on a branch, not detached.
    ref = subprocess.run(
        ["git", "symbolic-ref", "HEAD"],
        cwd=str(sandbox_with_history),
        capture_output=True,
        check=False,
    )
    assert ref.returncode == 0, "HEAD is detached after record_episodes"


def test_record_episodes_handles_broken_snapshot(
    env: BaseEnv,
    primitives: dict,
    tmp_path: Path,
) -> None:
    """A broken approach version is recorded with error=True, not raised."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    _git(sandbox, "init")
    _git(sandbox, "config", "user.email", "test@test.com")
    _git(sandbox, "config", "user.name", "test")

    (sandbox / "approach.py").write_text(_APPROACH_V1)
    _git(sandbox, "add", "-A")
    _git(sandbox, "commit", "-m", "good version")

    (sandbox / "approach.py").write_text(_APPROACH_BROKEN)
    _git(sandbox, "add", "-A")
    _git(sandbox, "commit", "-m", "broken version")

    snapshots = get_snapshots(sandbox)
    output_dir = tmp_path / "output"

    records = record_episodes(
        snapshots,
        sandbox,
        env,
        primitives,
        seed=42,
        max_steps=5,
        output_dir=output_dir,
    )

    assert len(records) == 2

    # First version should succeed normally.
    assert records[0]["solved"] in (True, False)
    assert "error" not in records[0]

    # Broken version should be recorded as an error.
    assert "RuntimeError" in records[1]["error"]
    assert records[1]["solved"] is False
    assert records[1]["num_steps"] == 0
