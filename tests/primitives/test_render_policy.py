"""Tests for render_policy primitive."""

# pylint: disable=redefined-outer-name

from __future__ import annotations

from functools import partial
from pathlib import Path

import gymnasium
import numpy as np
import pytest

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.primitives.check_action_collision import check_action_collision
from robocode.primitives.render_policy import render_policy
from robocode.primitives.render_state import render_state


@pytest.fixture()
def env() -> gymnasium.Env:
    """Create a KinderGeom2DEnv for testing."""
    e = KinderGeom2DEnv("kinder/Motion2D-p0-v0")
    e.reset(seed=0)
    return e


@pytest.fixture()
def primitives(env: gymnasium.Env) -> dict:
    """Build a minimal primitives dict."""
    return {
        "check_action_collision": partial(check_action_collision, env),
        "render_state": partial(render_state, env),
    }


@pytest.fixture()
def approach_dir(tmp_path: Path) -> Path:
    """Create a dummy approach.py in a sandbox dir."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "approach.py").write_text(
        "class GeneratedApproach:\n"
        "    def __init__(self, action_space, observation_space, primitives):\n"
        "        self._action_space = action_space\n"
        "    def reset(self, state, info):\n"
        "        pass\n"
        "    def get_action(self, state):\n"
        "        return self._action_space.sample()\n"
    )
    return tmp_path


def test_returns_frame_files(
    env: gymnasium.Env, primitives: dict, approach_dir: Path, tmp_path: Path
) -> None:
    """render_policy returns PNG filenames that exist on disk."""
    output_dir = tmp_path / "frames"
    files = render_policy(
        env,
        primitives,
        approach_dir=str(approach_dir / "sandbox"),
        seed=42,
        output_dir=str(output_dir),
        max_steps=5,
    )
    assert len(files) > 0
    for f in files:
        assert (output_dir / f).exists()
        assert f.endswith(".png")


def test_state_preserved(
    env: gymnasium.Env, primitives: dict, approach_dir: Path, tmp_path: Path
) -> None:
    """Env state is preserved after render_policy runs."""
    state_before = env.get_state().copy()
    render_policy(
        env,
        primitives,
        approach_dir=str(approach_dir / "sandbox"),
        seed=42,
        output_dir=str(tmp_path / "frames"),
        max_steps=5,
    )
    np.testing.assert_array_equal(env.get_state(), state_before)


def test_max_frames_limits_output(
    env: gymnasium.Env, primitives: dict, approach_dir: Path, tmp_path: Path
) -> None:
    """max_frames caps the number of saved PNGs."""
    files = render_policy(
        env,
        primitives,
        approach_dir=str(approach_dir / "sandbox"),
        seed=42,
        output_dir=str(tmp_path / "frames"),
        max_steps=20,
        max_frames=3,
    )
    assert len(files) == 3
