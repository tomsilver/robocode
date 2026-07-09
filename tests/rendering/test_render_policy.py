"""Tests for the render_policy rendering helper."""

# pylint: disable=redefined-outer-name

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium.spaces import Box

from robocode.environments.base_env import BaseEnv
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.primitives.check_action_collision import check_action_collision
from robocode.rendering.render_policy import render_policy
from robocode.rendering.render_state import render_state


class _RecordingEnv:
    """A minimal env that records the ``options`` passed to each ``reset``."""

    def __init__(self) -> None:
        space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = space
        self.observation_space = space
        self.reset_options: list[Any] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Record the reset options and return a trivial observation."""
        del seed
        self.reset_options.append(options)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Terminate immediately (one-frame rollout)."""
        del action
        return np.zeros(1, dtype=np.float32), 0.0, True, False, {}

    def get_state(self) -> np.ndarray:
        """Return a trivial snapshot."""
        return np.zeros(1, dtype=np.float32)

    def set_state(self, state: Any) -> None:
        """Ignore the restored state (no internal state to set)."""
        del state

    def render(self) -> np.ndarray:
        """Return a tiny RGB frame."""
        return np.zeros((2, 2, 3), dtype=np.uint8)


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
    env: BaseEnv, primitives: dict, approach_dir: Path, tmp_path: Path
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
    env: BaseEnv, primitives: dict, approach_dir: Path, tmp_path: Path
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
    env: BaseEnv, primitives: dict, approach_dir: Path, tmp_path: Path
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


def test_object_count_pins_the_rollout_reset(
    approach_dir: Path, tmp_path: Path
) -> None:
    """A variable-count env's rollout is reset at the pinned object_count, so the
    rendered episode matches the scored (seed, count) instance."""
    env = _RecordingEnv()
    render_policy(
        env,
        {},
        approach_dir=str(approach_dir / "sandbox"),
        seed=7,
        output_dir=str(tmp_path / "frames"),
        max_steps=2,
        object_count=3,
    )
    assert {"object_count": 3} in env.reset_options


def test_no_object_count_leaves_reset_unpinned(
    approach_dir: Path, tmp_path: Path
) -> None:
    """Without object_count the reset passes no options (fixed-count behavior)."""
    env = _RecordingEnv()
    render_policy(
        env,
        {},
        approach_dir=str(approach_dir / "sandbox"),
        seed=7,
        output_dir=str(tmp_path / "frames"),
        max_steps=2,
    )
    assert env.reset_options == [None]
