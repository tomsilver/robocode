"""Tests for episode utilities."""

# pylint: disable=redefined-outer-name

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pytest
from gymnasium import Env
from gymnasium.spaces import Box

from robocode.approaches.base_approach import BaseApproach
from robocode.utils.episode import (
    load_generated_approach,
    run_episode,
    save_frames,
    save_video,
)


@pytest.fixture()
def dummy_approach_file(tmp_path: Path) -> Path:
    """Write a minimal GeneratedApproach to a temp file."""
    approach_py = tmp_path / "approach.py"
    approach_py.write_text(
        "class GeneratedApproach:\n"
        "    def __init__(self, action_space, observation_space, primitives):\n"
        "        self._action_space = action_space\n"
        "        self._primitives = primitives\n"
        "    def reset(self, state, info):\n"
        "        pass\n"
        "    def get_action(self, state):\n"
        "        return self._action_space.sample()\n"
    )
    return approach_py


@pytest.fixture()
def sample_frames() -> list[np.ndarray]:
    """Create a list of small random RGB frames."""
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (8, 8, 3), dtype=np.uint8) for _ in range(5)]


def test_load_generated_approach(dummy_approach_file: Path) -> None:
    """load_generated_approach returns an instance with expected methods."""
    action_space = Box(low=-1, high=1, shape=(2,))
    obs_space = Box(low=0, high=1, shape=(4,))
    approach = load_generated_approach(dummy_approach_file, action_space, obs_space, {})
    assert hasattr(approach, "get_action")
    assert hasattr(approach, "reset")


def test_load_generated_approach_receives_primitives(
    dummy_approach_file: Path,
) -> None:
    """Primitives dict is passed through to the loaded approach."""
    action_space = Box(low=-1, high=1, shape=(2,))
    obs_space = Box(low=0, high=1, shape=(4,))
    prims = {"my_prim": lambda: None}
    approach = load_generated_approach(
        dummy_approach_file, action_space, obs_space, prims
    )
    assert approach._primitives is prims  # pylint: disable=protected-access


def test_load_cleans_sys_path(dummy_approach_file: Path) -> None:
    """sys.path is cleaned up after loading."""
    action_space = Box(low=-1, high=1, shape=(2,))
    obs_space = Box(low=0, high=1, shape=(4,))
    sandbox_dir = str(dummy_approach_file.parent.resolve())

    load_generated_approach(dummy_approach_file, action_space, obs_space, {})
    assert sandbox_dir not in sys.path


def test_run_episode_returns_final_state() -> None:
    """run_episode returns the observation the episode ended on."""

    class _CountEnv(Env):
        def __init__(self) -> None:
            self.observation_space = Box(0.0, 10.0, shape=(1,), dtype=np.float32)
            self.action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
            self._pos = 0.0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._pos = 0.0
            return np.array([self._pos], dtype=np.float32), {}

        def step(self, action):
            self._pos += 1.0
            obs = np.array([self._pos], dtype=np.float32)
            return obs, 0.0, self._pos >= 3.0, False, {}

        def render(self):
            return None

    class _NoopApproach(BaseApproach[Any, Any]):
        def _get_action(self) -> Any:
            return np.zeros(1, dtype=np.float32)

    env = _CountEnv()
    approach = _NoopApproach(env.action_space, env.observation_space, 0, {})
    metrics, _, final_state = run_episode(env, approach, seed=0, max_steps=10)
    assert metrics["solved"]
    assert final_state == np.array([3.0], dtype=np.float32)


def test_save_frames_creates_pngs(
    tmp_path: Path, sample_frames: list[np.ndarray]
) -> None:
    """save_frames writes PNG files and returns their names."""
    out = tmp_path / "frames"
    filenames = save_frames(sample_frames, out)
    assert len(filenames) == 5
    for f in filenames:
        assert f.endswith(".png")
        assert (out / f).exists()


def test_save_frames_max_frames(
    tmp_path: Path, sample_frames: list[np.ndarray]
) -> None:
    """save_frames respects the max_frames limit."""
    out = tmp_path / "frames"
    filenames = save_frames(sample_frames, out, max_frames=2)
    assert len(filenames) == 2


def test_save_frames_content_readable(
    tmp_path: Path, sample_frames: list[np.ndarray]
) -> None:
    """Saved frames can be read back as valid images."""
    out = tmp_path / "frames"
    filenames = save_frames(sample_frames, out)
    img = iio.imread(str(out / filenames[0]))
    assert img.shape[:2] == (8, 8)


def test_save_video_creates_gif(
    tmp_path: Path, sample_frames: list[np.ndarray]
) -> None:
    """save_video writes a GIF file."""
    gif_path = tmp_path / "test.gif"
    save_video(sample_frames, gif_path)
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
