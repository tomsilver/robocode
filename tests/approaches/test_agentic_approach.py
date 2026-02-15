"""Tests for agentic_approach.py."""

from functools import partial

import pytest

from robocode.approaches.agentic_approach import AgenticApproach
from robocode.environments.maze_env import MazeEnv
from robocode.primitives.check_action_collision import check_action_collision


def test_agentic_approach_fallback():
    """Without training, AgenticApproach falls back to random actions."""
    env = MazeEnv(5, 8, 5, 8)
    approach = AgenticApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=123,
        primitives={"check_action_collision": partial(check_action_collision, env)},
    )
    state, info = env.reset(seed=123)
    approach.reset(state, info)
    action = approach.step()
    assert env.action_space.contains(action)


def test_agentic_approach_with_generated():
    """AgenticApproach delegates to a generated approach when loaded."""
    env = MazeEnv(5, 8, 5, 8)
    approach = AgenticApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=123,
        primitives={"check_action_collision": partial(check_action_collision, env)},
        output_dir="/tmp/test_agentic",
    )

    # Write a minimal generated approach and load it.
    sandbox_dir = approach._output_dir / "sandbox"  # pylint: disable=protected-access
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    approach_file = sandbox_dir / "approach.py"
    approach_file.write_text(
        "class GeneratedApproach:\n"
        "    def __init__(self, action_space,"
        " observation_space,\n"
        "                 primitives):\n"
        "        self._n = action_space.n\n"
        "        self._step_count = 0\n"
        "    def reset(self, state, info):\n"
        "        self._step_count = 0\n"
        "    def get_action(self, state):\n"
        "        self._step_count += 1\n"
        "        return 0\n"
    )
    approach._load_generated(approach_file)  # pylint: disable=protected-access

    state, info = env.reset(seed=123)
    approach.reset(state, info)
    action = approach.step()
    assert action == 0
    assert env.action_space.contains(action)


def test_load_dir_skips_agent(tmp_path):
    """When load_dir is set, train() loads from it without calling the agent."""
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    approach_file = sandbox_dir / "approach.py"
    approach_file.write_text(
        "class GeneratedApproach:\n"
        "    def __init__(self, action_space,"
        " observation_space,\n"
        "                 primitives):\n"
        "        pass\n"
        "    def reset(self, state, info):\n"
        "        pass\n"
        "    def get_action(self, state):\n"
        "        return 0\n"
    )

    env = MazeEnv(5, 8, 5, 8)
    approach = AgenticApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=42,
        primitives={"check_action_collision": partial(check_action_collision, env)},
        load_dir=str(tmp_path),
    )
    approach.train()

    state, info = env.reset(seed=42)
    approach.reset(state, info)
    assert approach.step() == 0


def test_load_dir_missing_file_raises(tmp_path):
    """When load_dir points to a directory without approach.py, raise
    FileNotFoundError."""
    env = MazeEnv(5, 8, 5, 8)
    approach = AgenticApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=42,
        primitives={"check_action_collision": partial(check_action_collision, env)},
        load_dir=str(tmp_path),
    )
    with pytest.raises(FileNotFoundError):
        approach.train()
