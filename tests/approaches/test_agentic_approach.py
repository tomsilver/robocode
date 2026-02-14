"""Tests for agentic_approach.py."""

from robocode.approaches.agentic_approach import AgenticApproach
from robocode.environments.maze_env import MazeEnv


def test_agentic_approach_fallback():
    """Without training, AgenticApproach falls back to random actions."""
    env = MazeEnv(5, 8, 5, 8)
    approach = AgenticApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=123,
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
        output_dir="/tmp/test_agentic",
    )

    # Write a minimal generated approach and load it.
    sandbox_dir = approach._output_dir / "sandbox"  # pylint: disable=protected-access
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    approach_file = sandbox_dir / "approach.py"
    approach_file.write_text(
        "class GeneratedApproach:\n"
        "    def __init__(self, action_space, observation_space):\n"
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
