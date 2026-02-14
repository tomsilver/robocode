"""Tests for random_approach.py."""

from robocode.approaches.random_approach import RandomApproach
from robocode.environments.maze_env import MazeEnv


def test_random_approach():
    """Tests for RandomApproach()."""
    env = MazeEnv(5, 8, 5, 8)
    sim = MazeEnv(5, 8, 5, 8)
    approach = RandomApproach(sim, seed=123)
    state, info = env.reset(seed=123)
    approach.reset(state, info)
    action = approach.step()
    assert env.action_space.contains(action)
