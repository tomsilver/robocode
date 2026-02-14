"""Tests for maze_env.py."""

from robocode.environments.maze_env import MazeEnv, _MazeAction, _MazeState


def test_maze_env():
    """Tests for MazeEnv()."""
    env = MazeEnv(5, 8, 5, 8)
    state, _ = env.reset(seed=123)
    assert isinstance(state, _MazeState)
    action = env.action_space.sample()
    assert isinstance(action, _MazeAction)
    next_state, _, _, _, _ = env.step(action)
    assert isinstance(next_state, _MazeState)
    assert state != next_state
    env.set_state(state)
    assert env.get_state() == state
