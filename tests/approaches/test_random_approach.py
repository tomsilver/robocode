"""Tests for random_approach.py."""

from robocode.approaches.random_approach import RandomApproach
from robocode.environments.maze_env import MazeEnv


def test_random_approach():
    """Tests for RandomApproach()."""
    env = MazeEnv(5, 8, 5, 8)
    approach = RandomApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=123,
    )
    state, info = env.reset(seed=123)
    approach.reset(state, info)
    action = approach.step()
    assert env.action_space.contains(action)


def test_random_approach_train():
    """Test that train is a no-op and the approach still works afterward."""
    env = MazeEnv(5, 8, 5, 8)
    approach = RandomApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=456,
    )
    train_states = [env.reset(seed=s) for s in [10, 20, 30]]
    approach.train(train_states)

    state, info = env.reset(seed=99)
    approach.reset(state, info)
    action = approach.step()
    assert env.action_space.contains(action)
