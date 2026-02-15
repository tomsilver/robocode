"""Tests for random_approach.py."""

from functools import partial

from robocode.approaches.random_approach import RandomApproach
from robocode.environments.maze_env import MazeEnv
from robocode.primitives.check_action_collision import check_action_collision


def test_random_approach():
    """Tests for RandomApproach()."""
    env = MazeEnv(5, 8, 5, 8)
    approach = RandomApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=123,
        primitives={"check_action_collision": partial(check_action_collision, env)},
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
        primitives={"check_action_collision": partial(check_action_collision, env)},
    )
    approach.train()

    state, info = env.reset(seed=99)
    approach.reset(state, info)
    action = approach.step()
    assert env.action_space.contains(action)
