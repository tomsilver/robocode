"""Tests for random_approach.py."""

from copy import deepcopy

import gymnasium

from robocode.approaches.random_approach import RandomApproach


def test_random_approach():
    """Tests for RandomApproach()."""
    env = gymnasium.make("Taxi-v3")
    action_space = deepcopy(env.action_space)
    action_space.seed(123)
    approach = RandomApproach(action_space, seed=123)
    obs, info = env.reset(seed=123)
    approach.reset(obs, info)
    action = approach.step()
    assert env.action_space.contains(action)
