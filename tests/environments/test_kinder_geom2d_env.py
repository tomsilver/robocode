"""Tests for kinder_geom2d_env.py."""

import numpy as np

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv


def test_motion2d_basic():
    """Basic Motion2D functionality: reset, step, get/set state."""
    env = KinderGeom2DEnv("kinder/Motion2D-p1-v0")
    env.action_space.seed(123)
    state, _ = env.reset(seed=123)
    assert env.observation_space.contains(state)

    # Step returns a valid observation with expected reward.
    action = env.action_space.sample()
    assert env.action_space.contains(action)
    next_state, reward, _, truncated, _ = env.step(action)
    assert env.observation_space.contains(next_state)
    assert reward == -1.0
    assert not truncated

    # get_state reflects the latest observation.
    assert np.array_equal(env.get_state(), next_state)

    # set_state restores a previous state; stepping from it is reproducible
    # up to float32 vectorization tolerance.
    env.set_state(state)
    assert np.array_equal(env.get_state(), state)
    replayed_state, _, _, _, _ = env.step(action)
    np.testing.assert_allclose(replayed_state, next_state, atol=1e-6)

    env.close()


def test_motion2d_sample_next_state():
    """sample_next_state produces a valid next state."""
    env = KinderGeom2DEnv("kinder/Motion2D-p1-v0")
    env.action_space.seed(42)
    state, _ = env.reset(seed=42)
    action = env.action_space.sample()

    rng = np.random.default_rng(0)
    next_state = env.sample_next_state(state, action, rng)
    assert env.observation_space.contains(next_state)
    assert not np.array_equal(next_state, state)

    env.close()
