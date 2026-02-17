"""Tests for kinder_geom3d_env.py."""

import numpy as np
import pytest

from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv

ALL_3D_ENV_IDS = [
    "kinder/Motion3D-v0",
    "kinder/Obstruction3D-o0-v0",
    "kinder/Obstruction3D-o2-v0",
    "kinder/Obstruction3D-o4-v0",
    "kinder/Shelf3D-o1-v0",
    "kinder/Shelf3D-o3-v0",
    "kinder/Shelf3D-o5-v0",
    "kinder/Transport3D-o1-v0",
    "kinder/Transport3D-o2-v0",
    "kinder/Packing3D-p1-v0",
    "kinder/Packing3D-p2-v0",
    "kinder/Packing3D-p3-v0",
]


@pytest.mark.parametrize("env_id", ALL_3D_ENV_IDS)
def test_kinder_geom3d_basic(env_id: str) -> None:
    """Basic functionality: reset, step, get/set state."""
    env = KinderGeom3DEnv(env_id)
    env.action_space.seed(123)
    state, _ = env.reset(seed=123)
    assert env.observation_space.contains(state)

    # Step returns a valid observation.
    action = env.action_space.sample()
    assert env.action_space.contains(action)
    next_state, _reward, _terminated, truncated, _ = env.step(action)
    assert env.observation_space.contains(next_state)
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


@pytest.mark.parametrize("env_id", ALL_3D_ENV_IDS)
def test_kinder_geom3d_sample_next_state(env_id: str) -> None:
    """sample_next_state produces a valid next state."""
    env = KinderGeom3DEnv(env_id)
    env.action_space.seed(42)
    state, _ = env.reset(seed=42)
    action = env.action_space.sample()

    rng = np.random.default_rng(0)
    next_state = env.sample_next_state(state, action, rng)
    assert env.observation_space.contains(next_state)

    env.close()
