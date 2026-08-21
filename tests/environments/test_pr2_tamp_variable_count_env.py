"""Tests for pr2_tamp_variable_count_env.py."""

import numpy as np
import pytest
from relational_structs import ObjectCentricState

from robocode.environments.pr2_tamp_variable_count_env import (
    BLOCK_PREFIX,
    PR2PackedVariableCountEnv,
)
from robocode.environments.variable_count import VariableCountEnv


@pytest.fixture(name="env")
def _env():
    env = PR2PackedVariableCountEnv(design_counts=[1, 2], eval_counts=[1, 2, 3])
    yield env
    env.close()


def _num_blocks(state: ObjectCentricState) -> int:
    return sum(1 for n in state.get_object_names() if n.startswith(BLOCK_PREFIX))


def test_implements_variable_count_contract(env: PR2PackedVariableCountEnv) -> None:
    """The runner's count lifecycle keys off the shared base class."""
    assert isinstance(env, VariableCountEnv)
    assert env.design_counts == [1, 2]
    assert env.eval_counts == [1, 2, 3]
    assert env.max_steps_for_count(3) > env.max_steps_for_count(1)


def test_pinned_count_controls_instance_size(env: PR2PackedVariableCountEnv) -> None:
    """A pinned count reaches the backend and is reported back in info."""
    for count in env.eval_counts:
        state, info = env.reset(seed=0, options={"object_count": count})
        assert _num_blocks(state) == count
        assert info["object_count"] == count
        assert env.current_count == count


def test_unpinned_reset_stays_in_design_range(env: PR2PackedVariableCountEnv) -> None:
    """A held-out count reaches the env only through an explicit pin."""
    for seed in range(6):
        env.reset(seed=seed)
        assert env.current_count in env.design_counts


def test_observation_is_count_invariant(env: PR2PackedVariableCountEnv) -> None:
    """Every instance shares one schema, so one program spans all counts."""
    small, _ = env.reset(seed=0, options={"object_count": 1})
    large, _ = env.reset(seed=0, options={"object_count": 3})
    assert env.observation_space.contains(small)
    assert env.observation_space.contains(large)
    small_types = {o.type.name for o in small}
    assert small_types == {o.type.name for o in large}
    assert small_types == {"robot", "surface", "block"}


def test_step_and_state_roundtrip(env: PR2PackedVariableCountEnv) -> None:
    """Stepping advances the instance and set_state restores it."""
    env.action_space.seed(0)
    saved, _ = env.reset(seed=2, options={"object_count": 2})
    count = env.current_count
    for _ in range(10):
        _, reward, terminated, _, info = env.step(env.action_space.sample())
        assert reward == -1.0
        assert not terminated
        assert info["object_count"] == count
    # pylint: disable=protected-access
    moved = env._to_box(env.get_state(), count)
    assert not np.allclose(moved, env._to_box(saved, count))
    env.set_state(saved)
    restored = env._to_box(env.get_state(), count)
    assert np.allclose(restored, env._to_box(saved, count), atol=1e-5)
