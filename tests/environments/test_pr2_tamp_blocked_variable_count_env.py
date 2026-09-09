"""Tests for pr2_tamp_blocked_variable_count_env.py.

The wrapper differs from the ``packed`` one in that a count of zero is a real instance
rather than a degenerate one, so most of these pin behaviour at zero.
"""

import pytest

from robocode.environments.pr2_tamp_blocked_variable_count_env import (
    PR2BlockedVariableCountEnv,
)
from robocode.environments.variable_count import VariableCountEnv


@pytest.fixture(name="env")
def _env():
    env = PR2BlockedVariableCountEnv(design_counts=[0, 1], eval_counts=[0, 1, 2])
    yield env
    env.close()


def test_implements_variable_count_contract(env: PR2BlockedVariableCountEnv) -> None:
    """The runner's count-sweep lifecycle keys off this contract."""
    assert isinstance(env, VariableCountEnv)
    assert env.design_counts == [0, 1]
    assert env.eval_counts == [0, 1, 2]


def test_pinned_count_controls_instance_size(env: PR2BlockedVariableCountEnv) -> None:
    """A pinned count decides how many spares the instance has."""
    for count in env.eval_counts:
        state, info = env.reset(seed=0, options={"object_count": count})
        assert info["object_count"] == count
        assert env.current_count == count
        greens = [n for n in state.get_object_names() if n.startswith("green")]
        # One penned block plus the spares.
        assert len(greens) == count + 1


def test_zero_spares_is_a_real_count(env: PR2BlockedVariableCountEnv) -> None:
    """Zero is the hardest instance, not a missing one, so it must round-trip."""
    state, info = env.reset(seed=0, options={"object_count": 0})
    assert info["object_count"] == 0
    assert env._count_from_state(state) == 0  # pylint: disable=protected-access
    names = set(state.get_object_names())
    assert "green0" in names and "green1" not in names
    assert "blocker" in names


def test_unpinned_reset_stays_in_design_range(
    env: PR2BlockedVariableCountEnv,
) -> None:
    """An unpinned reset never reaches a held-out count."""
    for seed in range(8):
        _, info = env.reset(seed=seed)
        assert info["object_count"] in env.design_counts


def test_state_roundtrip_across_counts(env: PR2BlockedVariableCountEnv) -> None:
    """Object-centric states convert to the backend layout and back."""
    for count in env.eval_counts:
        state, _ = env.reset(seed=1, options={"object_count": count})
        # pylint: disable=protected-access
        flat = env._to_box(state, count)
        again = env._to_object_centric(flat, count)
        assert set(again.get_object_names()) == set(state.get_object_names())
        env.set_state(state)
        assert env.current_count == count


def test_step_preserves_the_count(env: PR2BlockedVariableCountEnv) -> None:
    """Stepping reports the same count it reset with."""
    _, info = env.reset(seed=0, options={"object_count": 1})
    _, _, _, _, step_info = env.step(env.action_space.sample())
    assert step_info["object_count"] == info["object_count"] == 1


def test_step_budget_is_mostly_flat_in_the_count(
    env: PR2BlockedVariableCountEnv,
) -> None:
    """Spares are alternatives, not extra work, so the budget barely grows.

    The contrast with ``packed`` is the point: there every extra block is another
    pick and place, here the goal is satisfied by one block whatever the count.
    """
    budgets = [env.max_steps_for_count(c) for c in (0, 4)]
    assert budgets[1] > budgets[0]
    assert budgets[1] < 2 * budgets[0]


def test_description_never_names_the_evaluation_counts(
    env: PR2BlockedVariableCountEnv,
) -> None:
    """Which counts an approach is scored on is the experimenter's to know."""
    for card in (env.env_description, env.env_description_blackbox):
        assert "design_counts" not in card
        assert "eval_counts" not in card
        for count in env.eval_counts:
            assert f"{count} spare" not in card
    assert "Source Code" not in env.env_description_blackbox


def test_rejects_negative_counts() -> None:
    """The count is a spare-block count, so negative is a configuration error."""
    with pytest.raises(ValueError, match="negative"):
        PR2BlockedVariableCountEnv(design_counts=[-1], eval_counts=[0]).close()
