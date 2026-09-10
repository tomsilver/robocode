"""Tests for rovers_variable_count_env.py."""

import pytest

from robocode.environments.rovers_variable_count_env import RoversVariableCountEnv
from robocode.environments.variable_count import VariableCountEnv


@pytest.fixture(name="env")
def _env():
    env = RoversVariableCountEnv(design_counts=[1, 2], eval_counts=[1, 2, 3, 4])
    yield env
    env.close()


def test_implements_variable_count_contract(env: RoversVariableCountEnv) -> None:
    """The runner's count-sweep lifecycle keys off this contract."""
    assert isinstance(env, VariableCountEnv)
    assert env.design_counts == [1, 2]
    assert env.eval_counts == [1, 2, 3, 4]


def test_pinned_count_controls_instance_size(env: RoversVariableCountEnv) -> None:
    """A pinned count decides how many objectives the instance has."""
    for count in env.eval_counts:
        state, info = env.reset(seed=0, options={"object_count": count})
        assert info["object_count"] == count
        names = state.get_object_names()
        assert sum(1 for n in names if n.startswith("objective")) == count
        # The rest of the cast is the same whatever the count.
        assert sum(1 for n in names if n.startswith("rover")) == 2
        assert "lander" in names


def test_unpinned_reset_stays_in_design_range(env: RoversVariableCountEnv) -> None:
    """An unpinned reset never reaches a held-out count."""
    for seed in range(8):
        _, info = env.reset(seed=seed)
        assert info["object_count"] in env.design_counts


def test_state_roundtrip_across_counts(env: RoversVariableCountEnv) -> None:
    """Object-centric states convert to the backend layout and back."""
    for count in env.eval_counts:
        state, _ = env.reset(seed=1, options={"object_count": count})
        # pylint: disable=protected-access
        backend = env.current_backend
        again = env._to_object_centric(
            env._to_box(state, count, backend), count, backend
        )
        assert set(again.get_object_names()) == set(state.get_object_names())
        assert env._count_from_state(state) == count
        env.set_state(state)
        assert env.current_count == count


def test_step_preserves_the_count(env: RoversVariableCountEnv) -> None:
    """Stepping reports the same count it reset with."""
    _, info = env.reset(seed=0, options={"object_count": 2})
    _, _, _, _, step_info = env.step(env.action_space.sample())
    assert step_info["object_count"] == info["object_count"] == 2


def test_budget_grows_with_the_count(env: RoversVariableCountEnv) -> None:
    """Each objective is another errand, so the budget has to pay for it."""
    budgets = [env.max_steps_for_count(c) for c in env.eval_counts]
    assert budgets == sorted(budgets)
    assert budgets[-1] > budgets[0]


def test_rejects_counts_the_scene_cannot_stage(env: RoversVariableCountEnv) -> None:
    """There are four mounds, so a fifth objective has nowhere of its own to stand."""
    with pytest.raises(ValueError, match="mounds"):
        env.reset(seed=0, options={"object_count": 5})


def test_rejects_empty_instances() -> None:
    """Every instance has at least one objective to photograph."""
    with pytest.raises(ValueError, match="at least one objective"):
        RoversVariableCountEnv(design_counts=[0], eval_counts=[1]).close()


def test_description_never_names_the_evaluation_counts(
    env: RoversVariableCountEnv,
) -> None:
    """Which counts an approach is scored on is the experimenter's to know."""
    for card in (env.env_description, env.env_description_blackbox):
        assert "design_counts" not in card
        assert "eval_counts" not in card
    assert "Source Code" not in env.env_description_blackbox
