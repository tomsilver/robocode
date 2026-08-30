"""Tests for the read-only view handed to generated code."""

from typing import Any

import numpy as np
import pytest

from robocode.environments.pr2_tamp_env import PR2PackedEnv
from robocode.primitives import build_primitives
from robocode.utils.scored_env_guard import ScoredEnvMutationError, readonly_view


@pytest.fixture(name="env")
def _env():
    env = PR2PackedEnv(num_blocks=1)
    env.reset(seed=0)
    yield env
    env.close()


def test_reads_pass_through(env: PR2PackedEnv) -> None:
    """Everything an approach legitimately needs still works."""
    view = readonly_view(env)
    assert view.robot == env.robot
    assert view.blocks == env.blocks
    assert np.array_equal(view.get_state(), env.get_state())


@pytest.mark.parametrize(
    "name", ["set_state", "sample_next_state", "reset", "step", "close"]
)
def test_mutators_are_blocked(env: PR2PackedEnv, name: str) -> None:
    """An approach must reach the goal through its actions, not by moving the env."""
    view = readonly_view(env)
    with pytest.raises(ScoredEnvMutationError):
        getattr(view, name)()


def test_attribute_assignment_is_blocked(env: PR2PackedEnv) -> None:
    """Rebinding an attribute would desync the env from the episode being scored."""
    view = readonly_view(env)
    with pytest.raises(ScoredEnvMutationError):
        view._attachment = None  # pylint: disable=protected-access


def test_isinstance_still_dispatches(env: PR2PackedEnv) -> None:
    """Primitives pick an implementation with isinstance, so the type must survive."""
    assert isinstance(readonly_view(env), PR2PackedEnv)


def test_readonly_view_is_idempotent(env: PR2PackedEnv) -> None:
    """Wrapping twice must not stack proxies."""
    view = readonly_view(env)
    assert readonly_view(view) is view


def test_env_bound_primitive_cannot_reach_the_scored_env(env: PR2PackedEnv) -> None:
    """The closure is the one path from generated code to the scored environment.

    ``check_action_collision`` is a partial over the live env, so an approach granted
    it could otherwise pull the env out of ``fn.args`` and move it into a solved
    state. This is what the guard exists for.
    """
    primitive: Any = build_primitives(env, ["check_action_collision"])[
        "check_action_collision"
    ]
    bound_env = primitive.args[0]
    with pytest.raises(ScoredEnvMutationError):
        bound_env.set_state(env.get_state())


def test_private_environments_are_unaffected() -> None:
    """A program planning in an environment it built itself is not restricted.

    This is the ordinary way to write a TAMP policy, and it cannot affect scoring. The
    source scan this guard replaced rejected it.
    """
    private = PR2PackedEnv(num_blocks=1)
    try:
        state, _ = private.reset(seed=0)
        private.step(np.zeros(11, dtype=np.float32))
        private.set_state(state)
        assert np.allclose(private.get_state(), state, atol=1e-5)
    finally:
        private.close()
