"""Tests for the PR2 blocked oracle approach.

These are the solvability check for the environment: if the oracle cannot reach the
goal within the step budget, the task is not solvable as specified and no result from
a synthesized approach would mean anything.
"""

from typing import Any

import pytest

from robocode.environments.pr2_tamp_blocked_env import PR2BlockedEnv
from robocode.environments.pr2_tamp_blocked_variable_count_env import (
    PR2BlockedVariableCountEnv,
)
from robocode.oracles.pr2blocked.approach import PR2BlockedOracleApproach

MAX_STEPS = 600
SEEDS = [0, 1]


def _run_episode(env: Any, seed: int, max_steps: int, count: int | None = None):
    """Run one episode; return (solved, num_steps, failure reason)."""
    approach = PR2BlockedOracleApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=seed,
        env=env,
    )
    options = None if count is None else {"object_count": count}
    state, info = env.reset(seed=seed, options=options)
    approach.reset(state, info)
    for step in range(max_steps):
        state, reward, terminated, truncated, info = env.step(approach.step())
        approach.update(state, float(reward), terminated or truncated, info)
        if terminated or truncated:
            return bool(terminated), step + 1, approach.failure
    return False, max_steps, approach.failure


@pytest.mark.parametrize("seed", SEEDS)
def test_oracle_solves_blocked_with_spares(seed: int) -> None:
    """The oracle gets a green block onto the plate when spares are available."""
    env = PR2BlockedEnv(num_spares=1)
    try:
        solved, steps, failure = _run_episode(env, seed, MAX_STEPS)
        assert solved, f"seed={seed}: not solved in {steps} steps ({failure})"
    finally:
        env.close()


@pytest.mark.parametrize("seed", SEEDS)
def test_oracle_solves_blocked_with_no_spares(seed: int) -> None:
    """The hardest instance: the blocker has to move, there is no alternative.

    With no spares the penned block is the only green one, so this is the case that
    proves the oracle can actually perform the blocking manipulation rather than
    routing around it.
    """
    env = PR2BlockedEnv(num_spares=0)
    try:
        solved, steps, failure = _run_episode(env, seed, MAX_STEPS)
        assert solved, f"seed={seed}: not solved in {steps} steps ({failure})"
    finally:
        env.close()


def test_oracle_moves_the_blocker_off_the_plate() -> None:
    """Relocating the blocker must not park it on the goal surface.

    A blocker left on the plate is in the way of the green block that has to land there,
    so the episode would be unwinnable from a state the oracle itself created.
    """
    env = PR2BlockedEnv(num_spares=0)
    try:
        solved, steps, failure = _run_episode(env, 0, MAX_STEPS)
        assert solved, f"not solved in {steps} steps ({failure})"
        # pylint: disable=import-outside-toplevel
        from robocode.environments import ss_pybullet as sp

        with env.client():
            assert not sp.is_placement(env.blocker, env.plate)
    finally:
        env.close()


def test_oracle_solves_within_variable_count_budget() -> None:
    """The oracle solves each configured count inside the runner's own budget.

    ``max_steps_for_count`` is what bounds an evaluation episode, so a solution that
    needs more steps than that is not a solution as far as the runner is concerned.
    """
    env = PR2BlockedVariableCountEnv(design_counts=[0, 1], eval_counts=[0, 1, 2])
    try:
        for count in env.eval_counts:
            budget = env.max_steps_for_count(count)
            solved, steps, failure = _run_episode(env, 0, budget, count=count)
            assert (
                solved
            ), f"count={count}: not solved in {steps} of {budget} steps ({failure})"
    finally:
        env.close()
