"""Tests for the PR2 packed oracle approach.

These are the solvability check for the environment: if the oracle cannot reach the
goal within the step budget, the task is not solvable as specified and no result from
a synthesized approach would mean anything.
"""

from typing import Any

import pytest

from robocode.environments.pr2_tamp_env import PR2PackedEnv
from robocode.environments.pr2_tamp_variable_count_env import (
    PR2PackedVariableCountEnv,
)
from robocode.oracles.pr2packed.approach import PR2PackedOracleApproach

MAX_STEPS = 600
SEEDS = [0, 1]


def _run_episode(env: Any, seed: int, max_steps: int, count: int | None = None):
    """Run one episode; return (solved, num_steps, failure reason)."""
    approach = PR2PackedOracleApproach(
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
def test_oracle_solves_packed(seed: int) -> None:
    """The oracle places every block on the plate."""
    env = PR2PackedEnv(num_blocks=3)
    try:
        solved, steps, failure = _run_episode(env, seed, MAX_STEPS)
        assert solved, f"seed={seed}: not solved in {steps} steps ({failure})"
    finally:
        env.close()


def test_oracle_solves_within_variable_count_budget() -> None:
    """The oracle solves a held-out count inside the budget the runner would give it.

    ``max_steps_for_count`` is what bounds an evaluation episode, so a solution that
    only fits in an unbounded rollout would not actually score.
    """
    env = PR2PackedVariableCountEnv(
        design_counts=[1, 2, 3], eval_counts=[1, 2, 3, 4, 5]
    )
    try:
        count = 4
        budget = env.max_steps_for_count(count)
        solved, steps, failure = _run_episode(env, 0, budget, count=count)
        assert (
            solved
        ), f"count={count}: not solved in {steps}/{budget} steps ({failure})"
    finally:
        env.close()
