"""Regression tests for the ClutteredStorage2D oracle approach."""

from __future__ import annotations

import pytest

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.oracles.clutteredstorage2d_medium.approach import (
    ClutteredStorage2DOracleApproach,
)
from robocode.oracles.clutteredstorage2d_medium.obs_helpers import (
    all_blocks_inside_shelf,
    inside_blocks,
    outside_blocks,
)

ENV_ID = "kinder/ClutteredStorage2D-b3-v0"
MAX_STEPS = 500
SOLVE_SEEDS = [0, 1, 2, 3]


def _run_episode(
    episode_env: KinderGeom2DEnv,
    approach: ClutteredStorage2DOracleApproach,
    seed: int,
) -> tuple[bool, int, list[str], list[str]]:
    """Run one episode and return solve status plus final block partition."""
    state, info = episode_env.reset(seed=seed)
    approach.reset(state, info)

    for step in range(MAX_STEPS):
        action = approach.step()
        state, reward, terminated, truncated, info = episode_env.step(action)
        approach.update(state, float(reward), terminated or truncated, info)
        if terminated or truncated:
            return (
                bool(terminated),
                step + 1,
                inside_blocks(state),
                outside_blocks(state),
            )

    return (
        all_blocks_inside_shelf(state),
        MAX_STEPS,
        inside_blocks(state),
        outside_blocks(state),
    )


@pytest.fixture(name="clutteredstorage_env")
def _clutteredstorage_env() -> KinderGeom2DEnv:
    """Create a KinderGeom2DEnv for the ClutteredStorage2D-b3 environment."""
    return KinderGeom2DEnv(ENV_ID)


@pytest.mark.parametrize("seed", SOLVE_SEEDS)
def test_oracle_solves_regression_seeds(
    clutteredstorage_env: KinderGeom2DEnv,
    seed: int,
) -> None:
    """The oracle should solve a fixed set of regression seeds."""
    approach = ClutteredStorage2DOracleApproach(
        action_space=clutteredstorage_env.action_space,
        observation_space=clutteredstorage_env.observation_space,
    )
    solved, steps, inside, outside = _run_episode(clutteredstorage_env, approach, seed)

    assert solved, (
        f"seed={seed}: not solved in {steps} steps; "
        f"inside={inside}, outside={outside}"
    )
    assert len(outside) == 0, f"seed={seed}: expected no outside blocks, got {outside}"
    assert len(inside) == 3, f"seed={seed}: expected 3 inside blocks, got {inside}"


def test_oracle_solves_all_regression_seeds_in_batch(
    clutteredstorage_env: KinderGeom2DEnv,
) -> None:
    """Fail loudly if any regression seed stops solving."""
    failures: list[str] = []

    for seed in SOLVE_SEEDS:
        approach = ClutteredStorage2DOracleApproach(
            action_space=clutteredstorage_env.action_space,
            observation_space=clutteredstorage_env.observation_space,
        )
        solved, steps, inside, outside = _run_episode(
            clutteredstorage_env, approach, seed
        )
        if not solved:
            failures.append(
                f"seed={seed} steps={steps} inside={inside} outside={outside}"
            )

    assert not failures, "Regression seeds failed:\n" + "\n".join(failures)
