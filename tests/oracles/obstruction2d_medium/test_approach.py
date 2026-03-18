"""Tests for the Obstruction2D oracle approach."""

import kinder
import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.oracles.obstruction2d_medium.approach import (
    Obstruction2DOracleApproach,
)
from tests.conftest import MAKE_VIDEOS

ENV_ID = "kinder/Obstruction2D-o2-v0"
MAX_STEPS = 500
SEEDS = list(np.random.choice(1000, size=5, replace=False))


def _run_episode(
    episode_env: KinderGeom2DEnv,
    approach: Obstruction2DOracleApproach,
    seed: int,
) -> tuple[bool, int]:
    """Run a single episode.

    Return (solved, num_steps).
    """
    state, info = episode_env.reset(seed=seed)
    approach.reset(state, info)

    for step in range(MAX_STEPS):
        action = approach.step()
        state, reward, terminated, truncated, info = episode_env.step(action)
        approach.update(state, float(reward), terminated or truncated, info)
        if terminated or truncated:
            return bool(terminated), step + 1

    return False, MAX_STEPS


@pytest.fixture(name="obstruction_env")
def _obstruction_env():
    """Create a KinderGeom2DEnv for the Obstruction2D-o2 environment."""
    return KinderGeom2DEnv(ENV_ID)


def test_oracle_solves_episode(obstruction_env: KinderGeom2DEnv):
    """The oracle approach should solve the environment for each seed."""
    seed = 636
    approach = Obstruction2DOracleApproach(
        action_space=obstruction_env.action_space,
        observation_space=obstruction_env.observation_space,
    )
    solved, steps = _run_episode(obstruction_env, approach, seed)
    assert solved, f"seed={seed}: not solved in {steps} steps"
    print(f"seed={seed}: solved in {steps} steps")


def test_oracle_solve_rate():
    """The oracle should achieve 100% solve rate across seeds."""
    render_mode = "rgb_array" if MAKE_VIDEOS else None
    shared_env = KinderGeom2DEnv(ENV_ID) if not MAKE_VIDEOS else None

    approach = Obstruction2DOracleApproach(
        action_space=KinderGeom2DEnv(ENV_ID).action_space,
        observation_space=KinderGeom2DEnv(ENV_ID).observation_space,
    )

    results = []
    for s in SEEDS:
        seed = int(s)
        if MAKE_VIDEOS:
            ep_env = RecordVideo(
                kinder.make(ENV_ID, render_mode=render_mode),
                f"unit_test_videos/approach_seed{seed}",
            )
        else:
            ep_env = shared_env
        solved, steps = _run_episode(ep_env, approach, seed)
        results.append({"seed": seed, "solved": solved, "steps": steps})
        print(f"seed={seed}: solved={solved} steps={steps}")
        if MAKE_VIDEOS:
            ep_env.close()

    solve_rate = np.mean([r["solved"] for r in results])
    mean_steps = np.mean([r["steps"] for r in results])
    print(f"\nSolve rate: {solve_rate:.0%}, Mean steps: {mean_steps:.0f}")
    assert solve_rate == 1.0, f"Solve rate {solve_rate:.0%} < 100%"
