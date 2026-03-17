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
    env: KinderGeom2DEnv,
    approach: Obstruction2DOracleApproach,
    seed: int,
) -> tuple[bool, int]:
    """Run a single episode.

    Return (solved, num_steps).
    """
    state, info = env.reset(seed=seed)
    approach.reset(state, info)

    for step in range(MAX_STEPS):
        action = approach.step()
        state, reward, terminated, truncated, info = env.step(action)
        approach.update(state, float(reward), terminated or truncated, info)
        if terminated or truncated:
            return bool(terminated), step + 1

    return False, MAX_STEPS


@pytest.fixture
def env():
    return KinderGeom2DEnv(ENV_ID)


# @pytest.mark.parametrize("seed", SEEDS)
def test_oracle_solves_episode(env: KinderGeom2DEnv):
    """The oracle approach should solve the environment for each seed."""
    seed = 636
    approach = Obstruction2DOracleApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
    )
    solved, steps = _run_episode(env, approach, seed)
    assert solved, f"seed={seed}: not solved in {steps} steps"
    print(f"seed={seed}: solved in {steps} steps")


def test_oracle_solve_rate():
    """The oracle should achieve 100% solve rate across seeds."""
    render_mode = "rgb_array" if MAKE_VIDEOS else None
    env = KinderGeom2DEnv(ENV_ID) if not MAKE_VIDEOS else None

    approach = Obstruction2DOracleApproach(
        action_space=KinderGeom2DEnv(ENV_ID).action_space,
        observation_space=KinderGeom2DEnv(ENV_ID).observation_space,
    )

    results = []
    for s in SEEDS:
        seed = int(s)
        ep_env = kinder.make(ENV_ID, render_mode=render_mode) if MAKE_VIDEOS else env
        if MAKE_VIDEOS:
            ep_env = RecordVideo(ep_env, f"unit_test_videos/approach_seed{seed}")
        solved, steps = _run_episode(ep_env, approach, seed)
        results.append({"seed": seed, "solved": solved, "steps": steps})
        print(f"seed={seed}: solved={solved} steps={steps}")
        if MAKE_VIDEOS:
            ep_env.close()

    solve_rate = np.mean([r["solved"] for r in results])
    mean_steps = np.mean([r["steps"] for r in results])
    print(f"\nSolve rate: {solve_rate:.0%}, Mean steps: {mean_steps:.0f}")
    assert solve_rate == 1.0, f"Solve rate {solve_rate:.0%} < 100%"
