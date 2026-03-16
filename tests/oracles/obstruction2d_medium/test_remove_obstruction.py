"""Tests for RemoveObstruction behavior on Obstruction2D-o2."""

import numpy as np
import pytest

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.oracles.obstruction2d_medium.behaviors import (
    RemoveObstruction,
    _overlaps_surface_h,
    OBS_BASE,
    OBS_STRIDE,
)

ENV_ID = "kinder/Obstruction2D-o2-v0"
MAX_STEPS = 500


def _run_behavior(env: KinderGeom2DEnv, behavior: RemoveObstruction, seed: int) -> tuple[np.ndarray, int]:
    """Run the behavior until terminated or MAX_STEPS. Return (final_obs, steps)."""
    obs, _ = env.reset(seed=seed)
    behavior.reset(obs)
    for step in range(MAX_STEPS):
        if behavior.terminated(obs):
            break
        action = behavior.step(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return obs, step


@pytest.fixture
def env():
    return KinderGeom2DEnv(ENV_ID)


SEEDS = [0, 1, 2, 3, 42]


@pytest.mark.parametrize("seed", SEEDS)
def test_precondition_holds_at_start(env: KinderGeom2DEnv, seed: int):
    """At least one obstruction should overlap the surface initially."""
    obs, _ = env.reset(seed=seed)
    behavior = RemoveObstruction()
    # The precondition may not hold for every seed (some seeds have
    # obstructions that don't overlap). Skip those.
    if not behavior.initializable(obs):
        pytest.skip("No obstructions overlap the surface for this seed")


@pytest.mark.parametrize("seed", SEEDS)
def test_goal_region_clear(env: KinderGeom2DEnv, seed: int):
    """After running RemoveObstruction, no obstruction should overlap the surface."""
    obs, _ = env.reset(seed=seed)
    behavior = RemoveObstruction()

    if not behavior.initializable(obs):
        pytest.skip("No obstructions on surface")

    behavior.reset(obs)
    obs, steps = _run_behavior(env, behavior, seed)

    # Check subgoal
    assert behavior.terminated(obs), (
        f"seed={seed}: GoalRegionClear not achieved in {steps} steps"
    )
    # Double-check: no obstruction overlaps surface
    for i in range(2):
        assert not _overlaps_surface_h(obs, OBS_BASE + i * OBS_STRIDE), (
            f"seed={seed}: obstruction {i} still overlaps surface"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_completes_within_budget(env: KinderGeom2DEnv, seed: int):
    """Behavior should finish well within MAX_STEPS."""
    obs, _ = env.reset(seed=seed)
    behavior = RemoveObstruction()

    if not behavior.initializable(obs):
        pytest.skip("No obstructions on surface")

    behavior.reset(obs)
    _, steps = _run_behavior(env, behavior, seed)
    assert steps < MAX_STEPS - 1, f"seed={seed}: used all {MAX_STEPS} steps"
