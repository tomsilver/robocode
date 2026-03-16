"""Tests for PickTargetBlock behavior on Obstruction2D-o2."""

import numpy as np
import pytest

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.oracles.obstruction2d_medium.behaviors import (
    PickTargetBlock,
    RemoveObstruction,
    _holding_block,
)

ENV_ID = "kinder/Obstruction2D-o2-v0"
MAX_STEPS = 500


def _clear_surface(env: KinderGeom2DEnv, seed: int) -> np.ndarray:
    """Run RemoveObstruction first and return the resulting observation."""
    obs, _ = env.reset(seed=seed)
    rm = RemoveObstruction()
    if rm.initializable(obs):
        rm.reset(obs)
        for _ in range(MAX_STEPS):
            if rm.terminated(obs):
                break
            action = rm.step(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    return obs


def _run_behavior(env: KinderGeom2DEnv, behavior: PickTargetBlock, obs: np.ndarray) -> tuple[np.ndarray, int]:
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
def test_holding_target_after_pick(env: KinderGeom2DEnv, seed: int):
    """After RemoveObstruction + PickTargetBlock, robot should hold the block."""
    obs = _clear_surface(env, seed)
    behavior = PickTargetBlock()

    if not behavior.initializable(obs):
        pytest.skip("Precondition not met after surface clear")

    obs, steps = _run_behavior(env, behavior, obs)
    assert behavior.terminated(obs), (
        f"seed={seed}: HoldingTarget not achieved in {steps} steps"
    )
    assert _holding_block(obs)


@pytest.mark.parametrize("seed", SEEDS)
def test_completes_within_budget(env: KinderGeom2DEnv, seed: int):
    """Behavior should finish well within MAX_STEPS."""
    obs = _clear_surface(env, seed)
    behavior = PickTargetBlock()

    if not behavior.initializable(obs):
        pytest.skip("Precondition not met")

    obs, steps = _run_behavior(env, behavior, obs)
    assert steps < MAX_STEPS - 1, f"seed={seed}: used all {MAX_STEPS} steps"
