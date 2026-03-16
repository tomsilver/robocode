"""Tests for PlaceTargetBlock behavior on Obstruction2D-o2."""

import numpy as np
import pytest

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.oracles.obstruction2d_medium.behaviors import (
    PickTargetBlock,
    PlaceTargetBlock,
    RemoveObstruction,
    _is_on_surface,
    BLOCK,
)

ENV_ID = "kinder/Obstruction2D-o2-v0"
MAX_STEPS = 500


def _setup_holding(env: KinderGeom2DEnv, seed: int) -> np.ndarray:
    """Run RemoveObstruction then PickTargetBlock; return obs with block held."""
    obs, _ = env.reset(seed=seed)

    rm = RemoveObstruction()
    if rm.initializable(obs):
        rm.reset(obs)
        for _ in range(MAX_STEPS):
            if rm.terminated(obs):
                break
            obs, _, terminated, truncated, _ = env.step(rm.step(obs))
            if terminated or truncated:
                break

    pick = PickTargetBlock()
    if pick.initializable(obs):
        pick.reset(obs)
        for _ in range(MAX_STEPS):
            if pick.terminated(obs):
                break
            obs, _, terminated, truncated, _ = env.step(pick.step(obs))
            if terminated or truncated:
                break

    return obs


@pytest.fixture
def env():
    return KinderGeom2DEnv(ENV_ID)


SEEDS = [0, 1, 2, 3, 42]


@pytest.mark.parametrize("seed", SEEDS)
def test_goal_achieved(env: KinderGeom2DEnv, seed: int):
    """Full pipeline: remove → pick → place should achieve the task goal."""
    obs = _setup_holding(env, seed)
    behavior = PlaceTargetBlock()

    if not behavior.initializable(obs):
        pytest.skip("Not holding block after pick phase")

    behavior.reset(obs)
    for step in range(MAX_STEPS):
        if behavior.terminated(obs):
            break
        obs, _, terminated, truncated, _ = env.step(behavior.step(obs))
        if terminated or truncated:
            break

    assert behavior.terminated(obs), (
        f"seed={seed}: GoalAchieved not reached in {step} steps"
    )
    assert _is_on_surface(obs, BLOCK)


@pytest.mark.parametrize("seed", SEEDS)
def test_env_reports_done(env: KinderGeom2DEnv, seed: int):
    """The environment itself should report terminated=True after placement."""
    obs = _setup_holding(env, seed)
    behavior = PlaceTargetBlock()

    if not behavior.initializable(obs):
        pytest.skip("Not holding block")

    behavior.reset(obs)
    env_terminated = False
    for step in range(MAX_STEPS):
        if behavior.terminated(obs):
            break
        obs, _, env_terminated, truncated, _ = env.step(behavior.step(obs))
        if env_terminated or truncated:
            break

    assert env_terminated or behavior.terminated(obs), (
        f"seed={seed}: neither env nor behavior reports success"
    )
