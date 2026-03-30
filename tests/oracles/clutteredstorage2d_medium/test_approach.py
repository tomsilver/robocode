"""Tests for the ClutteredStorage2D oracle approach."""

import kinder
import pytest
from gymnasium.wrappers import RecordVideo

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.oracles.clutteredstorage2d_medium.approach import (
    ClutteredStorage2DOracleApproach,
)
from robocode.oracles.clutteredstorage2d_medium.obs_helpers import (
    all_blocks_inside_shelf,
    inside_blocks,
    outside_blocks,
)
from tests.conftest import MAKE_VIDEOS

ENV_ID = "kinder/ClutteredStorage2D-b3-v0"
MAX_STEPS = 500
SOLVE_SEEDS = [0, 1]


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
def _clutteredstorage_env():
    """Create a KinderGeom2DEnv for the ClutteredStorage2D-b3 environment."""
    return KinderGeom2DEnv(ENV_ID)


@pytest.mark.parametrize("seed", SOLVE_SEEDS)
def test_oracle_solves_known_seeds(
    clutteredstorage_env: KinderGeom2DEnv,
    seed: int,
):
    """The oracle approach should solve known regression seeds."""
    approach = ClutteredStorage2DOracleApproach(
        action_space=clutteredstorage_env.action_space,
        observation_space=clutteredstorage_env.observation_space,
    )
    solved, steps, inside, outside = _run_episode(clutteredstorage_env, approach, seed)
    assert solved, (
        f"seed={seed}: not solved in {steps} steps; "
        f"inside={inside}, outside={outside}"
    )
    assert len(outside) == 0
    assert len(inside) == 3
    print(f"seed={seed}: solved in {steps} steps")


def test_oracle_episode_with_optional_video():
    """Run a full episode in the same style as the other oracle approach tests."""
    kinder.register_all_environments()
    render_mode = "rgb_array" if MAKE_VIDEOS else None
    base_env = kinder.make(ENV_ID, render_mode=render_mode)
    env = RecordVideo(base_env, "unit_test_videos") if MAKE_VIDEOS else base_env
    try:
        wrapped_env = env
        approach = ClutteredStorage2DOracleApproach(
            action_space=wrapped_env.action_space,
            observation_space=wrapped_env.observation_space,
        )
        solved, steps, inside, outside = _run_episode(wrapped_env, approach, seed=0)
        assert solved, (
            f"seed=0: not solved in {steps} steps; "
            f"inside={inside}, outside={outside}"
        )
    finally:
        env.close()
