"""Tests for StoreRemainingBlocks behavior on ClutteredStorage2D-b3."""

import kinder
from gymnasium.wrappers import RecordVideo

from robocode.oracles.clutteredstorage2d_medium.behaviors import StoreRemainingBlocks
from robocode.oracles.clutteredstorage2d_medium.obs_helpers import (
    all_blocks_inside_shelf,
    outside_blocks,
)
from tests.conftest import MAKE_VIDEOS

ENV_ID = "kinder/ClutteredStorage2D-b3-v0"
MAX_STEPS = 500


def test_store_remaining_blocks_solves_seed0():
    """StoreRemainingBlocks should solve the seed-0 rollout by itself."""
    kinder.register_all_environments()
    render_mode = "rgb_array" if MAKE_VIDEOS else None
    env = kinder.make(ENV_ID, render_mode=render_mode)
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    try:
        obs, _ = env.reset(seed=0)
        behavior = StoreRemainingBlocks()
        assert behavior.initializable(obs)
        assert not behavior.terminated(obs)

        behavior.reset(obs)
        for _ in range(MAX_STEPS):
            action = behavior.step(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        assert all_blocks_inside_shelf(obs), (
            f"Subgoal not achieved within {MAX_STEPS} steps; "
            f"outside={outside_blocks(obs)}"
        )
        assert behavior.terminated(obs)
    finally:
        env.close()
