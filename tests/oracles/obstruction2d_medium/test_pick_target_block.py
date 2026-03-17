"""Tests for PickPlaceTargetBlock behavior on Obstruction2D-o2."""

import kinder
from gymnasium.wrappers import RecordVideo

from robocode.oracles.obstruction2d_medium.behaviors import PickPlaceTargetBlock
from tests.conftest import MAKE_VIDEOS

ENV_ID = "kinder/Obstruction2D-o2-v0"
MAX_STEPS = 500


def test_block_on_surface_after_pick_place():
    """After PickPlaceTargetBlock, block should be on the target surface."""
    kinder.register_all_environments()
    render_mode = "rgb_array" if MAKE_VIDEOS else None
    env = kinder.make(ENV_ID, render_mode=render_mode)

    # Setup: get initial obs and move obstructions out of the way.
    obs_init, _ = env.reset(seed=0)
    behavior = PickPlaceTargetBlock()

    # Move obstructions away so the target region is clear.
    obs_init[29] += 0.5
    obs_init[39] += 0.5
    assert behavior.initializable(
        obs_init
    ), "Precondition should be satisfied when region is clear."

    # Wrap with RecordVideo *after* setup so only the real episode is recorded.
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    obs, _ = env.reset(options={"init_state": obs_init})
    assert not behavior.terminated(obs), "Subgoal should not be satisfied at the start."

    behavior.reset(obs)
    for s in range(MAX_STEPS):
        action = behavior.step(obs)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            print(f"Subgoal achieved in {s+1} steps.")
            break

    assert behavior.terminated(obs), f"Subgoal not achieved within {MAX_STEPS} steps."
    env.close()
