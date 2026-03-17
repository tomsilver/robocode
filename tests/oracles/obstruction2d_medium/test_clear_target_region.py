"""Tests for PickPlaceTargetBlock behavior on Obstruction2D-o2."""

import kinder

from gymnasium.wrappers import RecordVideo
from robocode.oracles.obstruction2d_medium.behaviors import ClearTargetRegion, PickPlaceTargetBlock
from tests.conftest import MAKE_VIDEOS


ENV_ID = "kinder/Obstruction2D-o2-v0"
MAX_STEPS = 500

def test_region_cleared_after_clear_target_region():
    """After ClearTargetRegion, robot should hold the block."""
    kinder.register_all_environments()
    render_mode = "rgb_array" if MAKE_VIDEOS else None
    env = kinder.make(ENV_ID, render_mode=render_mode)
    # Wrap with RecordVideo *after* setup so only the real episode is recorded.
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    # Setup: get initial obs and move obstructions out of the way.
    obs, _ = env.reset(seed=0)
    behavior = ClearTargetRegion()

    assert behavior.initializable(obs), "Precondition should be satisfied at the start."

    assert not behavior.terminated(obs), "Subgoal should not be satisfied at the start."

    behavior.reset(obs)
    for s in range(MAX_STEPS):
        action = behavior.step(obs)
        obs, _, _, _, _ = env.step(action)
        if behavior.terminated(obs):
            print(f"Subgoal achieved in {s+1} steps.")
            break

    assert behavior.terminated(obs), f"Subgoal not achieved within {MAX_STEPS} steps."
    env.close()


def test_clear_then_pick():
    """After ClearTargetRegion, robot should hold the block."""
    kinder.register_all_environments()
    render_mode = "rgb_array" if MAKE_VIDEOS else None
    env = kinder.make(ENV_ID, render_mode=render_mode)
    # Wrap with RecordVideo *after* setup so only the real episode is recorded.
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")
        
    # Setup: get initial obs and move obstructions out of the way.
    obs, _ = env.reset(seed=0)
    behavior_clear = ClearTargetRegion()
    behavior_pick = PickPlaceTargetBlock()

    assert behavior_clear.initializable(obs), "Precondition should be satisfied at the start."

    assert not behavior_clear.terminated(obs), "Subgoal should not be satisfied at the start."

    behavior_clear.reset(obs)
    for s in range(MAX_STEPS):
        action = behavior_clear.step(obs)
        obs, _, _, _, _ = env.step(action)
        if behavior_clear.terminated(obs):
            print(f"Subgoal achieved in {s+1} steps.")
            break

    assert behavior_clear.terminated(obs), f"Subgoal not achieved within {MAX_STEPS} steps."

    assert behavior_pick.initializable(obs), "Precondition for pick should be satisfied after clear."
    assert not behavior_pick.terminated(obs), "Subgoal for pick should not be satisfied at the start."
    behavior_pick.reset(obs)
    for s in range(MAX_STEPS):
        action = behavior_pick.step(obs)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            print(f"Pick subgoal achieved in {s+1} steps.")
            break

    assert behavior_pick.terminated(obs), f"Pick subgoal not achieved within {MAX_STEPS} steps."
    env.close()