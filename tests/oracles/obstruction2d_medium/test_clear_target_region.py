"""Tests for PickTargetBlock behavior on Obstruction2D-o2."""

import kinder

from gymnasium.wrappers import RecordVideo
from robocode.oracles.obstruction2d_medium.behaviors import ClearTargetRegion
from tests.conftest import MAKE_VIDEOS


ENV_ID = "kinder/Obstruction2D-o2-v0"
MAX_STEPS = 500

def test_holding_target_after_pick():
    """After ClearTargetRegion, robot should hold the block."""
    kinder.register_all_environments()
    render_mode = "rgb_array" if MAKE_VIDEOS else None
    env = kinder.make(ENV_ID, render_mode=render_mode)

    # Setup: get initial obs and move obstructions out of the way.
    obs, _ = env.reset(seed=0)
    behavior = ClearTargetRegion()

    assert behavior.initializable(obs), "Precondition should be satisfied at the start."

    # Wrap with RecordVideo *after* setup so only the real episode is recorded.
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

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