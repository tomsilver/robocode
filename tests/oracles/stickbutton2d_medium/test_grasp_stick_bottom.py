"""Tests for GraspStickBottom behavior on StickButton2D-b3."""

import kinder
from gymnasium.wrappers import RecordVideo

from robocode.oracles.stickbutton2d_medium.behaviors import GraspStickBottom
from tests.conftest import MAKE_VIDEOS

ENV_ID = "kinder/StickButton2D-b3-v0"
MAX_STEPS = 500


def test_stick_grasped_at_bottom():
    """After GraspStickBottom, the stick should be held near its bottom."""
    kinder.register_all_environments()
    render_mode = "rgb_array" if MAKE_VIDEOS else None
    env = kinder.make(ENV_ID, render_mode=render_mode)
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    obs, _ = env.reset(seed=1)
    behavior = GraspStickBottom()

    assert behavior.initializable(obs), "Precondition should be satisfied at the start."
    assert not behavior.terminated(obs), "Subgoal should not be satisfied at the start."

    behavior.reset(obs)
    for s in range(MAX_STEPS):
        action = behavior.step(obs)
        obs, _, _, _, _ = env.step(action)
        if behavior.terminated(obs):
            print(f"Subgoal achieved in {s + 1} steps.")
            break

    assert behavior.terminated(obs), f"Subgoal not achieved within {MAX_STEPS} steps."
    env.close()
