"""Tests for TouchAllButtons behavior on StickButton2D-b3."""

import kinder
from gymnasium.wrappers import RecordVideo

from robocode.oracles.stickbutton2d_medium.behaviors import (
    GraspStickBottom,
    TouchAllButtons,
)
from tests.conftest import MAKE_VIDEOS

ENV_ID = "kinder/StickButton2D-b3-v0"
MAX_STEPS = 500


def test_grasp_then_touch():
    """After GraspStickBottom then TouchAllButtons, all buttons should be pressed."""
    kinder.register_all_environments()
    render_mode = "rgb_array" if MAKE_VIDEOS else None
    env = kinder.make(ENV_ID, render_mode=render_mode)
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    obs, _ = env.reset(seed=0)
    grasp = GraspStickBottom()
    touch = TouchAllButtons()

    # Phase 1: grasp stick at bottom
    assert grasp.initializable(obs), "Grasp precondition should hold at start."
    grasp.reset(obs)
    for s in range(MAX_STEPS):
        action = grasp.step(obs)
        obs, _, _, _, _ = env.step(action)
        if grasp.terminated(obs):
            print(f"Stick grasped at bottom in {s + 1} steps.")
            break

    assert grasp.terminated(obs), f"Stick not grasped within {MAX_STEPS} steps."

    # Phase 2: touch all buttons
    assert touch.initializable(obs), "Touch precondition should hold after grasp."
    assert not touch.terminated(obs), "Buttons should not all be pressed yet."
    touch.reset(obs)
    for s in range(MAX_STEPS):
        action = touch.step(obs)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            print(f"All buttons pressed in {s + 1} steps.")
            break

    assert touch.terminated(obs), f"Not all buttons pressed within {MAX_STEPS} steps."
    env.close()
