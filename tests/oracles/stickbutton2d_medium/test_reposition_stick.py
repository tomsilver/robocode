"""Tests for RePositionStick behavior on StickButton2D-b3."""

import kinder
from gymnasium.wrappers import RecordVideo

from robocode.oracles.stickbutton2d_medium.behaviors import (
    GraspStickBottom,
    RePositionStick,
)
from robocode.oracles.stickbutton2d_medium.obs_helpers import (
    has_space_stick_bottom,
    no_space_stick_bottom,
    stick_bottom_grasped,
)
from tests.conftest import MAKE_VIDEOS

ENV_ID = "kinder/StickButton2D-b3-v0"
MAX_STEPS = 500


def test_reposition_then_grasp():
    """With seed=0 the stick is near the left wall.

    RePositionStick grabs the stick's closest long side horizontally and slides it
    toward the world centre.  Because the grasp is horizontal (not bottom-up),
    stick_bottom_grasped remains False, so GraspStickBottom is initializable and re-
    grabs at the bottom.
    """
    kinder.register_all_environments()
    render_mode = "rgb_array" if MAKE_VIDEOS else None
    env = kinder.make(ENV_ID, render_mode=render_mode)
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    obs, _ = env.reset(seed=0)
    reposition = RePositionStick()

    # Phase 1: reposition – stick starts too close to the left wall
    assert reposition.initializable(
        obs
    ), "RePositionStick precondition should hold (stick near left wall)."
    assert no_space_stick_bottom(obs)

    reposition.reset(obs)
    for s in range(MAX_STEPS):
        action = reposition.step(obs)
        obs, _, _, _, _ = env.step(action)
        if reposition.terminated(obs):
            print(f"RePositionStick done in {s + 1} steps.")
            break

    assert reposition.terminated(
        obs
    ), f"RePositionStick not done within {MAX_STEPS} steps."
    assert has_space_stick_bottom(
        obs
    ), "After reposition the stick should have clearance."
    # The side grasp should NOT satisfy stick_bottom_grasped
    assert not stick_bottom_grasped(
        obs
    ), "A horizontal side grasp should not count as a bottom grasp."

    # Phase 2: GraspStickBottom should now be initializable
    grasp = GraspStickBottom()
    assert grasp.initializable(
        obs
    ), "GraspStickBottom precondition should hold after reposition."

    grasp.reset(obs)
    for s in range(MAX_STEPS):
        action = grasp.step(obs)
        obs, _, _, _, _ = env.step(action)
        if grasp.terminated(obs):
            print(f"GraspStickBottom done in {s + 1} steps.")
            break

    assert grasp.terminated(obs), f"GraspStickBottom not done within {MAX_STEPS} steps."
    assert stick_bottom_grasped(
        obs
    ), "After GraspStickBottom the stick should be held at its bottom."
    env.close()
