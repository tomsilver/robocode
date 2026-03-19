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

    RePositionStick should move it toward the centre so that
    has_space_stick_bottom becomes True.  Because the behaviour grabs at
    the stick bottom, it may already satisfy stick_bottom_grasped, in
    which case GraspStickBottom can be skipped.  If not, GraspStickBottom
    should succeed afterwards.
    """
    kinder.register_all_environments()
    render_mode = "rgb_array" if MAKE_VIDEOS else None
    env = kinder.make(ENV_ID, render_mode=render_mode)
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    obs, _ = env.reset(seed=0)
    reposition = RePositionStick()

    # Phase 1: reposition – stick starts too close to the left wall
    assert reposition.initializable(obs), (
        "RePositionStick precondition should hold (stick near left wall)."
    )
    assert no_space_stick_bottom(obs)

    reposition.reset(obs)
    for s in range(MAX_STEPS):
        action = reposition.step(obs)
        obs, _, _, _, _ = env.step(action)
        if reposition.terminated(obs):
            print(f"RePositionStick done in {s + 1} steps.")
            break

    assert reposition.terminated(obs), (
        f"RePositionStick not done within {MAX_STEPS} steps."
    )
    assert has_space_stick_bottom(obs), (
        "After reposition the stick should have clearance."
    )

    # Phase 2: the reposition grab may already satisfy stick_bottom_grasped
    # (the behaviour grabs at the stick bottom).  If so, GraspStickBottom
    # can be skipped — exactly what the approach does.
    if stick_bottom_grasped(obs):
        print("Stick already grasped at bottom after reposition — "
              "GraspStickBottom can be skipped.")
    else:
        grasp = GraspStickBottom()
        assert grasp.initializable(obs), (
            "GraspStickBottom precondition should hold after reposition."
        )
        grasp.reset(obs)
        for s in range(MAX_STEPS):
            action = grasp.step(obs)
            obs, _, _, _, _ = env.step(action)
            if grasp.terminated(obs):
                print(f"GraspStickBottom done in {s + 1} steps.")
                break

        assert grasp.terminated(obs), (
            f"GraspStickBottom not done within {MAX_STEPS} steps."
        )

    env.close()
