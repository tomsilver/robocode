"""Tests for TouchAllButtons behavior on StickButton2D-b3."""

import kinder
from gymnasium.wrappers import RecordVideo

from robocode.oracles.stickbutton2d_medium.behaviors import (
    GraspStickBottom,
    RePositionStick,
    TouchAllButtons,
)
from robocode.oracles.stickbutton2d_medium.obs_helpers import (
    extract_circle,
    extract_rect,
    extract_robot,
    is_button_pressed,
    no_space_stick_bottom,
    stick_bottom_grasped,
)
from tests.conftest import MAKE_VIDEOS

ENV_ID = "kinder/StickButton2D-b3-v0"
MAX_STEPS = 500


def test_reposition_grasp_then_touch():
    """Seed=0: RePositionStick → GraspStickBottom → TouchAllButtons."""
    kinder.register_all_environments()
    render_mode = "rgb_array" if MAKE_VIDEOS else None
    env = kinder.make(ENV_ID, render_mode=render_mode)
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    obs, _ = env.reset(seed=0)

    # Print initial layout
    r = extract_robot(obs)
    s = extract_rect(obs, "stick")
    print(f"Initial: robot=({r.x:.2f},{r.y:.2f}), stick=({s.x:.3f},{s.y:.3f})")
    for i in range(3):
        b = extract_circle(obs, f"button{i}")
        print(f"  button{i}: ({b.x:.2f}, {b.y:.2f})")

    # Phase 1: RePositionStick (stick near left wall)
    assert no_space_stick_bottom(obs)
    reposition = RePositionStick()
    reposition.reset(obs)
    for step in range(MAX_STEPS):
        action = reposition.step(obs)
        obs, _, _, _, _ = env.step(action)
        if reposition.terminated(obs):
            print(f"RePositionStick done in {step + 1} steps.")
            break
    assert reposition.terminated(obs), "RePositionStick failed."

    # Phase 2: GraspStickBottom
    if not stick_bottom_grasped(obs):
        grasp = GraspStickBottom()
        assert grasp.initializable(obs)
        grasp.reset(obs)
        for step in range(MAX_STEPS):
            action = grasp.step(obs)
            obs, _, _, _, _ = env.step(action)
            if grasp.terminated(obs):
                print(f"GraspStickBottom done in {step + 1} steps.")
                break
        assert grasp.terminated(obs), "GraspStickBottom failed."
    else:
        print("stick_bottom_grasped already True, skipping GraspStickBottom.")

    assert stick_bottom_grasped(obs)

    # Debug: print state before TouchAllButtons
    r = extract_robot(obs)
    s = extract_rect(obs, "stick")
    print(
        f"Before touch: robot=({r.x:.2f},{r.y:.2f},th={r.theta:.2f},"
        f"arm={r.arm_joint:.2f},vac={r.vacuum:.0f}), "
        f"stick=({s.x:.3f},{s.y:.3f}), stick.top={s.top:.3f}"
    )

    # Phase 3: TouchAllButtons
    touch = TouchAllButtons()
    assert touch.initializable(obs), "Touch precondition should hold."
    assert not touch.terminated(obs)
    touch.reset(obs)
    for step in range(MAX_STEPS):
        action = touch.step(obs)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            print(f"All buttons pressed in {step + 1} steps.")
            break
        # Debug: every 50 steps print state
        if (step + 1) % 50 == 0:
            r2 = extract_robot(obs)
            s2 = extract_rect(obs, "stick")
            pressed = [is_button_pressed(obs, f"button{i}") for i in range(3)]
            print(
                f"  step {step+1}: robot=({r2.x:.2f},{r2.y:.2f}), "
                f"stick=({s2.x:.3f},{s2.y:.3f}), pressed={pressed}"
            )

    assert touch.terminated(obs), f"Not all buttons pressed within {MAX_STEPS} steps."
    env.close()
