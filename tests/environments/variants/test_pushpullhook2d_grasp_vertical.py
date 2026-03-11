"""Tests for pushpullhook2d_grasp_vertical.py."""

import kinder
import numpy as np
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo
from kinder.envs.geom2d.pushpullhook2d_grasp_vertical import (
    ObjectCentricPushPullHook2DGraspVerticalEnv,
)

from tests.conftest import MAKE_VIDEOS


def test_object_centric_pushpullhook2d_grasp_vertical_env():
    """Tests for ObjectCentricPushPullHook2DGraspVerticalEnv()."""
    env = ObjectCentricPushPullHook2DGraspVerticalEnv()
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")
    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    env.close()


def test_pushpullhook2d_grasp_vertical_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    kinder.register_all_environments()
    env = kinder.make("kinder/PushPullHook2D-GraspVertical-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


def test_pushpullhook2d_grasp_vertical_action_space():
    """Tests that actions are vectors with fixed dimensionality."""
    kinder.register_all_environments()
    env = kinder.make("kinder/PushPullHook2D-GraspVertical-v0")
    assert isinstance(env.action_space, Box)
    for _ in range(5):
        action = env.action_space.sample()
        assert env.action_space.contains(action)


def test_hook_always_vertical():
    """Verify hook theta = pi/2 across multiple seeds."""
    env = ObjectCentricPushPullHook2DGraspVerticalEnv()
    for seed in range(20):
        state, _ = env.reset(seed=seed)
        hook = next(o for o in state if o.name == "hook")
        assert np.isclose(
            state.get(hook, "theta"), np.pi / 2, atol=1e-6
        ), f"seed={seed}: hook theta={state.get(hook, 'theta')}"
    env.close()


def _solve_grasp(env, state, max_steps=500, step_env=None):
    """Scripted solver: navigate beside the hook bar, face it, extend arm,
    suction.

    At theta=pi/2 the L-shape long bar is vertical (width=0.05). The robot
    approaches from the side, positions so the gripper clears the bar, then
    extends the arm so only the suction zone (ZOrder.NONE) overlaps the bar.

    Three phases:
      1. Navigate to standoff position and face the bar (arm retracted).
      2. Extend arm fully (no movement).
      3. Turn on vacuum.
    """
    if step_env is None:
        step_env = env
    obj_map = {o.name: o for o in state}
    robot = obj_map["robot"]
    hook = obj_map["hook"]

    phase = "navigate"

    for step_i in range(max_steps):
        rx = state.get(robot, "x")
        ry = state.get(robot, "y")
        rt = state.get(robot, "theta")
        hx = state.get(hook, "x")
        hy = state.get(hook, "y")
        hw = state.get(hook, "width")
        hl1 = state.get(hook, "length_side1")
        arm_length = state.get(robot, "arm_length")
        gripper_w = state.get(robot, "gripper_width")
        base_r = state.get(robot, "base_radius")
        arm_joint = state.get(robot, "arm_joint")

        bar_right = hx + hw  # right edge of the vertical bar

        # Standoff: gripper (half-width 0.005) must clear bar_right,
        # suction zone (extends 0.02 past gripper edge) must overlap bar.
        # target_x = bar_right + arm_length + gripper_w/2 + margin
        standoff_from_bar_right = arm_length + gripper_w / 2 + 0.005

        # Approach from whichever side robot is on.
        bar_cx = hx + hw / 2
        if rx > bar_cx:
            target_x = bar_right + standoff_from_bar_right
            face_theta = np.pi  # face left
        else:
            target_x = hx - standoff_from_bar_right
            face_theta = 0.0  # face right

        # Target a y inside the bar span, below the table.
        bar_bottom_y = hy - hl1
        table_y = env.config.table_pose.y
        target_y = min(bar_bottom_y + hl1 * 0.5, table_y - base_r * 2)
        target_y = max(target_y, bar_bottom_y + base_r * 2)

        dx_t = target_x - rx
        dy_t = target_y - ry
        dist_to_target = np.sqrt(dx_t**2 + dy_t**2)

        # Angle control: face the bar.
        angle_err = (face_theta - rt + np.pi) % (2 * np.pi) - np.pi
        dtheta = np.clip(angle_err, -env.config.max_dtheta, env.config.max_dtheta)

        if phase == "navigate":
            # Move toward standoff position with arm retracted.
            if dist_to_target > 0.02:
                speed = 1.0 if abs(angle_err) < 0.4 else 0.3
                move_dx = np.clip(dx_t * speed, -env.config.max_dx, env.config.max_dx)
                move_dy = np.clip(dy_t * speed, -env.config.max_dy, env.config.max_dy)
            else:
                move_dx = 0.0
                move_dy = 0.0
            # Transition when at position and facing correctly.
            if dist_to_target < 0.03 and abs(angle_err) < 0.2:
                phase = "extend"
            action = np.array([move_dx, move_dy, dtheta, 0.0, 0.0], dtype=np.float32)

        elif phase == "extend":
            # Extend arm without moving.
            if arm_joint < arm_length - 0.01:
                action = np.array(
                    [0.0, 0.0, 0.0, env.config.max_darm, 0.0],
                    dtype=np.float32,
                )
            else:
                phase = "vacuum"
                action = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        else:  # vacuum
            action = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        state, _, terminated, _, _ = step_env.step(action)
        if terminated:
            return True, step_i + 1

    return False, max_steps


def test_grasp_vertical_solvable_seed0():
    """Test that the scripted solver solves the environment with seed=0."""
    env = ObjectCentricPushPullHook2DGraspVerticalEnv()
    step_env = env
    if MAKE_VIDEOS:
        step_env = RecordVideo(env, "unit_test_videos")
    state, _ = step_env.reset(seed=0)
    solved, steps = _solve_grasp(env, state, step_env=step_env)
    step_env.close()
    assert solved, f"Scripted solver failed on seed=0 after {steps} steps"
