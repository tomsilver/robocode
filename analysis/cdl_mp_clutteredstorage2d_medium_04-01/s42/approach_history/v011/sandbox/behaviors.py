"""Behavior classes for ClutteredStorage2D-b3.

Strategy:
  - Robot theta always kept at pi/2 (arm pointing straight up).
  - PickBlockBehavior: navigate below block, extend arm up to contact, vacuum on.
  - PlaceBlockBehavior: navigate to shelf center x, extend arm into shelf, vacuum off.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from behavior import Behavior
from obs_helpers import (
    BLOCK_NAMES,
    HOLDING_DIST_THRESHOLD,
    NUM_BLOCKS,
    SHELF_FLOOR_Y,
    WORLD_X_MAX,
    WORLD_X_MIN,
    WORLD_Y_MIN,
    extract_block,
    extract_robot,
    extract_shelf_inner,
    get_outside_block_indices,
    gripper_tip_position,
    is_block_in_shelf,
    is_holding_block,
)
from act_helpers import (
    ARM_MAX_JOINT,
    ARM_MIN_JOINT,
    ARM_TOL,
    DARM_LIM,
    DTH_LIM,
    DX_LIM,
    DY_LIM,
    GRASP_STEPS,
    PICK_ARM_EXTEND,
    PLACE_ARM_EXTEND,
    RELEASE_STEPS,
    THETA_TOL,
    VACUUM_OFF,
    VACUUM_ON,
    XY_TOL,
    arm_actions,
    birrt_xy_path,
    hold_actions,
    navigate_actions,
    path_to_actions,
    rotate_actions,
)

TARGET_THETA = math.pi / 2   # arm points straight up

# Margin above SHELF_FLOOR_Y where robot base stops (can't enter shelf)
ROBOT_Y_CLEARANCE = 0.30     # robot stays this far below shelf floor

# How far below the block the robot base sits during pick
PICK_ROBOT_OFFSET = 0.50     # robot_y = block_y - PICK_ROBOT_OFFSET

# Maximum arm extension used during picking
PICK_ARM_MAX = 0.72

# Arm extension for placing (block should clear shelf floor)
PLACE_ARM_TARGET = 0.65      # arm_joint when dropping block

# Y-position of robot during place
PLACE_ROBOT_Y_OFFSET = 0.55  # robot_y = shelf_floor_y - PLACE_ROBOT_Y_OFFSET

# Tolerance for "block inside shelf" height check
SHELF_ENTRY_ARM = 0.65       # guaranteed to put block inside shelf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _make_deque(lst) -> deque:
    return deque(lst)


# ---------------------------------------------------------------------------
# PickBlockBehavior
# ---------------------------------------------------------------------------

class PickBlockBehavior(Behavior):
    """Navigate to block, extend arm from below, vacuum on to grasp."""

    def __init__(self, block_idx: int, primitives: dict):
        self.block_idx = block_idx
        self.primitives = primitives
        self._actions: deque[np.ndarray] = deque()
        self._plan_count = 0

    # -- Behavior interface --------------------------------------------------

    def initializable(self, obs: np.ndarray) -> bool:
        return not is_block_in_shelf(obs, self.block_idx)

    def terminated(self, obs: np.ndarray) -> bool:
        return is_holding_block(obs, self.block_idx)

    def reset(self, obs: np.ndarray) -> None:
        self._actions = deque()
        self._plan_count = 0
        self._build_plan(obs)

    def step(self, obs: np.ndarray) -> np.ndarray:
        if not self._actions:
            # Re-plan from current state
            self._plan_count += 1
            self._build_plan(obs)
        return self._actions.popleft()

    # -- Planning ------------------------------------------------------------

    def _build_plan(self, obs: np.ndarray) -> None:
        robot = extract_robot(obs)
        block = extract_block(obs, self.block_idx)
        acts: list[np.ndarray] = []

        # 1. Retract arm to minimum first
        if robot.arm_joint > ARM_MIN_JOINT + ARM_TOL:
            acts += arm_actions(robot.arm_joint, ARM_MIN_JOINT, vacuum=VACUUM_OFF)
            robot = _fake_robot(robot, arm_joint=ARM_MIN_JOINT)

        # 2. Rotate to TARGET_THETA
        diff = _wrap_angle(TARGET_THETA - robot.theta)
        if abs(diff) > THETA_TOL:
            acts += rotate_actions(robot.theta, TARGET_THETA, vacuum=VACUUM_OFF)
            robot = _fake_robot(robot, theta=TARGET_THETA)

        # 3. Compute pick position: robot directly below block
        #    arm_joint will extend to reach block.y from robot.y
        arm_needed = block.y - robot.y
        arm_needed = max(ARM_MIN_JOINT + 0.05, min(PICK_ARM_MAX, arm_needed))
        goal_y = block.y - arm_needed
        goal_y = max(WORLD_Y_MIN + robot.base_radius + 0.05, goal_y)
        goal_y = min(SHELF_FLOOR_Y - robot.base_radius - 0.05, goal_y)
        goal_x = block.x

        # Recompute arm_needed after clamping goal_y
        arm_needed = block.y - goal_y
        arm_needed = max(ARM_MIN_JOINT, min(ARM_MAX_JOINT - 0.05, arm_needed))

        # 4. Navigate via BiRRT
        rng = np.random.default_rng(self._plan_count * 7 + self.block_idx)
        path = birrt_xy_path(
            self.primitives,
            start_xy=np.array([robot.x, robot.y]),
            goal_xy=np.array([goal_x, goal_y]),
            base_radius=robot.base_radius,
            shelf_floor_y=SHELF_FLOOR_Y,
            world_x_min=WORLD_X_MIN,
            world_x_max=WORLD_X_MAX,
            world_y_min=WORLD_Y_MIN,
            rng=rng,
        )
        acts += path_to_actions(path, vacuum=VACUUM_OFF)
        robot = _fake_robot(robot, x=goal_x, y=goal_y)

        # 5. Extend arm to reach block
        acts += arm_actions(ARM_MIN_JOINT, arm_needed, vacuum=VACUUM_OFF)
        robot = _fake_robot(robot, arm_joint=arm_needed)

        # 6. Vacuum on for several steps to ensure grasp
        acts += hold_actions(GRASP_STEPS, vacuum=VACUUM_ON)

        self._actions = _make_deque(acts)


# ---------------------------------------------------------------------------
# PlaceBlockBehavior
# ---------------------------------------------------------------------------

class PlaceBlockBehavior(Behavior):
    """Carry block to shelf opening, extend arm into shelf, release vacuum."""

    def __init__(self, block_idx: int, primitives: dict):
        self.block_idx = block_idx
        self.primitives = primitives
        self._actions: deque[np.ndarray] = deque()
        self._plan_count = 0

    # -- Behavior interface --------------------------------------------------

    def initializable(self, obs: np.ndarray) -> bool:
        # Precondition: robot is holding this block
        return is_holding_block(obs, self.block_idx)

    def terminated(self, obs: np.ndarray) -> bool:
        return is_block_in_shelf(obs, self.block_idx)

    def reset(self, obs: np.ndarray) -> None:
        self._actions = deque()
        self._plan_count = 0
        self._build_plan(obs)

    def step(self, obs: np.ndarray) -> np.ndarray:
        if not self._actions:
            self._plan_count += 1
            self._build_plan(obs)
        return self._actions.popleft()

    # -- Planning ------------------------------------------------------------

    def _build_plan(self, obs: np.ndarray) -> None:
        robot = extract_robot(obs)
        shelf = extract_shelf_inner(obs)
        acts: list[np.ndarray] = []

        # 1. Rotate to TARGET_THETA (keeping vacuum on throughout)
        diff = _wrap_angle(TARGET_THETA - robot.theta)
        if abs(diff) > THETA_TOL:
            acts += rotate_actions(robot.theta, TARGET_THETA, vacuum=VACUUM_ON)
            robot = _fake_robot(robot, theta=TARGET_THETA)

        # 2. Retract arm (block comes along, stays near robot)
        if robot.arm_joint > ARM_MIN_JOINT + ARM_TOL:
            acts += arm_actions(robot.arm_joint, ARM_MIN_JOINT, vacuum=VACUUM_ON)
            robot = _fake_robot(robot, arm_joint=ARM_MIN_JOINT)

        # 3. Navigate to shelf center x, slightly below shelf
        goal_x = shelf.cx
        goal_y = SHELF_FLOOR_Y - PLACE_ROBOT_Y_OFFSET
        goal_y = max(WORLD_Y_MIN + robot.base_radius + 0.05, goal_y)

        rng = np.random.default_rng(self._plan_count * 13 + self.block_idx + 100)
        path = birrt_xy_path(
            self.primitives,
            start_xy=np.array([robot.x, robot.y]),
            goal_xy=np.array([goal_x, goal_y]),
            base_radius=robot.base_radius,
            shelf_floor_y=SHELF_FLOOR_Y,
            world_x_min=WORLD_X_MIN,
            world_x_max=WORLD_X_MAX,
            world_y_min=WORLD_Y_MIN,
            rng=rng,
        )
        acts += path_to_actions(path, vacuum=VACUUM_ON)
        robot = _fake_robot(robot, x=goal_x, y=goal_y)

        # 4. Extend arm into shelf (block rises into shelf interior)
        acts += arm_actions(ARM_MIN_JOINT, PLACE_ARM_TARGET, vacuum=VACUUM_ON)
        robot = _fake_robot(robot, arm_joint=PLACE_ARM_TARGET)

        # 5. Release vacuum
        acts += hold_actions(RELEASE_STEPS, vacuum=VACUUM_OFF)

        # 6. Retract arm
        acts += arm_actions(PLACE_ARM_TARGET, ARM_MIN_JOINT, vacuum=VACUUM_OFF)

        self._actions = _make_deque(acts)


# ---------------------------------------------------------------------------
# AllDoneBehavior (terminal no-op)
# ---------------------------------------------------------------------------

class AllDoneBehavior(Behavior):
    """No-op behavior when all blocks are already in shelf."""

    def __init__(self):
        pass

    def initializable(self, obs: np.ndarray) -> bool:
        return len(get_outside_block_indices(obs)) == 0

    def terminated(self, obs: np.ndarray) -> bool:
        return True

    def reset(self, obs: np.ndarray) -> None:
        pass

    def step(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(5, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fake robot helper (for pre-planning, no env step)
# ---------------------------------------------------------------------------

def _fake_robot(r: object, **kwargs):
    """Return a copy of RobotPose with overridden fields."""
    from obs_helpers import RobotPose
    d = {
        "x": r.x, "y": r.y, "theta": r.theta,
        "base_radius": r.base_radius, "arm_joint": r.arm_joint,
        "arm_length": r.arm_length, "vacuum": r.vacuum,
        "gripper_height": r.gripper_height, "gripper_width": r.gripper_width,
    }
    d.update(kwargs)
    return RobotPose(**d)
