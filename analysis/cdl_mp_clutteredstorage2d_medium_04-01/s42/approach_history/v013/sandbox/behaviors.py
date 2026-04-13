"""Behavior classes for ClutteredStorage2D-b3.

Strategy:
  - Robot theta = pi/2 (arm pointing straight up) for pick and place.
  - PickBlockBehavior: navigate robot directly BELOW the block's BOTTOM vertex
    so arm can extend upward to touch block bottom without body collision.
  - PlaceBlockBehavior: carry block to shelf center, extend arm into shelf, release.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from behavior import Behavior
from obs_helpers import (
    SHELF_FLOOR_Y,
    WORLD_X_MAX,
    WORLD_X_MIN,
    WORLD_Y_MIN,
    block_center,
    block_vertices,
    extract_block,
    extract_robot,
    extract_shelf_inner,
    get_outside_block_indices,
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
    RELEASE_STEPS,
    THETA_TOL,
    VACUUM_OFF,
    VACUUM_ON,
    XY_TOL,
    arm_actions,
    birrt_xy_path,
    hold_actions,
    path_to_actions,
    rotate_actions,
)

TARGET_THETA = math.pi / 2   # arm points straight up

# Small gap to leave between gripper top and block bottom face
GRASP_CLEARANCE = 0.005      # gripper stops this far below block face

# Desired arm extension during pick (distance from center to gripper)
PICK_ARM = 0.40              # good balance: far enough to reach, not too far

# Robot y below shelf for placing
PLACE_ROBOT_Y_BELOW_SHELF = 0.50  # robot_y = shelf_y - this value
# Arm extension for placing (should put block well inside shelf)
PLACE_ARM = 0.60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _block_bottom_y(block) -> float:
    """Return the y-coordinate of the lowest vertex of the block."""
    verts = block_vertices(block)
    return min(vy for _, vy in verts)


def _block_top_y(block) -> float:
    """Return the y-coordinate of the highest vertex of the block."""
    verts = block_vertices(block)
    return max(vy for _, vy in verts)


# ---------------------------------------------------------------------------
# PickBlockBehavior
# ---------------------------------------------------------------------------

class PickBlockBehavior(Behavior):
    """Navigate below block, extend arm upward to touch block bottom, vacuum on."""

    def __init__(self, block_idx: int, primitives: dict):
        self.block_idx = block_idx
        self.primitives = primitives
        self._actions: deque[np.ndarray] = deque()
        self._plan_count = 0
        self._grasp_done = False

    def initializable(self, obs: np.ndarray) -> bool:
        return not is_block_in_shelf(obs, self.block_idx)

    def terminated(self, obs: np.ndarray) -> bool:
        # Only terminate when action queue is empty AND actually holding block
        if self._actions:
            return False
        return is_holding_block(obs, self.block_idx)

    def reset(self, obs: np.ndarray) -> None:
        self._actions = deque()
        self._plan_count = 0
        self._grasp_done = False
        self._build_plan(obs)

    def step(self, obs: np.ndarray) -> np.ndarray:
        if not self._actions:
            self._plan_count += 1
            self._build_plan(obs)
        return self._actions.popleft()

    def _build_plan(self, obs: np.ndarray) -> None:
        robot = extract_robot(obs)
        block = extract_block(obs, self.block_idx)
        bcx, bcy = block_center(block)
        block_bot_y = _block_bottom_y(block)

        acts: list[np.ndarray] = []

        # 1. Retract arm first
        if robot.arm_joint > ARM_MIN_JOINT + ARM_TOL:
            acts += arm_actions(robot.arm_joint, ARM_MIN_JOINT, vacuum=VACUUM_OFF)
            robot = _fake_robot(robot, arm_joint=ARM_MIN_JOINT)

        # 2. Turn vacuum off
        acts += hold_actions(2, vacuum=VACUUM_OFF)

        # 3. Rotate to TARGET_THETA
        diff = _wrap_angle(TARGET_THETA - robot.theta)
        if abs(diff) > THETA_TOL:
            acts += rotate_actions(robot.theta, TARGET_THETA, vacuum=VACUUM_OFF)
            robot = _fake_robot(robot, theta=TARGET_THETA)

        # 4. Compute pick position:
        #    Robot positions so that at arm_joint=PICK_ARM, the GRIPPER TOP
        #    is just below the block's bottom face (with GRASP_CLEARANCE gap).
        #    gripper_top = robot_y + arm_joint + gripper_w = block_bot_y - GRASP_CLEARANCE
        #    → robot_y = block_bot_y - GRASP_CLEARANCE - PICK_ARM - gripper_w
        gripper_w = robot.gripper_width
        arm_desired = PICK_ARM
        goal_y = block_bot_y - GRASP_CLEARANCE - arm_desired - gripper_w
        goal_y = max(WORLD_Y_MIN + robot.base_radius + 0.05, goal_y)
        goal_y = min(SHELF_FLOOR_Y - robot.base_radius - 0.10, goal_y)
        goal_x = bcx

        # Recalculate actual arm needed from final robot_y
        arm_desired = block_bot_y - GRASP_CLEARANCE - goal_y - gripper_w
        arm_desired = max(ARM_MIN_JOINT + 0.01, min(ARM_MAX_JOINT - 0.1, arm_desired))

        # 5. Navigate via BiRRT
        rng = np.random.default_rng(self._plan_count * 7 + self.block_idx + 42)
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

        # 6. Extend arm from current (min) to arm_desired
        acts += arm_actions(ARM_MIN_JOINT, arm_desired, vacuum=VACUUM_OFF)

        # 7. Vacuum on for many steps to ensure grasp engagement
        acts += hold_actions(GRASP_STEPS, vacuum=VACUUM_ON)

        self._actions = deque(acts)


# ---------------------------------------------------------------------------
# PlaceBlockBehavior
# ---------------------------------------------------------------------------

class PlaceBlockBehavior(Behavior):
    """Carry grasped block to shelf, extend arm into opening, release vacuum."""

    def __init__(self, block_idx: int, primitives: dict):
        self.block_idx = block_idx
        self.primitives = primitives
        self._actions: deque[np.ndarray] = deque()
        self._plan_count = 0

    def initializable(self, obs: np.ndarray) -> bool:
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

    def _build_plan(self, obs: np.ndarray) -> None:
        robot = extract_robot(obs)
        shelf = extract_shelf_inner(obs)
        acts: list[np.ndarray] = []

        # 1. Rotate to TARGET_THETA (keep vacuum on)
        diff = _wrap_angle(TARGET_THETA - robot.theta)
        if abs(diff) > THETA_TOL:
            acts += rotate_actions(robot.theta, TARGET_THETA, vacuum=VACUUM_ON)
            robot = _fake_robot(robot, theta=TARGET_THETA)

        # 2. Retract arm (block comes along)
        if robot.arm_joint > ARM_MIN_JOINT + ARM_TOL:
            acts += arm_actions(robot.arm_joint, ARM_MIN_JOINT, vacuum=VACUUM_ON)
            robot = _fake_robot(robot, arm_joint=ARM_MIN_JOINT)

        # 3. Navigate to (shelf_cx, shelf_floor_y - offset), keeping vacuum on
        goal_x = shelf.cx
        goal_y = SHELF_FLOOR_Y - PLACE_ROBOT_Y_BELOW_SHELF
        goal_y = max(WORLD_Y_MIN + robot.base_radius + 0.05, goal_y)

        rng = np.random.default_rng(self._plan_count * 13 + self.block_idx + 200)
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

        # 4. Extend arm into shelf
        acts += arm_actions(ARM_MIN_JOINT, PLACE_ARM, vacuum=VACUUM_ON)

        # 5. Hold briefly with arm extended (let block settle inside shelf)
        acts += hold_actions(5, vacuum=VACUUM_ON)

        # 6. Release vacuum
        acts += hold_actions(RELEASE_STEPS, vacuum=VACUUM_OFF)

        # 7. Retract arm
        acts += arm_actions(PLACE_ARM, ARM_MIN_JOINT, vacuum=VACUUM_OFF)

        self._actions = deque(acts)


# ---------------------------------------------------------------------------
# AllDoneBehavior
# ---------------------------------------------------------------------------

class AllDoneBehavior(Behavior):
    """No-op when all blocks already in shelf."""

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
# Fake robot (for pre-planning)
# ---------------------------------------------------------------------------

def _fake_robot(r, **kwargs):
    from obs_helpers import RobotPose
    d = {
        "x": r.x, "y": r.y, "theta": r.theta,
        "base_radius": r.base_radius, "arm_joint": r.arm_joint,
        "arm_length": r.arm_length, "vacuum": r.vacuum,
        "gripper_height": r.gripper_height, "gripper_width": r.gripper_width,
    }
    d.update(kwargs)
    return RobotPose(**d)
