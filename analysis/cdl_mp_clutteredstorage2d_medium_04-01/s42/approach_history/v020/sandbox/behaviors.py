"""Behavior classes for ClutteredStorage2D-b3.

Strategy:
  - Face-normal approach: rotate robot to face the block's face outward normal,
    navigate to approach position, extend arm to touch block face, vacuum on.
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

# Small gap between gripper tip and block face
GRASP_CLEARANCE = 0.005

# Arm extension during pick (arm_joint value when grasping)
PICK_ARM = 0.38

# Robot y below shelf for placing
PLACE_ROBOT_Y_BELOW_SHELF = 0.55
# Arm extension for placing (should put block well inside shelf)
PLACE_ARM = 0.62

WORLD_Y_MAX = 3.0


def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _block_face_approach(block, gripper_width: float, robot_x: float = 0.0, robot_y: float = 0.0):
    """Compute face-normal approach parameters for picking a block.

    Returns (theta_approach, goal_x, goal_y, arm_desired) where:
    - theta_approach: robot orientation to face the block
    - goal_x, goal_y: robot base position for grasping
    - arm_desired: arm extension to just touch block face

    Scoring prefers approach positions closest to (robot_x, robot_y)
    to avoid navigating around the block being picked.
    """
    bcx, bcy = block_center(block)
    t = block.theta
    w, h = block.width, block.height

    # 4 outward face normals of the block (for width-faces and height-faces)
    # Width faces (at y=0 and y=h in local frame): normals along -y_local and +y_local
    # Height faces (at x=0 and x=w in local frame): normals along -x_local and +x_local
    # In world frame:
    # y_local direction: (sin(t), -cos(t)) wait, let me be careful.
    # local_x direction in world: (cos(t), sin(t))
    # local_y direction in world: (-sin(t), cos(t))
    # Face normals for the 4 sides:
    #   bottom face (local y=0): outward normal = -local_y = (sin(t), -cos(t))
    #   top face (local y=h): outward normal = +local_y = (-sin(t), cos(t))
    #   left face (local x=0): outward normal = -local_x = (-cos(t), -sin(t))
    #   right face (local x=w): outward normal = +local_x = (cos(t), sin(t))

    cos_t = math.cos(t)
    sin_t = math.sin(t)

    # Face centers and normals (all 4 faces)
    faces = [
        # (face_center_x, face_center_y, normal_x, normal_y)
        # bottom face (local y=0): center at (w/2, 0) local
        (bcx - (-sin_t)*h/2, bcy - cos_t*h/2, sin_t, -cos_t),  # wait, let me redo
    ]

    # Redo properly:
    # block center in world: bcx, bcy
    # local_x in world: (cos_t, sin_t)
    # local_y in world: (-sin_t, cos_t)

    # Width faces (perpendicular to local_y):
    #   bottom (local y=0): face_center = bcx - (-sin_t)*h/2 ...
    #   Actually: center of bottom face = block_origin + (w/2)*local_x + 0*local_y
    #   = block_origin + (w/2)*(cos_t, sin_t)
    # But block center = block_origin + (w/2)*local_x + (h/2)*local_y
    # So bottom face center = block_center - (h/2)*local_y = block_center - (h/2)*(-sin_t, cos_t)
    # = (bcx + (h/2)*sin_t, bcy - (h/2)*cos_t)
    # with outward normal = -local_y = (sin_t, -cos_t)

    # Top face center = block_center + (h/2)*local_y = (bcx - (h/2)*sin_t, bcy + (h/2)*cos_t)
    # with outward normal = +local_y = (-sin_t, cos_t)

    # Left face center = block_center - (w/2)*local_x = (bcx - (w/2)*cos_t, bcy - (w/2)*sin_t)
    # with outward normal = -local_x = (-cos_t, -sin_t)

    # Right face center = block_center + (w/2)*local_x = (bcx + (w/2)*cos_t, bcy + (w/2)*sin_t)
    # with outward normal = +local_x = (cos_t, sin_t)

    face_data = [
        # (face_cx, face_cy, nx, ny)
        (bcx + (h/2)*sin_t, bcy - (h/2)*cos_t, sin_t, -cos_t),   # bottom (local y=0)
        (bcx - (h/2)*sin_t, bcy + (h/2)*cos_t, -sin_t, cos_t),   # top (local y=h)
        (bcx - (w/2)*cos_t, bcy - (w/2)*sin_t, -cos_t, -sin_t),  # left (local x=0)
        (bcx + (w/2)*cos_t, bcy + (w/2)*sin_t, cos_t, sin_t),    # right (local x=w)
    ]

    best = None
    best_score = float('inf')
    for fcx, fcy, nx, ny in face_data:
        # Robot approaches from the direction of the outward normal
        # Robot base position: face_center + (arm + gripper_w + clearance) * normal
        arm_reach = PICK_ARM + gripper_width + GRASP_CLEARANCE
        rx = fcx + arm_reach * nx
        ry = fcy + arm_reach * ny

        # Skip positions outside the valid robot zone (below shelf)
        if ry > SHELF_FLOOR_Y - 0.25 or ry < WORLD_Y_MIN + 0.25:
            continue
        if rx < WORLD_X_MIN + 0.25 or rx > WORLD_X_MAX - 0.25:
            continue

        # Robot theta = angle toward block face = direction from robot to face center
        theta_approach = math.atan2(-ny, -nx)  # approach direction = -normal

        # Score: prefer the approach angle that makes the block land HORIZONTALLY
        # when placed (PlaceBlockBehavior rotates robot to pi/2 for placing).
        # Block theta after place = block.theta + (pi/2 - theta_approach).
        # For horizontal (theta=0 mod pi): theta_approach = pi/2 + block.theta (mod pi).
        desired_approach = math.pi / 2 + t  # = pi/2 + block.theta
        orientation_score = abs(_wrap_angle(theta_approach - desired_approach))
        # Tiebreak: prefer closer approach positions (avoids navigating through block)
        dist_score = math.sqrt((rx - robot_x) ** 2 + (ry - robot_y) ** 2) / 100.0
        score = orientation_score + dist_score

        if score < best_score:
            best_score = score
            best = (fcx, fcy, nx, ny, rx, ry, theta_approach)

    if best is None:
        # Fallback: approach from directly below
        fcx, fcy = bcx, bcy
        nx, ny = 0.0, -1.0
        arm_reach = PICK_ARM + gripper_width + GRASP_CLEARANCE
        rx = bcx
        ry = fcy + arm_reach * ny
        theta_approach = math.pi / 2
        best = (fcx, fcy, nx, ny, rx, ry, theta_approach)

    fcx, fcy, nx, ny, rx, ry, theta_approach = best

    # arm_desired: distance from robot center to face = distance to (fcx, fcy)
    # arm_joint = distance - gripper_width - GRASP_CLEARANCE ... but let's just use PICK_ARM
    arm_desired = PICK_ARM

    return theta_approach, rx, ry, arm_desired


# ---------------------------------------------------------------------------
# PickBlockBehavior
# ---------------------------------------------------------------------------

class PickBlockBehavior(Behavior):
    """Navigate to face-normal approach position, extend arm to block face, vacuum on."""

    def __init__(self, block_idx: int, primitives: dict):
        self.block_idx = block_idx
        self.primitives = primitives
        self._actions: deque[np.ndarray] = deque()
        self._plan_count = 0

    def initializable(self, obs: np.ndarray) -> bool:
        return not is_block_in_shelf(obs, self.block_idx)

    def terminated(self, obs: np.ndarray) -> bool:
        if self._actions:
            return False
        return is_holding_block(obs, self.block_idx)

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
        block = extract_block(obs, self.block_idx)

        acts: list[np.ndarray] = []

        # 1. Retract arm first (if needed)
        if robot.arm_joint > ARM_MIN_JOINT + ARM_TOL:
            acts += arm_actions(robot.arm_joint, ARM_MIN_JOINT, vacuum=VACUUM_OFF)
            robot = _fake_robot(robot, arm_joint=ARM_MIN_JOINT)

        # 2. Turn vacuum off
        acts += hold_actions(2, vacuum=VACUUM_OFF)

        # 3. Compute face-normal approach (score by distance from current robot pos)
        theta_approach, goal_x, goal_y, arm_desired = _block_face_approach(
            block, robot.gripper_width, robot_x=robot.x, robot_y=robot.y
        )

        # Clamp goal to valid region
        clearance = robot.base_radius + 0.05
        goal_x = max(WORLD_X_MIN + clearance, min(WORLD_X_MAX - clearance, goal_x))
        goal_y = max(WORLD_Y_MIN + clearance, min(SHELF_FLOOR_Y - clearance, goal_y))

        # 4. Rotate to approach theta
        diff = _wrap_angle(theta_approach - robot.theta)
        if abs(diff) > THETA_TOL:
            acts += rotate_actions(robot.theta, theta_approach, vacuum=VACUUM_OFF)
            robot = _fake_robot(robot, theta=theta_approach)

        # 5. Navigate to approach position via BiRRT (avoid all floor blocks incl. target)
        # Include ALL non-shelf blocks as obstacles (robot must go around them).
        # The goal (approach position) is outside the target block's bounding circle by design.
        block_obstacles = []
        for oi in range(3):
            ob = extract_block(obs, oi)
            ocx, ocy = block_center(ob)
            # Only treat as obstacle if below shelf
            if ocy < SHELF_FLOOR_Y - 0.1:
                half_diag = math.sqrt((ob.width / 2) ** 2 + (ob.height / 2) ** 2)
                block_obstacles.append((ocx, ocy, half_diag))

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
            block_obstacles=block_obstacles,
        )
        acts += path_to_actions(path, vacuum=VACUUM_OFF)

        # 6. Extend arm to PICK_ARM
        acts += arm_actions(ARM_MIN_JOINT, arm_desired, vacuum=VACUUM_OFF)

        # 7. Vacuum on for GRASP_STEPS
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
        # Wait for the full plan (hold + release + retract) to finish before declaring done.
        # This prevents the arm retract from pulling the block back out of the shelf.
        if self._actions:
            return False
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

        # 1. Retract arm (keep vacuum on so we don't drop block)
        if robot.arm_joint > ARM_MIN_JOINT + ARM_TOL:
            acts += arm_actions(robot.arm_joint, ARM_MIN_JOINT, vacuum=VACUUM_ON)
            robot = _fake_robot(robot, arm_joint=ARM_MIN_JOINT)

        # 2. Rotate to pi/2 (arm pointing up, toward shelf) with vacuum on
        target_theta = math.pi / 2
        diff = _wrap_angle(target_theta - robot.theta)
        if abs(diff) > THETA_TOL:
            acts += rotate_actions(robot.theta, target_theta, vacuum=VACUUM_ON)
            robot = _fake_robot(robot, theta=target_theta)

        # 3. Navigate to below shelf center
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

        # 5. Hold briefly
        acts += hold_actions(5, vacuum=VACUUM_ON)

        # 6. Release vacuum
        acts += hold_actions(RELEASE_STEPS, vacuum=VACUUM_OFF)

        # 7. Retract arm
        acts += arm_actions(PLACE_ARM, ARM_MIN_JOINT, vacuum=VACUUM_OFF)

        self._actions = deque(acts)


# ---------------------------------------------------------------------------
# MoveBlock0UpBehavior
# ---------------------------------------------------------------------------

# Robot y for lifting block0 (far enough from shelf to have good arm range)
LIFT_ROBOT_Y = 2.150
# Arm extension to push block0 high in the shelf
LIFT_ARM_HIGH = 0.72


class MoveBlock0UpBehavior(Behavior):
    """Push block 0 (pre-placed in shelf) higher to free space for blocks 1 & 2.

    Strategy:
      1. Navigate to (shelf.cx, LIFT_ROBOT_Y) with theta=pi/2
      2. Extend arm to just touch block0 bottom face
      3. Vacuum ON to suction block0
      4. Extend arm to LIFT_ARM_HIGH → block0 rises
      5. Vacuum OFF (block0 stays at higher position)
      6. Retract arm
    """

    def __init__(self, primitives: dict):
        self.primitives = primitives
        self._actions: deque[np.ndarray] = deque()
        self._plan_count = 0

    def initializable(self, obs: np.ndarray) -> bool:
        # Only run if block0 is in shelf AND its center_y is below 2.85
        if not is_block_in_shelf(obs, 0):
            return False
        block0 = extract_block(obs, 0)
        _, bcy = block_center(block0)
        return bcy < 2.85

    def terminated(self, obs: np.ndarray) -> bool:
        if self._actions:
            return False
        # Done when block0 center_y is above 2.80
        block0 = extract_block(obs, 0)
        _, bcy = block_center(block0)
        return bcy > 2.80

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
        block0 = extract_block(obs, 0)
        shelf = extract_shelf_inner(obs)
        acts: list[np.ndarray] = []

        # 1. Retract arm, vacuum off
        if robot.arm_joint > ARM_MIN_JOINT + ARM_TOL:
            acts += arm_actions(robot.arm_joint, ARM_MIN_JOINT, vacuum=VACUUM_OFF)
            robot = _fake_robot(robot, arm_joint=ARM_MIN_JOINT)
        acts += hold_actions(2, vacuum=VACUUM_OFF)

        # 2. Rotate to pi/2 (arm pointing up)
        target_theta = math.pi / 2
        diff = _wrap_angle(target_theta - robot.theta)
        if abs(diff) > THETA_TOL:
            acts += rotate_actions(robot.theta, target_theta, vacuum=VACUUM_OFF)
            robot = _fake_robot(robot, theta=target_theta)

        # 3. Navigate to (shelf.cx, LIFT_ROBOT_Y)
        goal_x = shelf.cx
        goal_y = LIFT_ROBOT_Y
        goal_y = max(WORLD_Y_MIN + robot.base_radius + 0.05, goal_y)
        rng = np.random.default_rng(self._plan_count * 17 + 100)
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

        # 4. Compute arm_joint to touch block0 bottom face
        verts = block_vertices(block0)
        block0_bottom_y = min(vy for _, vy in verts)
        arm_touch = block0_bottom_y - goal_y - robot.gripper_width - GRASP_CLEARANCE
        arm_touch = max(ARM_MIN_JOINT + 0.01, min(ARM_MAX_JOINT - 0.05, arm_touch))

        # 5. Extend arm to touch block0
        acts += arm_actions(ARM_MIN_JOINT, arm_touch, vacuum=VACUUM_OFF)

        # 6. Vacuum ON to suction block0
        acts += hold_actions(GRASP_STEPS, vacuum=VACUUM_ON)

        # 7. Extend arm to LIFT_ARM_HIGH (pushes block0 up)
        acts += arm_actions(arm_touch, LIFT_ARM_HIGH, vacuum=VACUUM_ON)

        # 8. Hold briefly
        acts += hold_actions(5, vacuum=VACUUM_ON)

        # 9. Release vacuum (block0 stays at new higher position)
        acts += hold_actions(RELEASE_STEPS, vacuum=VACUUM_OFF)

        # 10. Retract arm
        acts += arm_actions(LIFT_ARM_HIGH, ARM_MIN_JOINT, vacuum=VACUUM_OFF)

        self._actions = deque(acts)


# ---------------------------------------------------------------------------
# AllDoneBehavior
# ---------------------------------------------------------------------------

class AllDoneBehavior(Behavior):
    """No-op when all blocks already in shelf."""

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
