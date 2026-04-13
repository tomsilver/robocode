"""Behavior classes for ClutteredStorage2D-b3-v0.

Two main behaviors:
  PickupBlock(block_name) — navigate to block, grasp it
  PlaceBlock()            — navigate to shelf, release block

Each has internal phases to handle the multi-step sub-task.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray

from behavior import Behavior
from obs_helpers import (
    ARM_MAX,
    ARM_MIN,
    BLOCK_HEIGHT,
    BLOCK_NAMES,
    BLOCK_WIDTH,
    GRIPPER_WIDTH,
    ROBOT_BASE_RADIUS,
    ROBOT_MAX_Y_NAV,
    SHELF_FLOOR_Y,
    WORLD_MAX_X,
    WORLD_MAX_Y,
    WORLD_MIN_X,
    WORLD_MIN_Y,
    extract_rect,
    extract_robot,
    get_block_center,
    get_blocks_outside_shelf,
    get_shelf_slot,
    is_block_in_shelf,
    wrap_angle,
)
from act_helpers import (
    APPROACH_DIST,
    BLOCK_CLEAR,
    DARM_LIM,
    DTH_LIM,
    DX_LIM,
    DY_LIM,
    NAV_MARGIN,
    SHELF_APPROACH_X,
    SHELF_APPROACH_Y,
    SHELF_PLACE_ARM,
    SHELF_PLACE_THETA,
    SUCTION_DIST_OFFSET,
    VAC_OFF,
    VAC_ON,
    make_action,
    servo_arm,
    servo_theta,
)

# ---------------------------------------------------------------------------
# Collision helpers for BiRRT
# ---------------------------------------------------------------------------

# Navigation safety margins
NAV_ROBOT_R = ROBOT_BASE_RADIUS + 0.02   # small extra clearance for robot circle

# Phases
_PH_NAVIGATE = 0
_PH_ORIENT = 1
_PH_EXTEND = 2
_PH_VACUUM = 3
_PH_DONE = 4

# Tolerances
XY_TOL = 0.06       # position tolerance (meters)
THETA_TOL = 0.08    # angle tolerance (radians)
ARM_TOL = 0.04      # arm_joint tolerance (meters)
VACUUM_SETTLE = 3   # steps to hold vacuum on before considering grasped
DARM_SMALL = 0.02   # small arm step for precise extension


def _robot_collides(rx: float, ry: float, rr: float,
                    obs: NDArray, exclude_block: str | None = None) -> bool:
    """Check if a robot circle (rx, ry, rr) collides with world or blocks."""
    # World boundary
    if rx - rr < WORLD_MIN_X or rx + rr > WORLD_MAX_X:
        return True
    if ry - rr < WORLD_MIN_Y or ry + rr > WORLD_MAX_Y:
        return True
    # Shelf floor: robot base must stay below shelf
    if ry + rr > SHELF_FLOOR_Y:
        return True
    # Block obstacles
    for name in BLOCK_NAMES:
        if name == exclude_block:
            continue
        cx, cy = get_block_center(obs, name)
        dist = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
        # Conservative clearance: robot radius + block half-diagonal
        block_diag = math.sqrt(BLOCK_WIDTH ** 2 + BLOCK_HEIGHT ** 2) / 2
        if dist < rr + block_diag + 0.02:
            return True
    return False


def _build_birrt(obs: NDArray, exclude_block: str | None, primitives: dict) -> Any:
    """Build a BiRRT planner for navigating robot base in (x, y)."""
    BiRRT = primitives["BiRRT"]
    rng = np.random.default_rng(42)

    rr = NAV_ROBOT_R

    def sample_fn(state: NDArray) -> NDArray:
        x = rng.uniform(WORLD_MIN_X + rr, WORLD_MAX_X - rr)
        y = rng.uniform(WORLD_MIN_Y + rr, ROBOT_MAX_Y_NAV - rr)
        return np.array([x, y])

    def extend_fn(s1: NDArray, s2: NDArray):
        dist = np.linalg.norm(s2 - s1)
        step = 0.05
        n = max(1, int(dist / step))
        pts = []
        for i in range(1, n + 1):
            t = i / n
            pts.append(s1 + t * (s2 - s1))
        return pts

    def collision_fn(state: NDArray) -> bool:
        return _robot_collides(state[0], state[1], rr, obs, exclude_block)

    def distance_fn(s1: NDArray, s2: NDArray) -> float:
        return float(np.linalg.norm(s2 - s1))

    birrt = BiRRT(sample_fn, extend_fn, collision_fn, distance_fn, rng,
                  num_attempts=5, num_iters=2000, smooth_amt=50)
    return birrt


# ---------------------------------------------------------------------------
# PickupBlock behavior
# ---------------------------------------------------------------------------

class PickupBlock(Behavior):
    """Navigate to a block and grasp it."""

    def __init__(self, block_name: str, primitives: dict):
        self._block_name = block_name
        self._primitives = primitives
        self._phase = _PH_DONE
        self._path: list = []
        self._path_idx: int = 0
        self._target_theta: float = 0.0
        self._target_arm: float = 0.0
        self._vacuum_count: int = 0
        self._approach_rx: float = 0.0
        self._approach_ry: float = 0.0

    def initializable(self, obs: NDArray) -> bool:
        robot = extract_robot(obs)
        return (not is_block_in_shelf(obs, self._block_name)) and robot.vacuum < 0.5

    def terminated(self, obs: NDArray) -> bool:
        robot = extract_robot(obs)
        return robot.vacuum > 0.5 and self._vacuum_count >= VACUUM_SETTLE

    def reset(self, obs: NDArray) -> None:
        self._phase = _PH_NAVIGATE
        self._path = []
        self._path_idx = 0
        self._vacuum_count = 0
        self._plan_navigation(obs)

    def _compute_approach(self, obs: NDArray) -> tuple[float, float, float]:
        """Compute (approach_rx, approach_ry, target_theta) for grasping block."""
        cx, cy = get_block_center(obs, self._block_name)
        p = extract_rect(obs, self._block_name)
        theta_b = p.theta
        # Two possible approach directions (perpendicular to block long axis)
        # Short axis 1: (-sin, cos)
        n1x, n1y = -math.sin(theta_b), math.cos(theta_b)
        # Short axis 2: (sin, -cos)
        n2x, n2y = math.sin(theta_b), -math.cos(theta_b)

        D = APPROACH_DIST
        # Try both directions, pick the one with valid approach position
        candidates = [
            (cx + D * n1x, cy + D * n1y, math.atan2(-n1y, -n1x)),
            (cx + D * n2x, cy + D * n2y, math.atan2(-n2y, -n2x)),
            # Also try approach from further away if needed
            (cx + (D + 0.1) * n1x, cy + (D + 0.1) * n1y, math.atan2(-n1y, -n1x)),
            (cx + (D + 0.1) * n2x, cy + (D + 0.1) * n2y, math.atan2(-n2y, -n2x)),
        ]
        rr = NAV_ROBOT_R
        best = None
        for rx, ry, rth in candidates:
            if (WORLD_MIN_X + rr < rx < WORLD_MAX_X - rr and
                    WORLD_MIN_Y + rr < ry < ROBOT_MAX_Y_NAV - rr):
                best = (rx, ry, rth)
                break
        if best is None:
            # Fallback: approach from below
            best = (cx, max(WORLD_MIN_Y + rr + 0.05, cy - D), math.pi / 2)
        return best

    def _plan_navigation(self, obs: NDArray) -> None:
        """Plan BiRRT path to approach position."""
        robot = extract_robot(obs)
        rx, ry, rth = self._compute_approach(obs)
        self._approach_rx = rx
        self._approach_ry = ry
        self._target_theta = rth

        # Build BiRRT and plan
        birrt = _build_birrt(obs, self._block_name, self._primitives)
        start = np.array([robot.x, robot.y])
        goal = np.array([rx, ry])
        path = birrt.query(start, goal)

        if path is None or len(path) == 0:
            # Fallback: direct path
            path = [start, goal]
        self._path = path
        self._path_idx = 0

        # Compute arm target: put suction zone just inside block front face
        # Block front face from approach direction = dist - block_height/2
        # Suction zone at arm_joint + SUCTION_DIST_OFFSET
        # Target: arm_joint = dist - block_height/2 - SUCTION_DIST_OFFSET + small_margin
        cx, cy = get_block_center(obs, self._block_name)
        dist = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
        # Aim to put suction zone just inside block (arm_joint + offset = dist - half_block_short)
        half_short = BLOCK_HEIGHT / 2  # approaching along short axis
        suction_target_dist = dist - half_short + 0.01  # slightly inside block
        self._target_arm = max(ARM_MIN, min(ARM_MAX - 0.1,
                                            suction_target_dist - SUCTION_DIST_OFFSET))

    def step(self, obs: NDArray) -> NDArray:
        robot = extract_robot(obs)

        if self._phase == _PH_NAVIGATE:
            # Advance past close waypoints
            while self._path_idx < len(self._path):
                wp = self._path[self._path_idx]
                dx = wp[0] - robot.x
                dy = wp[1] - robot.y
                if math.sqrt(dx ** 2 + dy ** 2) < XY_TOL:
                    self._path_idx += 1
                else:
                    break
            if self._path_idx >= len(self._path):
                self._phase = _PH_ORIENT
            else:
                wp = self._path[self._path_idx]
                dx = wp[0] - robot.x
                dy = wp[1] - robot.y
                adx = np.clip(dx, -DX_LIM, DX_LIM)
                ady = np.clip(dy, -DY_LIM, DY_LIM)
                cx, cy = get_block_center(obs, self._block_name)
                desired_th = math.atan2(cy - robot.y, cx - robot.x)
                dth = servo_theta(robot.theta, desired_th)
                return make_action(adx, ady, dth, 0.0, VAC_OFF)

        if self._phase == _PH_ORIENT:
            # Check if close enough to approach pos
            dx = self._approach_rx - robot.x
            dy = self._approach_ry - robot.y
            dist = math.sqrt(dx ** 2 + dy ** 2)
            if dist > XY_TOL:
                # Still need to move
                adx = np.clip(dx, -DX_LIM, DX_LIM)
                ady = np.clip(dy, -DY_LIM, DY_LIM)
                dth = servo_theta(robot.theta, self._target_theta)
                return make_action(adx, ady, dth, 0.0, VAC_OFF)
            # Orient toward block
            dth = servo_theta(robot.theta, self._target_theta)
            if abs(wrap_angle(robot.theta - self._target_theta)) < THETA_TOL:
                self._phase = _PH_EXTEND
            return make_action(0.0, 0.0, dth, 0.0, VAC_OFF)

        if self._phase == _PH_EXTEND:
            # Check position still ok
            dx = self._approach_rx - robot.x
            dy = self._approach_ry - robot.y
            dist = math.sqrt(dx ** 2 + dy ** 2)
            if dist > XY_TOL * 2:
                # Drifted — re-navigate
                self._phase = _PH_NAVIGATE
                self._plan_navigation(obs)
                return make_action(0.0, 0.0, 0.0, 0.0, VAC_OFF)
            # Orient toward block
            dth = servo_theta(robot.theta, self._target_theta)
            if abs(wrap_angle(robot.theta - self._target_theta)) > THETA_TOL:
                return make_action(0.0, 0.0, dth, 0.0, VAC_OFF)
            # Extend arm with vacuum ON using small steps to avoid over-shooting
            darm = min(DARM_SMALL, max(-DARM_LIM, self._target_arm - robot.arm_joint))
            if robot.arm_joint >= self._target_arm - 0.005:
                self._phase = _PH_VACUUM
            return make_action(0.0, 0.0, dth * 0.1, darm, VAC_ON)

        if self._phase == _PH_VACUUM:
            # Keep vacuum on, count settle steps
            self._vacuum_count += 1
            dth = servo_theta(robot.theta, self._target_theta)
            return make_action(0.0, 0.0, dth * 0.1, 0.0, VAC_ON)

        # Fallback
        return make_action(0.0, 0.0, 0.0, 0.0, VAC_OFF)


# ---------------------------------------------------------------------------
# PlaceBlock behavior
# ---------------------------------------------------------------------------

class PlaceBlock(Behavior):
    """Navigate to shelf with held block and release it."""

    _PH_NAVIGATE = 0
    _PH_ORIENT = 1
    _PH_EXTEND = 2
    _PH_RELEASE = 3
    _PH_RETRACT2 = 4
    _PH_DONE = 5

    def __init__(self, primitives: dict):
        self._primitives = primitives
        self._phase = self._PH_DONE
        self._path: list = []
        self._path_idx: int = 0
        self._held_block: str | None = None
        self._release_count: int = 0

    def initializable(self, obs: NDArray) -> bool:
        robot = extract_robot(obs)
        return robot.vacuum > 0.5

    def terminated(self, obs: NDArray) -> bool:
        # Done when the held block is in the shelf (vacuum off + block placed)
        robot = extract_robot(obs)
        if robot.vacuum > 0.5:
            return False
        # Check if any block that was outside is now in shelf
        # (approximate: check all blocks)
        return all(is_block_in_shelf(obs, name) for name in BLOCK_NAMES
                   if name != self._held_block) or self._phase == self._PH_DONE

    def reset(self, obs: NDArray) -> None:
        self._phase = self._PH_NAVIGATE
        self._path = []
        self._path_idx = 0
        self._release_count = 0
        self._target_x = 0.0
        self._target_y = 0.0
        # Identify held block (we'll infer it from context — use the one closest to gripper)
        robot = extract_robot(obs)
        gx = robot.x + math.cos(robot.theta) * robot.arm_joint
        gy = robot.y + math.sin(robot.theta) * robot.arm_joint
        min_dist = float('inf')
        self._held_block = None
        for name in BLOCK_NAMES:
            if is_block_in_shelf(obs, name):
                continue
            cx, cy = get_block_center(obs, name)
            d = math.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
            if d < min_dist:
                min_dist = d
                self._held_block = name

        # Plan navigation immediately
        self._plan_navigation(obs)

    def _plan_navigation(self, obs: NDArray) -> None:
        robot = extract_robot(obs)
        x1, y1, w1, h1 = get_shelf_slot(obs)
        # Target: center of shelf slot x, below shelf
        target_x = x1 + w1 / 2
        target_x = max(WORLD_MIN_X + NAV_ROBOT_R + 0.05,
                       min(WORLD_MAX_X - NAV_ROBOT_R - 0.05, target_x))
        target_y = SHELF_APPROACH_Y

        birrt = _build_birrt(obs, self._held_block, self._primitives)
        start = np.array([robot.x, robot.y])
        goal = np.array([target_x, target_y])
        path = birrt.query(start, goal)
        if path is None or len(path) == 0:
            path = [start, goal]
        self._path = path
        self._path_idx = 0
        self._target_x = target_x
        self._target_y = target_y

    def step(self, obs: NDArray) -> NDArray:
        robot = extract_robot(obs)

        if self._phase == self._PH_NAVIGATE:
            # Advance past close waypoints
            while self._path_idx < len(self._path):
                wp = self._path[self._path_idx]
                dx = wp[0] - robot.x
                dy = wp[1] - robot.y
                if math.sqrt(dx ** 2 + dy ** 2) < XY_TOL:
                    self._path_idx += 1
                else:
                    break
            if self._path_idx >= len(self._path):
                self._phase = self._PH_ORIENT
            else:
                wp = self._path[self._path_idx]
                dx = wp[0] - robot.x
                dy = wp[1] - robot.y
                adx = np.clip(dx, -DX_LIM, DX_LIM)
                ady = np.clip(dy, -DY_LIM, DY_LIM)
                dth = servo_theta(robot.theta, SHELF_PLACE_THETA)
                return make_action(adx, ady, dth, 0.0, VAC_ON)

        if self._phase == self._PH_ORIENT:
            # Servo theta and also correct robot x so block ends up centered in slot
            dth = servo_theta(robot.theta, SHELF_PLACE_THETA)
            th_err = abs(wrap_angle(robot.theta - SHELF_PLACE_THETA))
            # Compute adjusted robot target x: move robot so block centers on slot
            slot = get_shelf_slot(obs)
            slot_cx = slot[0] + slot[2] / 2
            if self._held_block:
                bcx, _ = get_block_center(obs, self._held_block)
                block_offset_x = bcx - robot.x
            else:
                block_offset_x = 0.0
            adj_x = slot_cx - block_offset_x  # robot x that puts block at slot center
            dx = adj_x - robot.x
            dy = self._target_y - robot.y
            pos_ok = abs(dx) < 0.015 and abs(dy) < 0.05
            th_ok = th_err < 0.03
            if pos_ok and th_ok:
                self._phase = self._PH_EXTEND
            adx = np.clip(dx, -DX_LIM, DX_LIM)
            ady = np.clip(dy, -DY_LIM, DY_LIM)
            return make_action(adx, ady, dth, 0.0, VAC_ON)

        if self._phase == self._PH_EXTEND:
            # Keep robot x corrected for block offset; extend arm with small steps
            slot = get_shelf_slot(obs)
            slot_cx = slot[0] + slot[2] / 2
            if self._held_block:
                bcx, _ = get_block_center(obs, self._held_block)
                block_offset_x = bcx - robot.x
            else:
                block_offset_x = 0.0
            adj_x = slot_cx - block_offset_x
            dx = adj_x - robot.x
            dy = self._target_y - robot.y
            dth = servo_theta(robot.theta, SHELF_PLACE_THETA)
            adx = np.clip(dx * 0.5, -DX_LIM, DX_LIM)
            ady = np.clip(dy * 0.5, -DY_LIM, DY_LIM)
            darm = min(DARM_SMALL, max(-DARM_LIM, SHELF_PLACE_ARM - robot.arm_joint))
            if robot.arm_joint >= SHELF_PLACE_ARM - 0.005:
                self._phase = self._PH_RELEASE
            return make_action(adx, ady, dth * 0.2, darm, VAC_ON)

        if self._phase == self._PH_RELEASE:
            # Turn off vacuum to release block
            self._release_count += 1
            if self._release_count >= 3:
                self._phase = self._PH_RETRACT2
            return make_action(0.0, 0.0, 0.0, 0.0, VAC_OFF)

        if self._phase == self._PH_RETRACT2:
            # Retract arm
            darm = servo_arm(robot.arm_joint, ARM_MIN)
            if abs(robot.arm_joint - ARM_MIN) < ARM_TOL:
                self._phase = self._PH_DONE
            return make_action(0.0, 0.0, 0.0, darm, VAC_OFF)

        if self._phase == self._PH_DONE:
            return make_action(0.0, 0.0, 0.0, 0.0, VAC_OFF)

        return make_action(0.0, 0.0, 0.0, 0.0, VAC_OFF)

    def terminated(self, obs: NDArray) -> bool:
        return self._phase == self._PH_DONE and extract_robot(obs).vacuum < 0.5
