"""Behavior classes for ClutteredStorage2D-b3-v0.

Each behavior handles a single sub-task:
  PickBlockBehavior  — navigate to block, grasp, retract arm
  PlaceBlockBehavior — navigate to shelf, extend arm, release
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from behavior import Behavior
from obs_helpers import (
    BLOCK_NAMES,
    ARM_MIN_JOINT,
    ARM_MAX_JOINT,
    ROBOT_RADIUS,
    WORLD_MIN_X,
    WORLD_MAX_X,
    WORLD_MIN_Y,
    block_center_from_obs,
    extract_robot,
    get_outside_blocks,
    gripper_pos,
    is_block_in_shelf,
    shelf_inner_center,
    shelf_y_bottom,
)
from act_helpers import (
    APPROACH_DIST,
    ARM_TOL,
    DEPOSIT_ARM_JOINT,
    DEPOSIT_SLOT_Y_NAV,
    DARM_LIM,
    DTH_LIM,
    DX_LIM,
    DY_LIM,
    EXTEND_DARM,
    POS_TOL,
    THETA_TOL,
    THETA_UP,
    angle_diff,
    build_action,
    make_birrt,
    zero_action,
)

# ---------------------------------------------------------------------------
# Internal phase tags
# ---------------------------------------------------------------------------
_ALIGN = "ALIGN"
_NAVIGATE = "NAVIGATE"
_EXTEND = "EXTEND"
_RETRACT = "RETRACT"
_RELEASE = "RELEASE"
_DONE = "DONE"


# ---------------------------------------------------------------------------
# PickBlockBehavior
# ---------------------------------------------------------------------------


class PickBlockBehavior(Behavior):
    """Navigate to a block from below, grasp it, and retract the arm.

    Precondition:  target block is NOT in the shelf
    Subgoal:       robot holds the block (vacuum on, arm retracted)
    """

    def __init__(self, block_name: str, primitives: dict) -> None:
        self._block_name = block_name
        self._primitives = primitives
        self._phase: str = _ALIGN
        self._path: list[tuple[float, float]] = []
        self._path_step: int = 0
        self._arm_target: float = APPROACH_DIST
        self._rng: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------
    def initializable(self, obs: np.ndarray) -> bool:
        """Precondition: block is not yet inside the shelf."""
        return not is_block_in_shelf(obs, self._block_name)

    def terminated(self, obs: np.ndarray) -> bool:
        """Subgoal: arm retracted and vacuum on (block held)."""
        robot = extract_robot(obs)
        return (
            self._phase == _DONE
            and robot.arm_joint <= ARM_MIN_JOINT + ARM_TOL
            and robot.vacuum >= 0.5
        )

    # ------------------------------------------------------------------
    def reset(self, obs: np.ndarray) -> None:
        self._phase = _ALIGN
        self._path = []
        self._path_step = 0
        self._rng = np.random.default_rng()
        bcx, bcy = block_center_from_obs(obs, self._block_name)
        # Clamp approach y so robot stays in world
        raw_target_y = bcy - APPROACH_DIST
        self._target_robot_x = bcx
        self._target_robot_y = max(raw_target_y, ROBOT_RADIUS + 0.05)
        # Actual arm extension to reach block from clamped position
        self._arm_target = min(bcy - self._target_robot_y, ARM_MAX_JOINT - 0.05)

    # ------------------------------------------------------------------
    def step(self, obs: np.ndarray) -> np.ndarray:
        robot = extract_robot(obs)

        if self._phase == _ALIGN:
            # Rotate arm to point upward (theta = pi/2)
            dth = angle_diff(THETA_UP, robot.theta)
            if abs(dth) < THETA_TOL:
                self._phase = _NAVIGATE
                self._plan_path(obs)
                return zero_action()
            dtheta = np.clip(dth, -DTH_LIM, DTH_LIM)
            return build_action(dtheta=dtheta, vacuum=0.0)

        elif self._phase == _NAVIGATE:
            if self._path_step >= len(self._path):
                self._phase = _EXTEND
                return zero_action()
            tx, ty = self._path[self._path_step]
            dx = np.clip(tx - robot.x, -DX_LIM, DX_LIM)
            dy = np.clip(ty - robot.y, -DY_LIM, DY_LIM)
            # Advance waypoint when close enough
            if abs(tx - robot.x) < POS_TOL and abs(ty - robot.y) < POS_TOL:
                self._path_step += 1
            return build_action(dx=dx, dy=dy, vacuum=0.0)

        elif self._phase == _EXTEND:
            # Extend arm toward block with SMALL steps to enter suction window.
            # Suction zone is just beyond gripper tip; large steps overshoot window.
            if robot.arm_joint >= self._arm_target - ARM_TOL:
                self._phase = _RETRACT
                return build_action(vacuum=1.0)
            darm = min(EXTEND_DARM, self._arm_target - robot.arm_joint)
            return build_action(darm=darm, vacuum=1.0)

        elif self._phase == _RETRACT:
            # Retract arm (block held by vacuum)
            if robot.arm_joint <= ARM_MIN_JOINT + ARM_TOL:
                self._phase = _DONE
                return build_action(vacuum=1.0)
            darm = -min(DARM_LIM, robot.arm_joint - ARM_MIN_JOINT)
            return build_action(darm=darm, vacuum=1.0)

        else:  # _DONE
            return build_action(vacuum=1.0)

    # ------------------------------------------------------------------
    def _plan_path(self, obs: np.ndarray) -> None:
        robot = extract_robot(obs)
        sy = shelf_y_bottom(obs)
        birrt = make_birrt(self._primitives, sy, self._rng)
        start = (robot.x, robot.y)
        goal = (self._target_robot_x, self._target_robot_y)
        path = birrt.query(start, goal)
        if path is None or len(path) < 2:
            self._path = [goal]
        else:
            self._path = list(path[1:])  # skip start (already there)
        self._path_step = 0


# ---------------------------------------------------------------------------
# PlaceBlockBehavior
# ---------------------------------------------------------------------------


class PlaceBlockBehavior(Behavior):
    """Navigate to below the shelf and deposit the held block inside.

    Precondition:  robot holds a block (vacuum on)
    Subgoal:       all target blocks are inside the shelf
    """

    def __init__(self, slot_idx: int, primitives: dict) -> None:
        """
        slot_idx: 0 or 1 — which deposit slot y position to use.
        """
        self._slot_idx = slot_idx
        self._primitives = primitives
        self._phase: str = _ALIGN
        self._path: list[tuple[float, float]] = []
        self._path_step: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------
    def initializable(self, obs: np.ndarray) -> bool:
        """Precondition: robot vacuum is on (holding a block)."""
        return extract_robot(obs).vacuum >= 0.5

    def terminated(self, obs: np.ndarray) -> bool:
        """Subgoal: robot vacuum is off and arm retracted."""
        robot = extract_robot(obs)
        return (
            self._phase == _DONE
            and robot.vacuum < 0.5
            and robot.arm_joint <= ARM_MIN_JOINT + ARM_TOL
        )

    # ------------------------------------------------------------------
    def reset(self, obs: np.ndarray) -> None:
        self._phase = _ALIGN
        self._path = []
        self._path_step = 0
        self._rng = np.random.default_rng()
        # Target x = shelf inner centre x
        scx, _ = shelf_inner_center(obs)
        self._target_robot_x = scx
        self._target_robot_y = float(DEPOSIT_SLOT_Y_NAV[self._slot_idx])

    # ------------------------------------------------------------------
    def step(self, obs: np.ndarray) -> np.ndarray:
        robot = extract_robot(obs)

        if self._phase == _ALIGN:
            dth = angle_diff(THETA_UP, robot.theta)
            if abs(dth) < THETA_TOL:
                self._phase = _NAVIGATE
                self._plan_path(obs)
                return build_action(vacuum=1.0)
            dtheta = np.clip(dth, -DTH_LIM, DTH_LIM)
            return build_action(dtheta=dtheta, vacuum=1.0)

        elif self._phase == _NAVIGATE:
            if self._path_step >= len(self._path):
                self._phase = _EXTEND
                return build_action(vacuum=1.0)
            tx, ty = self._path[self._path_step]
            dx = np.clip(tx - robot.x, -DX_LIM, DX_LIM)
            dy = np.clip(ty - robot.y, -DY_LIM, DY_LIM)
            if abs(tx - robot.x) < POS_TOL and abs(ty - robot.y) < POS_TOL:
                self._path_step += 1
            return build_action(dx=dx, dy=dy, vacuum=1.0)

        elif self._phase == _EXTEND:
            # Extend arm upward into shelf
            if robot.arm_joint >= DEPOSIT_ARM_JOINT - ARM_TOL:
                self._phase = _RELEASE
                return build_action(vacuum=1.0)
            darm = min(DARM_LIM, DEPOSIT_ARM_JOINT - robot.arm_joint)
            return build_action(darm=darm, vacuum=1.0)

        elif self._phase == _RELEASE:
            # Turn off vacuum to drop block in shelf
            self._phase = _RETRACT
            return build_action(vacuum=0.0)

        elif self._phase == _RETRACT:
            if robot.arm_joint <= ARM_MIN_JOINT + ARM_TOL:
                self._phase = _DONE
                return build_action(vacuum=0.0)
            darm = -min(DARM_LIM, robot.arm_joint - ARM_MIN_JOINT)
            return build_action(darm=darm, vacuum=0.0)

        else:  # _DONE
            return build_action(vacuum=0.0)

    # ------------------------------------------------------------------
    def _plan_path(self, obs: np.ndarray) -> None:
        robot = extract_robot(obs)
        sy = shelf_y_bottom(obs)
        birrt = make_birrt(self._primitives, sy, self._rng)
        start = (robot.x, robot.y)
        goal = (self._target_robot_x, self._target_robot_y)
        path = birrt.query(start, goal)
        if path is None or len(path) < 2:
            self._path = [goal]
        else:
            self._path = list(path[1:])
        self._path_step = 0
