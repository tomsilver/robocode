"""Behavior classes for ClutteredStorage2D-b3-v0."""

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
    block_center,
    block_center_from_obs,
    extract_rect,
    extract_robot,
    get_outside_blocks,
    gripper_pos,
    is_block_in_shelf,
    shelf_inner_center,
    shelf_inner_bounds,
    shelf_y_bottom,
)
from act_helpers import (
    APPROACH_DIST,
    ARM_TOL,
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

# When navigating while holding a suctioned block (arm retracted, pointing up),
# the block center is ~0.25 above the robot center and has ~0.02 half-height.
# To avoid block-shelf collision (outer shelf at y=2.625), cap robot y at:
#   2.625 - 0.25 - 0.02 - 0.08 (safety) ≈ 2.27
HOLDING_NAV_Y_CEILING = 2.25

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
# Deposit geometry constants
# ---------------------------------------------------------------------------
DEPOSIT_Y_NAV = 2.025          # robot y for all deposits (below shelf)
DEPOSIT_ARM_SLOT = [0.755, 0.665]  # arm_joint for slot 0 (highest) and slot 1 (middle)
HOLDINSHELF_ARM_TARGET = 0.650     # extend until block enters shelf, then terminated fires
TEMP_DROP_Y_NAV = 0.40         # robot y for temp drop (low so block clears all obstacles)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_pick_angle(block_theta: float) -> float:
    """Return arm angle α so block becomes horizontal (theta≈π) when arm is at π/2.

    After picking at α, rotating arm to π/2 adds (π/2 − α) to block theta.
    Target: block_theta + (π/2 − α) = k·π → α = block_theta + π/2 − k·π.
    Choose k so that sin(α) > 0 (arm points into upper half-plane).
    """
    def _norm(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    alpha_k0 = _norm(block_theta + math.pi / 2)   # k=0
    alpha_k1 = _norm(block_theta - math.pi / 2)   # k=1

    if math.sin(alpha_k0) > 0:
        return alpha_k0
    return alpha_k1


# ---------------------------------------------------------------------------
# PickBlockBehavior
# ---------------------------------------------------------------------------

class PickBlockBehavior(Behavior):
    """Navigate to a block (with auto-reorienting pick angle), grasp, retract.

    If pick_angle is None, auto-computes an angle that reorients the block to
    horizontal (theta≈π) when the arm is later rotated to π/2 for deposit.

    Precondition:  target block is NOT held (vacuum off) — or block is anywhere
    Subgoal:       robot holds the block (vacuum on, arm retracted)
    """

    def __init__(self, block_name: str, primitives: dict,
                 pick_angle: float | None = None) -> None:
        self._block_name = block_name
        self._primitives = primitives
        self._pick_angle_override = pick_angle
        self._phase: str = _ALIGN
        self._align_target: float = THETA_UP
        self._path: list[tuple[float, float]] = []
        self._path_step: int = 0
        self._arm_target: float = APPROACH_DIST
        self._rng: np.random.Generator = np.random.default_rng()

    def initializable(self, obs: np.ndarray) -> bool:
        return True  # can always try to pick

    def terminated(self, obs: np.ndarray) -> bool:
        robot = extract_robot(obs)
        return (
            self._phase == _DONE
            and robot.arm_joint <= ARM_MIN_JOINT + ARM_TOL
            and robot.vacuum >= 0.5
        )

    def reset(self, obs: np.ndarray) -> None:
        self._phase = _ALIGN
        self._path = []
        self._path_step = 0
        self._rng = np.random.default_rng()

        rect = extract_rect(obs, self._block_name)
        bcx, bcy = block_center(rect)

        # Compute pick angle
        if self._pick_angle_override is not None:
            alpha = self._pick_angle_override
        else:
            alpha = compute_pick_angle(rect.theta)
        self._align_target = alpha

        # Approach position: APPROACH_DIST behind block in direction alpha
        raw_x = bcx - APPROACH_DIST * math.cos(alpha)
        raw_y = bcy - APPROACH_DIST * math.sin(alpha)

        y_nav_max = shelf_y_bottom(obs) - ROBOT_RADIUS - 0.05
        self._target_robot_x = float(np.clip(raw_x, ROBOT_RADIUS + 0.05,
                                              WORLD_MAX_X - ROBOT_RADIUS - 0.05))
        self._target_robot_y = float(np.clip(raw_y, ROBOT_RADIUS + 0.05, y_nav_max))

        # Arm extension = actual distance from clamped robot pos to block center
        self._arm_target = min(
            math.sqrt((bcx - self._target_robot_x) ** 2 +
                      (bcy - self._target_robot_y) ** 2),
            ARM_MAX_JOINT - 0.05,
        )

    def step(self, obs: np.ndarray) -> np.ndarray:
        robot = extract_robot(obs)

        if self._phase == _ALIGN:
            dth = angle_diff(self._align_target, robot.theta)
            if abs(dth) < THETA_TOL:
                self._phase = _NAVIGATE
                self._plan_path(obs)
                return zero_action()
            return build_action(dtheta=np.clip(dth, -DTH_LIM, DTH_LIM), vacuum=0.0)

        elif self._phase == _NAVIGATE:
            if self._path_step >= len(self._path):
                self._phase = _EXTEND
                return zero_action()
            tx, ty = self._path[self._path_step]
            dx = np.clip(tx - robot.x, -DX_LIM, DX_LIM)
            dy = np.clip(ty - robot.y, -DY_LIM, DY_LIM)
            if abs(tx - robot.x) < POS_TOL and abs(ty - robot.y) < POS_TOL:
                self._path_step += 1
            return build_action(dx=dx, dy=dy, vacuum=0.0)

        elif self._phase == _EXTEND:
            if robot.arm_joint >= self._arm_target - ARM_TOL:
                self._phase = _RETRACT
                return build_action(vacuum=1.0)
            darm = min(EXTEND_DARM, self._arm_target - robot.arm_joint)
            return build_action(darm=darm, vacuum=1.0)

        elif self._phase == _RETRACT:
            if robot.arm_joint <= ARM_MIN_JOINT + ARM_TOL:
                self._phase = _DONE
                return build_action(vacuum=1.0)
            darm = -min(DARM_LIM, robot.arm_joint - ARM_MIN_JOINT)
            return build_action(darm=darm, vacuum=1.0)

        else:  # _DONE
            return build_action(vacuum=1.0)

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


# ---------------------------------------------------------------------------
# TempDropBehavior
# ---------------------------------------------------------------------------

class TempDropBehavior(Behavior):
    """Navigate to a safe temp location and release the held block.

    Precondition:  robot holds a block (vacuum on)
    Subgoal:       block released at temp location (vacuum off)
    """

    def __init__(self, primitives: dict) -> None:
        self._primitives = primitives
        self._phase: str = _ALIGN
        self._path: list[tuple[float, float]] = []
        self._path_step: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

    def initializable(self, obs: np.ndarray) -> bool:
        return extract_robot(obs).vacuum >= 0.5

    def terminated(self, obs: np.ndarray) -> bool:
        robot = extract_robot(obs)
        return self._phase == _DONE and robot.vacuum < 0.5

    def reset(self, obs: np.ndarray) -> None:
        self._phase = _ALIGN
        self._path = []
        self._path_step = 0
        self._rng = np.random.default_rng()

        # Choose temp x far from shelf
        sx_min, sx_max, _, _ = shelf_inner_bounds(obs)
        shelf_cx = (sx_min + sx_max) / 2.0
        if shelf_cx < 2.5:
            temp_x = min(4.5, WORLD_MAX_X - ROBOT_RADIUS - 0.1)
        else:
            temp_x = max(0.5, WORLD_MIN_X + ROBOT_RADIUS + 0.1)
        self._target_robot_x = float(np.clip(temp_x, ROBOT_RADIUS + 0.05,
                                              WORLD_MAX_X - ROBOT_RADIUS - 0.05))
        self._target_robot_y = TEMP_DROP_Y_NAV
        self._start_robot_x: float = 0.0  # filled in _plan_path

    def step(self, obs: np.ndarray) -> np.ndarray:
        robot = extract_robot(obs)

        if self._phase == _ALIGN:
            dth = angle_diff(THETA_UP, robot.theta)
            if abs(dth) < THETA_TOL:
                self._phase = _NAVIGATE
                self._plan_path(obs)
                return build_action(vacuum=1.0)
            return build_action(dtheta=np.clip(dth, -DTH_LIM, DTH_LIM), vacuum=1.0)

        elif self._phase == _NAVIGATE:
            if self._path_step >= len(self._path):
                self._phase = _RELEASE
                return build_action(vacuum=1.0)
            tx, ty = self._path[self._path_step]
            dx = np.clip(tx - robot.x, -DX_LIM, DX_LIM)
            dy = np.clip(ty - robot.y, -DY_LIM, DY_LIM)
            if abs(tx - robot.x) < POS_TOL and abs(ty - robot.y) < POS_TOL:
                self._path_step += 1
            return build_action(dx=dx, dy=dy, vacuum=1.0)

        elif self._phase == _RELEASE:
            self._phase = _DONE
            return build_action(vacuum=0.0)

        else:  # _DONE
            return build_action(vacuum=0.0)

    def _plan_path(self, obs: np.ndarray) -> None:
        """Use simple 2-waypoint path: descend first, then move sideways.

        This avoids BiRRT routing near the y_ceiling where the suctioned block
        could collide with the outer shelf floor (y=2.625).
        """
        robot = extract_robot(obs)
        # Step 1: descend to target y (staying at current x)
        # Step 2: move sideways to target x
        self._path = [
            (robot.x, self._target_robot_y),
            (self._target_robot_x, self._target_robot_y),
        ]
        self._path_step = 0


# ---------------------------------------------------------------------------
# PlaceBlockBehavior
# ---------------------------------------------------------------------------

class PlaceBlockBehavior(Behavior):
    """Navigate below shelf and deposit held block at specified arm height.

    Precondition:  robot holds a block (vacuum on)
    Subgoal:       block deposited in shelf, arm retracted, vacuum off
    """

    def __init__(self, primitives: dict,
                 deposit_arm_joint: float = 0.72) -> None:
        self._primitives = primitives
        self._deposit_arm_joint = deposit_arm_joint
        self._phase: str = _ALIGN
        self._path: list[tuple[float, float]] = []
        self._path_step: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

    def initializable(self, obs: np.ndarray) -> bool:
        return extract_robot(obs).vacuum >= 0.5

    def terminated(self, obs: np.ndarray) -> bool:
        robot = extract_robot(obs)
        return (
            self._phase == _DONE
            and robot.vacuum < 0.5
            and robot.arm_joint <= ARM_MIN_JOINT + ARM_TOL
        )

    def reset(self, obs: np.ndarray) -> None:
        self._phase = _ALIGN
        self._path = []
        self._path_step = 0
        self._rng = np.random.default_rng()

        sx_min, sx_max, _, _ = shelf_inner_bounds(obs)
        shelf_cx = (sx_min + sx_max) / 2.0
        self._target_robot_x = float(np.clip(shelf_cx,
                                              ROBOT_RADIUS + 0.05,
                                              WORLD_MAX_X - ROBOT_RADIUS - 0.05))
        self._target_robot_y = DEPOSIT_Y_NAV

    def step(self, obs: np.ndarray) -> np.ndarray:
        robot = extract_robot(obs)

        if self._phase == _ALIGN:
            dth = angle_diff(THETA_UP, robot.theta)
            if abs(dth) < THETA_TOL:
                self._phase = _NAVIGATE
                self._plan_path(obs)
                return build_action(vacuum=1.0)
            return build_action(dtheta=np.clip(dth, -DTH_LIM, DTH_LIM), vacuum=1.0)

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
            if robot.arm_joint >= self._deposit_arm_joint - ARM_TOL:
                self._phase = _RELEASE
                return build_action(vacuum=1.0)
            darm = min(DARM_LIM, self._deposit_arm_joint - robot.arm_joint)
            return build_action(darm=darm, vacuum=1.0)

        elif self._phase == _RELEASE:
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

    def _plan_path(self, obs: np.ndarray) -> None:
        robot = extract_robot(obs)
        sy = shelf_y_bottom(obs)
        # Use reduced y_ceiling: suctioned block must not hit outer shelf floor
        birrt = make_birrt(self._primitives, sy, self._rng,
                           y_ceiling_override=HOLDING_NAV_Y_CEILING)
        path = birrt.query((robot.x, robot.y),
                           (self._target_robot_x, self._target_robot_y))
        if path is None or len(path) < 2:
            self._path = [(self._target_robot_x, self._target_robot_y)]
        else:
            self._path = list(path[1:])
        self._path_step = 0


# ---------------------------------------------------------------------------
# HoldInShelfBehavior
# ---------------------------------------------------------------------------

class HoldInShelfBehavior(Behavior):
    """Navigate below shelf and slowly extend arm until block enters shelf.

    The environment checks terminated after every step; once all blocks are
    inside the shelf inner region the episode ends.

    Precondition:  robot holds a block (vacuum on)
    Subgoal:       env fires terminated (all blocks in shelf)
    """

    def __init__(self, primitives: dict) -> None:
        self._primitives = primitives
        self._phase: str = _ALIGN
        self._path: list[tuple[float, float]] = []
        self._path_step: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

    def initializable(self, obs: np.ndarray) -> bool:
        return extract_robot(obs).vacuum >= 0.5

    def terminated(self, obs: np.ndarray) -> bool:
        # Env fires terminated; we just mark _DONE when arm reaches target.
        robot = extract_robot(obs)
        return self._phase == _DONE and robot.arm_joint >= HOLDINSHELF_ARM_TARGET - ARM_TOL

    def reset(self, obs: np.ndarray) -> None:
        self._phase = _ALIGN
        self._path = []
        self._path_step = 0
        self._rng = np.random.default_rng()

        sx_min, sx_max, _, _ = shelf_inner_bounds(obs)
        shelf_cx = (sx_min + sx_max) / 2.0
        self._target_robot_x = float(np.clip(shelf_cx,
                                              ROBOT_RADIUS + 0.05,
                                              WORLD_MAX_X - ROBOT_RADIUS - 0.05))
        self._target_robot_y = DEPOSIT_Y_NAV

    def step(self, obs: np.ndarray) -> np.ndarray:
        robot = extract_robot(obs)

        if self._phase == _ALIGN:
            dth = angle_diff(THETA_UP, robot.theta)
            if abs(dth) < THETA_TOL:
                self._phase = _NAVIGATE
                self._plan_path(obs)
                return build_action(vacuum=1.0)
            return build_action(dtheta=np.clip(dth, -DTH_LIM, DTH_LIM), vacuum=1.0)

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
            # Use small steps (EXTEND_DARM) to avoid jumping over the
            # no-collision window between block0 and block2.
            if robot.arm_joint >= HOLDINSHELF_ARM_TARGET - ARM_TOL:
                self._phase = _DONE
                return build_action(vacuum=1.0)
            darm = min(EXTEND_DARM, HOLDINSHELF_ARM_TARGET - robot.arm_joint)
            return build_action(darm=darm, vacuum=1.0)

        else:  # _DONE
            return build_action(vacuum=1.0)

    def _plan_path(self, obs: np.ndarray) -> None:
        robot = extract_robot(obs)
        sy = shelf_y_bottom(obs)
        # Use reduced y_ceiling: suctioned block must not hit outer shelf floor
        birrt = make_birrt(self._primitives, sy, self._rng,
                           y_ceiling_override=HOLDING_NAV_Y_CEILING)
        path = birrt.query((robot.x, robot.y),
                           (self._target_robot_x, self._target_robot_y))
        if path is None or len(path) < 2:
            self._path = [(self._target_robot_x, self._target_robot_y)]
        else:
            self._path = list(path[1:])
        self._path_step = 0
