"""Oracle behaviors for Obstruction2D-o2 (medium, 2 obstructions).

Three sequential behaviors that solve the task:
  RemoveObstruction  -> GoalRegionClear
  PickTargetBlock    -> HoldingTarget
  PlaceTargetBlock   -> GoalAchieved

Observation layout (49 features):
  Robot          [0:9]   x y theta base_r arm_j arm_l vac grip_h grip_w
  Target surface [9:19]  x y theta static cr cg cb z w h
  Target block   [19:29] x y theta static cr cg cb z w h
  Obstruction 0  [29:39] x y theta static cr cg cb z w h
  Obstruction 1  [39:49] x y theta static cr cg cb z w h

Position convention: (x, y) is the bottom-left corner of each rectangle.
"""

from __future__ import annotations

import enum
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from robocode.utils.structs import Behavior

# ---------------------------------------------------------------------------
# Observation indices
# ---------------------------------------------------------------------------
R_X, R_Y, R_THETA = 0, 1, 2
R_BASE_R, R_ARM_J, R_ARM_L = 3, 4, 5
R_VAC, R_GRIP_H, R_GRIP_W = 6, 7, 8

_X, _Y, _THETA = 0, 1, 2  # offsets within each 10-feature rect block
_W, _H = 8, 9

SURF = 9
BLOCK = 19
OBS_BASE = 29
OBS_STRIDE = 10

# Physics
TABLE_TOP = 0.1
DOWN = -np.pi / 2
GRIP_OFFSET = 0.015  # gripper_width + suction_width / 2

# Proportional-control tolerances and limits
POS_TOL = 0.008
ANG_TOL = 0.05
ARM_TOL = 0.008
DX_LIM = 0.05
DY_LIM = 0.05
DTH_LIM = np.pi / 16
DARM_LIM = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rect(obs: NDArray, base: int) -> tuple[float, float, float, float]:
    """(x, y, w, h) of a rectangle object."""
    return float(obs[base + _X]), float(obs[base + _Y]), float(obs[base + _W]), float(obs[base + _H])


def _center_x(obs: NDArray, base: int) -> float:
    return float(obs[base + _X] + obs[base + _W] / 2)


def _center_y(obs: NDArray, base: int) -> float:
    return float(obs[base + _Y] + obs[base + _H] / 2)


def _overlaps_surface_h(obs: NDArray, obj_base: int) -> bool:
    """True if a rectangle overlaps the target surface horizontally and sits
    at the table level."""
    ox, oy, ow, _ = _rect(obs, obj_base)
    sx, sy, sw, sh = _rect(obs, SURF)
    h_overlap = min(ox + ow, sx + sw) - max(ox, sx)
    at_table = abs(oy - (sy + sh)) < 0.05
    return h_overlap > 1e-4 and at_table


def _is_on_surface(obs: NDArray, obj_base: int) -> bool:
    """Simplified ``is_on`` for axis-aligned rectangles.

    The 2 bottom vertices of *obj*, offset down by ``tol``, must lie inside the
    target surface rectangle.
    """
    ox, oy, ow, _ = _rect(obs, obj_base)
    sx, sy, sw, sh = _rect(obs, SURF)
    tol = 0.025
    py = oy - tol
    return (ox >= sx - 1e-4 and ox + ow <= sx + sw + 1e-4
            and py >= sy - 1e-4 and py <= sy + sh + 1e-4)


def _angle_diff(target: float, current: float) -> float:
    d = target - current
    return (d + np.pi) % (2 * np.pi) - np.pi


def _make_action(
    obs: NDArray,
    target_x: float,
    target_y: float,
    target_theta: float,
    target_arm: float,
    target_vac: float,
) -> NDArray:
    """Proportional controller toward a target configuration."""
    dx = np.clip(target_x - obs[R_X], -DX_LIM, DX_LIM)
    dy = np.clip(target_y - obs[R_Y], -DY_LIM, DY_LIM)
    dth = np.clip(_angle_diff(target_theta, obs[R_THETA]), -DTH_LIM, DTH_LIM)
    darm = np.clip(target_arm - obs[R_ARM_J], -DARM_LIM, DARM_LIM)
    return np.array([dx, dy, dth, darm, target_vac], dtype=np.float32)


def _at_pos(obs: NDArray, tx: float, ty: float) -> bool:
    return abs(obs[R_X] - tx) < POS_TOL and abs(obs[R_Y] - ty) < POS_TOL


def _at_angle(obs: NDArray, tth: float) -> bool:
    return abs(_angle_diff(tth, obs[R_THETA])) < ANG_TOL


def _at_arm(obs: NDArray, tarm: float) -> bool:
    return abs(obs[R_ARM_J] - tarm) < ARM_TOL


def _holding_block(obs: NDArray) -> bool:
    """Vacuum on and target block lifted off the table."""
    return obs[R_VAC] > 0.5 and obs[BLOCK + _Y] > TABLE_TOP + 0.04


def _pickup_robot_y(obs: NDArray, obj_base: int) -> float:
    """Robot *y* that places the suction zone at the vertical centre of *obj*
    when the arm is fully extended."""
    arm_l = float(obs[R_ARM_L])
    obj_cy = _center_y(obs, obj_base)
    return obj_cy + arm_l + GRIP_OFFSET


def _dump_x(obs: NDArray) -> float:
    """X position far from the target surface to dump an obstruction."""
    surf_cx = _center_x(obs, SURF)
    return 0.2 if surf_cx > 0.809 else 1.4


# ---------------------------------------------------------------------------
# Internal phase enum
# ---------------------------------------------------------------------------

class _Phase(enum.Enum):
    FIND_TARGET = 0
    APPROACH = 1
    DESCEND = 2
    GRAB = 3
    LIFT = 4
    TRAVEL = 5
    LOWER = 6
    RELEASE = 7


# ---------------------------------------------------------------------------
# RemoveObstruction
# ---------------------------------------------------------------------------

class RemoveObstruction(Behavior[NDArray, NDArray]):
    """Clears all obstructions from the target surface region.

    Subgoal  (GoalRegionClear): no obstruction overlaps the surface.
    Precond: at least one obstruction overlaps the surface.
    """

    def __init__(self, num_obstructions: int = 2) -> None:
        self._num_obs = num_obstructions
        self.subgoal: Callable[[NDArray], bool] = self._goal_region_clear
        self.precondition: Callable[[NDArray], bool] = self._has_obstruction
        self.policy: Callable[[NDArray], NDArray] = self.step

    def reset(self, x: NDArray) -> None:
        self._phase = _Phase.FIND_TARGET
        self._target_base: int = -1
        self._grab_counter = 0

    def initializable(self, x: NDArray) -> bool:
        return self._has_obstruction(x)

    def terminated(self, x: NDArray) -> bool:
        return self._goal_region_clear(x)

    def step(self, x: NDArray) -> NDArray:
        arm_l = float(x[R_ARM_L])
        base_r = float(x[R_BASE_R])

        if self._phase == _Phase.FIND_TARGET:
            self._target_base = self._next_obstructing(x)
            if self._target_base < 0:
                return _make_action(x, x[R_X], x[R_Y], DOWN, base_r, 0.0)
            self._phase = _Phase.APPROACH

        tx = _center_x(x, self._target_base)
        ty = _pickup_robot_y(x, self._target_base)

        if self._phase == _Phase.APPROACH:
            if _at_pos(x, tx, ty) and _at_angle(x, DOWN) and _at_arm(x, base_r):
                self._phase = _Phase.DESCEND
            return _make_action(x, tx, ty, DOWN, base_r, 0.0)

        if self._phase == _Phase.DESCEND:
            if _at_arm(x, arm_l):
                self._phase = _Phase.GRAB
                self._grab_counter = 0
            return _make_action(x, tx, ty, DOWN, arm_l, 0.0)

        if self._phase == _Phase.GRAB:
            self._grab_counter += 1
            if self._grab_counter >= 2:
                self._phase = _Phase.LIFT
            return _make_action(x, tx, ty, DOWN, arm_l, 1.0)

        if self._phase == _Phase.LIFT:
            if _at_arm(x, base_r):
                self._phase = _Phase.TRAVEL
            return _make_action(x, tx, ty, DOWN, base_r, 1.0)

        dump = _dump_x(x)
        if self._phase == _Phase.TRAVEL:
            if _at_pos(x, dump, ty):
                self._phase = _Phase.LOWER
            return _make_action(x, dump, ty, DOWN, base_r, 1.0)

        if self._phase == _Phase.LOWER:
            if _at_arm(x, arm_l):
                self._phase = _Phase.RELEASE
            return _make_action(x, dump, ty, DOWN, arm_l, 1.0)

        if self._phase == _Phase.RELEASE:
            self._phase = _Phase.FIND_TARGET
            return _make_action(x, dump, ty, DOWN, arm_l, 0.0)

        return _make_action(x, x[R_X], x[R_Y], DOWN, base_r, 0.0)

    # -- predicates ----------------------------------------------------------

    def _goal_region_clear(self, x: NDArray) -> bool:
        for i in range(self._num_obs):
            if _overlaps_surface_h(x, OBS_BASE + i * OBS_STRIDE):
                return False
        return True

    def _has_obstruction(self, x: NDArray) -> bool:
        return not self._goal_region_clear(x)

    def _next_obstructing(self, x: NDArray) -> int:
        for i in range(self._num_obs):
            base = OBS_BASE + i * OBS_STRIDE
            if _overlaps_surface_h(x, base):
                return base
        return -1


# ---------------------------------------------------------------------------
# PickTargetBlock
# ---------------------------------------------------------------------------

class PickTargetBlock(Behavior[NDArray, NDArray]):
    """Pick up the target block.

    Subgoal  (HoldingTarget): vacuum on and block lifted off the table.
    Precond: goal region is clear and robot is not already holding the block.
    """

    def __init__(self, num_obstructions: int = 2) -> None:
        self._num_obs = num_obstructions
        self.subgoal: Callable[[NDArray], bool] = _holding_block
        self.precondition: Callable[[NDArray], bool] = self._precond
        self.policy: Callable[[NDArray], NDArray] = self.step

    def reset(self, x: NDArray) -> None:
        self._phase = _Phase.APPROACH
        self._grab_counter = 0

    def initializable(self, x: NDArray) -> bool:
        return self._precond(x)

    def terminated(self, x: NDArray) -> bool:
        return _holding_block(x)

    def step(self, x: NDArray) -> NDArray:
        arm_l = float(x[R_ARM_L])
        base_r = float(x[R_BASE_R])
        tx = _center_x(x, BLOCK)
        ty = _pickup_robot_y(x, BLOCK)

        if self._phase == _Phase.APPROACH:
            if _at_pos(x, tx, ty) and _at_angle(x, DOWN) and _at_arm(x, base_r):
                self._phase = _Phase.DESCEND
            return _make_action(x, tx, ty, DOWN, base_r, 0.0)

        if self._phase == _Phase.DESCEND:
            if _at_arm(x, arm_l):
                self._phase = _Phase.GRAB
                self._grab_counter = 0
            return _make_action(x, tx, ty, DOWN, arm_l, 0.0)

        if self._phase == _Phase.GRAB:
            self._grab_counter += 1
            if self._grab_counter >= 2:
                self._phase = _Phase.LIFT
            return _make_action(x, tx, ty, DOWN, arm_l, 1.0)

        if self._phase == _Phase.LIFT:
            return _make_action(x, tx, ty, DOWN, base_r, 1.0)

        return _make_action(x, x[R_X], x[R_Y], DOWN, base_r, 0.0)

    def _precond(self, x: NDArray) -> bool:
        for i in range(self._num_obs):
            if _overlaps_surface_h(x, OBS_BASE + i * OBS_STRIDE):
                return False
        return not _holding_block(x)


# ---------------------------------------------------------------------------
# PlaceTargetBlock
# ---------------------------------------------------------------------------

class PlaceTargetBlock(Behavior[NDArray, NDArray]):
    """Place the held target block onto the target surface.

    Subgoal  (GoalAchieved): block satisfies ``is_on`` w.r.t. the surface.
    Precond: robot is holding the target block.
    """

    def __init__(self) -> None:
        self.subgoal: Callable[[NDArray], bool] = lambda x: _is_on_surface(x, BLOCK)
        self.precondition: Callable[[NDArray], bool] = _holding_block
        self.policy: Callable[[NDArray], NDArray] = self.step

    def reset(self, x: NDArray) -> None:
        self._phase = _Phase.TRAVEL
        # Remember the suction-to-block-center x offset (constant while held)
        self._block_x_offset = (x[BLOCK + _X] + x[BLOCK + _W] / 2) - x[R_X]

    def initializable(self, x: NDArray) -> bool:
        return _holding_block(x)

    def terminated(self, x: NDArray) -> bool:
        return _is_on_surface(x, BLOCK)

    def step(self, x: NDArray) -> NDArray:
        arm_l = float(x[R_ARM_L])
        base_r = float(x[R_BASE_R])
        surf_cx = _center_x(x, SURF)
        surf_top = float(x[SURF + _Y] + x[SURF + _H])

        # Robot x that centres the block on the surface
        target_rx = surf_cx - self._block_x_offset
        # Robot y that places block bottom at the surface top when arm extended
        block_h = float(x[BLOCK + _H])
        place_ry = surf_top + block_h + arm_l + GRIP_OFFSET

        if self._phase == _Phase.TRAVEL:
            if _at_pos(x, target_rx, place_ry) and _at_angle(x, DOWN):
                self._phase = _Phase.LOWER
            return _make_action(x, target_rx, place_ry, DOWN, base_r, 1.0)

        if self._phase == _Phase.LOWER:
            if _at_arm(x, arm_l):
                self._phase = _Phase.RELEASE
            return _make_action(x, target_rx, place_ry, DOWN, arm_l, 1.0)

        if self._phase == _Phase.RELEASE:
            return _make_action(x, target_rx, place_ry, DOWN, arm_l, 0.0)

        return _make_action(x, x[R_X], x[R_Y], DOWN, base_r, 0.0)
