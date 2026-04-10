"""Behavior classes for Obstruction2D-o4-v0."""
import numpy as np
from behavior import Behavior
from obs_helpers import (
    NUM_OBSTRUCTIONS, THETA_DOWN,
    extract_robot, extract_target_surface, extract_target_block, extract_obstruction,
    is_block_on_surface, is_obstruction_on_surface, any_obstruction_on_surface,
    is_holding, gripper_tip_xy, approach_xy_for_pick, get_obstacle_rects,
)
from act_helpers import (
    THETA_DOWN, NAV_HEIGHT, DROP_Y, DROP_XS, ARM_TOL, ANGLE_TOL, POS_TOL,
    make_action, step_toward_xy, rotate_toward, extend_arm,
    plan_base_path, placement_robot_y,
)

# ─── Phases ──────────────────────────────────────────────────────────────────
_FIND           = 'FIND'
_PLAN_APPROACH  = 'PLAN_APPROACH'
_NAVIGATE_HIGH  = 'NAVIGATE_HIGH'
_NAVIGATE_DOWN  = 'NAVIGATE_DOWN'
_ROTATE         = 'ROTATE'
_EXTEND_ARM     = 'EXTEND_ARM'
_GRASP          = 'GRASP'
_RETRACT_ARM    = 'RETRACT_ARM'
_PLAN_DROP      = 'PLAN_DROP'
_NAVIGATE_DROP  = 'NAVIGATE_DROP'
_RELEASE        = 'RELEASE'
_DONE           = 'DONE'

# Additional phases for PlaceTargetBlock
_PLAN_PLACE     = 'PLAN_PLACE'
_NAVIGATE_PLACE = 'NAVIGATE_PLACE'
_LOWER          = 'LOWER'
_PLACE_EXTEND   = 'PLACE_EXTEND'
_PLACE_RELEASE  = 'PLACE_RELEASE'


class ClearAllObstructions(Behavior):
    """Pick up each obstruction on the target surface and carry it to a drop zone."""

    def __init__(self, primitives):
        self._primitives = primitives
        self._phase = _FIND
        self._target_idx = -1
        self._path = []
        self._path_step = 0
        self._drop_zone_idx = 0
        self._pick_x = 0.0
        self._pick_y = 0.0
        self._nav_x = 0.0
        self._nav_y = 0.0

    def reset(self, obs):
        self._phase = _FIND
        self._target_idx = -1
        self._path = []
        self._path_step = 0
        self._drop_zone_idx = 0

    def initializable(self, obs) -> bool:
        return any_obstruction_on_surface(obs)

    def terminated(self, obs) -> bool:
        return not any_obstruction_on_surface(obs)

    def step(self, obs) -> np.ndarray:
        robot = extract_robot(obs)

        # ── FIND: choose next obstruction on surface ──────────────────────────
        if self._phase == _FIND:
            for i in range(NUM_OBSTRUCTIONS):
                if is_obstruction_on_surface(obs, i):
                    self._target_idx = i
                    ob = extract_obstruction(obs, i)
                    self._pick_x, self._pick_y = approach_xy_for_pick(ob, robot['arm_length'])
                    # Navigate high first, then down
                    self._nav_x = self._pick_x
                    self._nav_y = NAV_HEIGHT
                    self._phase = _PLAN_APPROACH
                    break
            else:
                # Nothing to clear
                return make_action()

        # ── PLAN_APPROACH: plan path to high position above obstruction ───────
        if self._phase == _PLAN_APPROACH:
            obs_rects = get_obstacle_rects(obs, exclude_idx=self._target_idx)
            path = plan_base_path(obs, self._primitives,
                                  self._nav_x, self._nav_y,
                                  obstacle_rects=obs_rects)
            if path is None:
                # Fallback: move directly
                path = [np.array([self._nav_x, self._nav_y])]
            self._path = path
            self._path_step = 0
            self._phase = _NAVIGATE_HIGH

        # ── NAVIGATE_HIGH: follow path to high position ───────────────────────
        if self._phase == _NAVIGATE_HIGH:
            if self._path_step < len(self._path):
                wp = self._path[self._path_step]
                if np.linalg.norm(np.array([robot['x'], robot['y']]) - wp) < POS_TOL:
                    self._path_step += 1
                else:
                    return step_toward_xy(robot['x'], robot['y'], wp[0], wp[1],
                                          darm=-0.1)  # retract arm during nav
            if self._path_step >= len(self._path):
                self._phase = _NAVIGATE_DOWN

        # ── NAVIGATE_DOWN: lower to pick position ────────────────────────────
        if self._phase == _NAVIGATE_DOWN:
            dx = np.clip(self._pick_x - robot['x'], -0.05, 0.05)
            dy = np.clip(self._pick_y - robot['y'], -0.05, 0.05)
            at_pick = (abs(robot['x'] - self._pick_x) < POS_TOL and
                       abs(robot['y'] - self._pick_y) < POS_TOL)
            if not at_pick:
                return make_action(dx=dx, dy=dy, darm=-0.1)
            self._phase = _ROTATE

        # ── ROTATE: face downward (theta = -pi/2) ────────────────────────────
        if self._phase == _ROTATE:
            if abs(robot['theta'] - THETA_DOWN) > ANGLE_TOL:
                return rotate_toward(robot['theta'], THETA_DOWN)
            self._phase = _EXTEND_ARM

        # ── EXTEND_ARM: extend arm to arm_length ─────────────────────────────
        if self._phase == _EXTEND_ARM:
            if robot['arm_joint'] < robot['arm_length'] - ARM_TOL:
                return extend_arm(robot['arm_joint'], robot['arm_length'], vacuum=1.0)
            self._phase = _GRASP

        # ── GRASP: turn vacuum on (a few steps to ensure contact) ────────────
        if self._phase == _GRASP:
            self._phase = _RETRACT_ARM
            return make_action(vac=1.0)

        # ── RETRACT_ARM: retract arm while holding ────────────────────────────
        if self._phase == _RETRACT_ARM:
            if robot['arm_joint'] > robot['base_radius'] + ARM_TOL:
                return extend_arm(robot['arm_joint'], robot['base_radius'], vacuum=1.0)
            # Plan route to drop zone
            drop_x = DROP_XS[self._drop_zone_idx % len(DROP_XS)]
            self._drop_zone_idx += 1
            self._nav_x = drop_x
            self._nav_y = DROP_Y
            self._phase = _PLAN_DROP

        # ── PLAN_DROP: plan path to drop zone ────────────────────────────────
        if self._phase == _PLAN_DROP:
            # No obstacle rects during carry (block is with us, table is static)
            path = plan_base_path(obs, self._primitives,
                                  self._nav_x, self._nav_y,
                                  obstacle_rects=[])  # nothing blocks us
            if path is None:
                path = [np.array([self._nav_x, self._nav_y])]
            self._path = path
            self._path_step = 0
            self._phase = _NAVIGATE_DROP

        # ── NAVIGATE_DROP: carry obstruction to drop zone ─────────────────────
        if self._phase == _NAVIGATE_DROP:
            if self._path_step < len(self._path):
                wp = self._path[self._path_step]
                if np.linalg.norm(np.array([robot['x'], robot['y']]) - wp) < POS_TOL:
                    self._path_step += 1
                else:
                    return step_toward_xy(robot['x'], robot['y'],
                                          wp[0], wp[1], vacuum=1.0)
            if self._path_step >= len(self._path):
                self._phase = _RELEASE

        # ── RELEASE: turn off vacuum ──────────────────────────────────────────
        if self._phase == _RELEASE:
            self._phase = _FIND  # go pick up next obstruction
            return make_action(vac=0.0)

        # Default: idle
        return make_action(vac=0.0)


class PickupTargetBlock(Behavior):
    """Navigate to target block and grasp it with vacuum."""

    def __init__(self, primitives):
        self._primitives = primitives
        self._phase = _PLAN_APPROACH
        self._path = []
        self._path_step = 0
        self._pick_x = 0.0
        self._pick_y = 0.0

    def reset(self, obs):
        self._phase = _PLAN_APPROACH
        self._path = []
        self._path_step = 0
        robot = extract_robot(obs)
        block = extract_target_block(obs)
        self._pick_x, self._pick_y = approach_xy_for_pick(block, robot['arm_length'])

    def initializable(self, obs) -> bool:
        return not any_obstruction_on_surface(obs) and not is_holding(obs)

    def terminated(self, obs) -> bool:
        return is_holding(obs)

    def step(self, obs) -> np.ndarray:
        robot = extract_robot(obs)

        # ── PLAN_APPROACH ─────────────────────────────────────────────────────
        if self._phase == _PLAN_APPROACH:
            # Navigate to high position above block
            nav_x, nav_y = self._pick_x, NAV_HEIGHT
            obs_rects = get_obstacle_rects(obs, exclude_idx=-1)
            path = plan_base_path(obs, self._primitives, nav_x, nav_y,
                                  obstacle_rects=obs_rects)
            if path is None:
                path = [np.array([nav_x, nav_y])]
            # Append the lower pick position
            path.append(np.array([self._pick_x, self._pick_y]))
            self._path = path
            self._path_step = 0
            self._phase = _NAVIGATE_HIGH

        # ── NAVIGATE_HIGH ─────────────────────────────────────────────────────
        if self._phase == _NAVIGATE_HIGH:
            if self._path_step < len(self._path):
                wp = self._path[self._path_step]
                dist = np.linalg.norm(np.array([robot['x'], robot['y']]) - wp)
                if dist < POS_TOL:
                    self._path_step += 1
                else:
                    return step_toward_xy(robot['x'], robot['y'],
                                          wp[0], wp[1], darm=-0.1)
            if self._path_step >= len(self._path):
                self._phase = _ROTATE

        # ── ROTATE ────────────────────────────────────────────────────────────
        if self._phase == _ROTATE:
            if abs(robot['theta'] - THETA_DOWN) > ANGLE_TOL:
                return rotate_toward(robot['theta'], THETA_DOWN)
            self._phase = _EXTEND_ARM

        # ── EXTEND_ARM ────────────────────────────────────────────────────────
        if self._phase == _EXTEND_ARM:
            if robot['arm_joint'] < robot['arm_length'] - ARM_TOL:
                return extend_arm(robot['arm_joint'], robot['arm_length'], vacuum=1.0)
            self._phase = _GRASP

        # ── GRASP ─────────────────────────────────────────────────────────────
        if self._phase == _GRASP:
            return make_action(vac=1.0)

        return make_action(vac=1.0)


class PlaceTargetBlock(Behavior):
    """Carry target block to target surface and release."""

    def __init__(self, primitives):
        self._primitives = primitives
        self._phase = _PLAN_PLACE
        self._path = []
        self._path_step = 0
        self._place_x = 0.0
        self._place_y = 0.0

    def reset(self, obs):
        self._phase = _PLAN_PLACE
        self._path = []
        self._path_step = 0
        robot = extract_robot(obs)
        surf  = extract_target_surface(obs)
        block = extract_target_block(obs)
        self._place_x = surf['x']
        self._place_y = placement_robot_y(
            block['height'], robot['arm_length'],
            surf['y'], surf['height'])

    def initializable(self, obs) -> bool:
        # Can start if we're holding something (target block)
        return is_holding(obs)

    def terminated(self, obs) -> bool:
        return is_block_on_surface(obs)

    def step(self, obs) -> np.ndarray:
        robot = extract_robot(obs)

        # ── PLAN_PLACE ────────────────────────────────────────────────────────
        if self._phase == _PLAN_PLACE:
            surf  = extract_target_surface(obs)
            block = extract_target_block(obs)
            self._place_x = surf['x']
            self._place_y = placement_robot_y(
                block['height'], robot['arm_length'],
                surf['y'], surf['height'])
            # Navigate carrying block at high altitude, then to placement position
            # Navigate to (place_x, NAV_HEIGHT) first
            nav_x, nav_y = self._place_x, NAV_HEIGHT
            path = plan_base_path(obs, self._primitives, nav_x, nav_y,
                                  obstacle_rects=[],  # carrying, no obstacles
                                  )
            if path is None:
                path = [np.array([nav_x, nav_y])]
            # Then add final placement position
            path.append(np.array([self._place_x, self._place_y]))
            self._path = path
            self._path_step = 0
            self._phase = _NAVIGATE_PLACE

        # ── NAVIGATE_PLACE ────────────────────────────────────────────────────
        if self._phase == _NAVIGATE_PLACE:
            if self._path_step < len(self._path):
                wp = self._path[self._path_step]
                dist = np.linalg.norm(np.array([robot['x'], robot['y']]) - wp)
                if dist < POS_TOL:
                    self._path_step += 1
                else:
                    return step_toward_xy(robot['x'], robot['y'],
                                          wp[0], wp[1], vacuum=1.0)
            if self._path_step >= len(self._path):
                self._phase = _ROTATE

        # ── ROTATE ────────────────────────────────────────────────────────────
        if self._phase == _ROTATE:
            if abs(robot['theta'] - THETA_DOWN) > ANGLE_TOL:
                return rotate_toward(robot['theta'], THETA_DOWN, vacuum=1.0)
            self._phase = _PLACE_EXTEND

        # ── PLACE_EXTEND: extend arm fully ────────────────────────────────────
        if self._phase == _PLACE_EXTEND:
            if robot['arm_joint'] < robot['arm_length'] - ARM_TOL:
                return extend_arm(robot['arm_joint'], robot['arm_length'], vacuum=1.0)
            self._phase = _PLACE_RELEASE

        # ── PLACE_RELEASE: release block ──────────────────────────────────────
        if self._phase == _PLACE_RELEASE:
            return make_action(vac=0.0)

        return make_action(vac=0.0)
