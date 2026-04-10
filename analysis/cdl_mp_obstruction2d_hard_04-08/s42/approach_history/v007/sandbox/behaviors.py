"""Behavior classes for Obstruction2D-o4-v0."""
import numpy as np
from behavior import Behavior
from obs_helpers import (
    NUM_OBSTRUCTIONS,
    extract_robot, extract_target_surface, extract_target_block, extract_obstruction,
    is_block_on_surface, is_obstruction_on_surface, any_obstruction_on_surface,
    is_obstruction_blocking_surface, any_obstruction_blocking_surface,
    is_holding, gripper_tip_xy, approach_xy_for_pick, get_obstacle_rects,
)
from act_helpers import (
    THETA_DOWN, NAV_HEIGHT, ARM_TOL, ANGLE_TOL, POS_TOL,
    make_action, step_toward_xy, rotate_toward, extend_arm,
    plan_base_path, placement_robot_y, PLACE_EPSILON,
)

# ─── Internal phases ──────────────────────────────────────────────────────────
_PLAN_HIGH   = 'PLAN_HIGH'
_NAV_HIGH    = 'NAV_HIGH'
_PLAN_DOWN   = 'PLAN_DOWN'
_NAV_DOWN    = 'NAV_DOWN'
_ROTATE      = 'ROTATE'
_EXTEND      = 'EXTEND'
_GRASP       = 'GRASP'
_PLAN_UP     = 'PLAN_UP'
_NAV_UP      = 'NAV_UP'
_RELEASE             = 'RELEASE'
_FIND        = 'FIND'
_PLAN_SURF   = 'PLAN_SURF'
_NAV_SURF    = 'NAV_SURF'
_PLAN_PLACE  = 'PLAN_PLACE'
_NAV_PLACE   = 'NAV_PLACE'


def _follow_path(robot_x, robot_y, path, path_step, vacuum, darm=0.0):
    """Return (action, new_path_step, arrived) for following a path."""
    if path_step >= len(path):
        return make_action(vac=vacuum), path_step, True
    wp = path[path_step]
    dist = np.hypot(robot_x - wp[0], robot_y - wp[1])
    if dist < POS_TOL:
        path_step += 1
        if path_step >= len(path):
            return make_action(vac=vacuum, darm=darm), path_step, True
        wp = path[path_step]
    action = step_toward_xy(robot_x, robot_y, wp[0], wp[1], vacuum=vacuum, darm=darm)
    return action, path_step, False


class ClearAllObstructions(Behavior):
    """Pick each obstruction blocking the surface and release it straight above."""

    def __init__(self, primitives):
        self._primitives = primitives
        self._reset_state()

    def _reset_state(self):
        self._phase = _FIND
        self._target_idx = -1
        self._path = []
        self._path_step = 0
        self._pick_x = 0.0
        self._pick_y = 0.0

    def reset(self, obs):
        self._reset_state()

    def initializable(self, obs) -> bool:
        return any_obstruction_blocking_surface(obs)

    def terminated(self, obs) -> bool:
        return not any_obstruction_blocking_surface(obs) and not is_holding(obs)

    def step(self, obs) -> np.ndarray:
        robot = extract_robot(obs)
        rx, ry = robot['x'], robot['y']

        if self._phase == _FIND:
            # Pick TALLEST blocking obstruction first
            best_i = -1
            best_top = -1.0
            for i in range(NUM_OBSTRUCTIONS):
                if is_obstruction_blocking_surface(obs, i):
                    ob_i = extract_obstruction(obs, i)
                    ob_top = ob_i['y'] + ob_i['height']
                    if ob_top > best_top:
                        best_top = ob_top
                        best_i = i
            if best_i >= 0:
                self._target_idx = best_i
                ob = extract_obstruction(obs, best_i)
                self._pick_x, self._pick_y = approach_xy_for_pick(ob, robot['arm_length'])
                self._phase = _PLAN_HIGH
            else:
                return make_action()

        if self._phase == _PLAN_HIGH:
            obs_rects = get_obstacle_rects(obs, exclude_idx=self._target_idx)
            path = plan_base_path(obs, self._primitives, self._pick_x, NAV_HEIGHT,
                                  obstacle_rects=obs_rects)
            self._path = path or [np.array([self._pick_x, NAV_HEIGHT])]
            self._path_step = 0
            self._phase = _NAV_HIGH
            return make_action(darm=-0.1)

        if self._phase == _NAV_HIGH:
            action, self._path_step, arrived = _follow_path(rx, ry, self._path,
                                                             self._path_step, 0.0, -0.1)
            if arrived:
                self._phase = _PLAN_DOWN
            return action

        if self._phase == _PLAN_DOWN:
            obs_rects = get_obstacle_rects(obs, exclude_idx=self._target_idx)
            path = plan_base_path(obs, self._primitives, self._pick_x, self._pick_y,
                                  obstacle_rects=obs_rects)
            self._path = path or [np.array([self._pick_x, self._pick_y])]
            self._path_step = 0
            self._phase = _NAV_DOWN
            return make_action(darm=-0.1)

        if self._phase == _NAV_DOWN:
            action, self._path_step, arrived = _follow_path(rx, ry, self._path,
                                                             self._path_step, 0.0, -0.1)
            if arrived:
                self._phase = _ROTATE
            return action

        if self._phase == _ROTATE:
            if abs(robot['theta'] - THETA_DOWN) > ANGLE_TOL:
                return rotate_toward(robot['theta'], THETA_DOWN)
            self._phase = _EXTEND
            return make_action()

        if self._phase == _EXTEND:
            if robot['arm_joint'] < robot['arm_length'] - ARM_TOL:
                return extend_arm(robot['arm_joint'], robot['arm_length'], vacuum=1.0)
            self._phase = _GRASP
            return make_action(vac=1.0)

        if self._phase == _GRASP:
            self._phase = _PLAN_UP
            return make_action(vac=1.0)

        if self._phase == _PLAN_UP:
            # Rise straight up to NAV_HEIGHT carrying obstruction
            path = plan_base_path(obs, self._primitives, self._pick_x, NAV_HEIGHT,
                                  obstacle_rects=[])
            self._path = path or [np.array([self._pick_x, NAV_HEIGHT])]
            self._path_step = 0
            self._phase = _NAV_UP
            return make_action(vac=1.0)

        if self._phase == _NAV_UP:
            action, self._path_step, arrived = _follow_path(rx, ry, self._path,
                                                             self._path_step, 1.0, 0.0)
            if arrived:
                # Release here - obstruction floats at NAV_HEIGHT above pick_x
                self._phase = _RELEASE
            return action

        if self._phase == _RELEASE:
            # Release; next iteration will retract arm & find next target
            self._phase = _FIND
            return make_action(vac=0.0)

        return make_action()


class PickupTargetBlock(Behavior):
    """Navigate to target block and grasp it."""

    def __init__(self, primitives):
        self._primitives = primitives
        self._phase = _PLAN_HIGH
        self._path = []
        self._path_step = 0
        self._pick_x = 0.0
        self._pick_y = 0.0

    def reset(self, obs):
        self._phase = _PLAN_HIGH
        self._path = []
        self._path_step = 0
        robot = extract_robot(obs)
        block = extract_target_block(obs)
        self._pick_x, self._pick_y = approach_xy_for_pick(block, robot['arm_length'])

    def initializable(self, obs) -> bool:
        return not any_obstruction_blocking_surface(obs) and not is_holding(obs)

    def terminated(self, obs) -> bool:
        """True when block is near gripper tip (actually grasped)."""
        if not is_holding(obs):
            return False
        tx, ty = gripper_tip_xy(obs)
        block = extract_target_block(obs)
        dist = np.hypot(block['x'] - tx, block['y'] - ty)
        return dist < 0.15  # block follows gripper

    def step(self, obs) -> np.ndarray:
        robot = extract_robot(obs)
        rx, ry = robot['x'], robot['y']

        if self._phase == _PLAN_HIGH:
            block = extract_target_block(obs)
            self._pick_x, self._pick_y = approach_xy_for_pick(block, robot['arm_length'])
            obs_rects = get_obstacle_rects(obs, exclude_idx=-1)  # all movable as obstacles
            # Don't include target block as obstacle (we want to reach it)
            path = plan_base_path(obs, self._primitives, self._pick_x, NAV_HEIGHT,
                                  obstacle_rects=obs_rects)
            self._path = path or [np.array([self._pick_x, NAV_HEIGHT])]
            self._path_step = 0
            self._phase = _NAV_HIGH
            return make_action(darm=-0.1)

        if self._phase == _NAV_HIGH:
            action, self._path_step, arrived = _follow_path(rx, ry, self._path,
                                                             self._path_step, 0.0, -0.1)
            if arrived:
                self._phase = _PLAN_DOWN
            return action

        if self._phase == _PLAN_DOWN:
            # Descend to pick position, excluding target block from obstacles
            obs_rects = get_obstacle_rects(obs, exclude_idx=-1)
            # Remove target block from rects (last entry added by get_obstacle_rects)
            if obs_rects:
                obs_rects = obs_rects[:-1]  # remove block (last item)
            path = plan_base_path(obs, self._primitives, self._pick_x, self._pick_y,
                                  obstacle_rects=obs_rects)
            self._path = path or [np.array([self._pick_x, self._pick_y])]
            self._path_step = 0
            self._phase = _NAV_DOWN
            return make_action(darm=-0.1)

        if self._phase == _NAV_DOWN:
            action, self._path_step, arrived = _follow_path(rx, ry, self._path,
                                                             self._path_step, 0.0, -0.1)
            if arrived:
                self._phase = _ROTATE
            return action

        if self._phase == _ROTATE:
            if abs(robot['theta'] - THETA_DOWN) > ANGLE_TOL:
                return rotate_toward(robot['theta'], THETA_DOWN)
            self._phase = _EXTEND
            return make_action()

        if self._phase == _EXTEND:
            if robot['arm_joint'] < robot['arm_length'] - ARM_TOL:
                return extend_arm(robot['arm_joint'], robot['arm_length'], vacuum=1.0)
            self._phase = _GRASP
            return make_action(vac=1.0)

        if self._phase == _GRASP:
            # Re-plan if we haven't grasped yet
            if not self.terminated(obs):
                # Try re-approaching
                self._phase = _PLAN_HIGH
            return make_action(vac=1.0)

        return make_action(vac=1.0)


class PlaceTargetBlock(Behavior):
    """Carry target block to target surface and set it down."""

    def __init__(self, primitives):
        self._primitives = primitives
        self._phase = _PLAN_HIGH
        self._path = []
        self._path_step = 0
        self._surf_x = 0.0
        self._place_y = 0.0
        self._block_x = 0.0
        self._block_y_pick = 0.0

    def reset(self, obs):
        self._phase = _PLAN_HIGH
        self._path = []
        self._path_step = 0
        self._update_targets(obs)

    def _update_targets(self, obs):
        robot = extract_robot(obs)
        surf = extract_target_surface(obs)
        block = extract_target_block(obs)
        # surf['x'] is bottom-left; use center for alignment
        self._surf_x = surf['x'] + surf['width'] / 2.0
        self._place_y = placement_robot_y(
            block['height'], robot['arm_length'], surf['y'], surf['height'])
        self._block_x = block['x'] + block['width'] / 2.0

    def initializable(self, obs) -> bool:
        return is_holding(obs)

    def terminated(self, obs) -> bool:
        return is_block_on_surface(obs)

    def _obs_rects_for_carry(self, obs):
        """Obstacles during carry: all obstructions (center-based, for BiRRT)."""
        from obs_helpers import get_obstacle_rects
        # Include ALL obstructions but not the target block (being carried)
        rects = []
        for i in range(NUM_OBSTRUCTIONS):
            ob = extract_obstruction(obs, i)
            cx = ob['x'] + ob['width'] / 2.0
            cy = ob['y'] + ob['height'] / 2.0
            rects.append((cx, cy, ob['width'], ob['height']))
        return rects

    def step(self, obs) -> np.ndarray:
        robot = extract_robot(obs)
        rx, ry = robot['x'], robot['y']
        vac = 1.0  # always keep vacuum on

        if self._phase == _PLAN_HIGH:
            self._update_targets(obs)
            obs_rects = self._obs_rects_for_carry(obs)
            # Rise to NAV_HEIGHT
            path = plan_base_path(obs, self._primitives, rx, NAV_HEIGHT,
                                  obstacle_rects=obs_rects)
            self._path = path or [np.array([rx, NAV_HEIGHT])]
            self._path_step = 0
            self._phase = _NAV_HIGH
            return make_action(vac=vac)

        if self._phase == _NAV_HIGH:
            action, self._path_step, arrived = _follow_path(rx, ry, self._path,
                                                             self._path_step, vac)
            if arrived:
                self._phase = _PLAN_SURF
            return action

        if self._phase == _PLAN_SURF:
            obs_rects = self._obs_rects_for_carry(obs)
            # Fly horizontally to surf_x at NAV_HEIGHT
            path = plan_base_path(obs, self._primitives, self._surf_x, NAV_HEIGHT,
                                  obstacle_rects=obs_rects)
            self._path = path or [np.array([self._surf_x, NAV_HEIGHT])]
            self._path_step = 0
            self._phase = _NAV_SURF
            return make_action(vac=vac)

        if self._phase == _NAV_SURF:
            action, self._path_step, arrived = _follow_path(rx, ry, self._path,
                                                             self._path_step, vac)
            if arrived:
                self._phase = _PLAN_PLACE
            return action

        if self._phase == _PLAN_PLACE:
            obs_rects = self._obs_rects_for_carry(obs)
            # Descend to placement position
            path = plan_base_path(obs, self._primitives, self._surf_x, self._place_y,
                                  obstacle_rects=obs_rects)
            self._path = path or [np.array([self._surf_x, self._place_y])]
            self._path_step = 0
            self._phase = _NAV_PLACE
            return make_action(vac=vac)

        if self._phase == _NAV_PLACE:
            action, self._path_step, arrived = _follow_path(rx, ry, self._path,
                                                             self._path_step, vac)
            if arrived:
                # At placement position - release or block already on surface
                return make_action(vac=0.0)
            return action

        return make_action(vac=vac)
