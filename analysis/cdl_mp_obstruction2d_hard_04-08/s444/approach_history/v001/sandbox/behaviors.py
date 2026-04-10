"""Behavior classes for Obstruction2D-o4-v0."""
import numpy as np
from behavior import Behavior
from obs_helpers import (
    get_robot, get_surface, get_block, get_obstruction,
    gripper_pos, is_vacuum_on, is_holding_block,
    obstruction_overlaps_surface, any_obstruction_on_surface,
    block_on_surface, NUM_OBSTRUCTIONS,
    ARM_MIN, ARM_MAX, TABLE_OBJ_Y, HOLDING_DIST_THRESH,
)
from act_helpers import (
    clip_action, toward_pos, control_arm, normalize_angle,
    pos_reached, theta_reached, arm_reached,
    make_birrt_fns, get_drop_position,
    PICK_ARM_JOINT, DROP_ARM_JOINT, RETRACT_ARM,
    NAV_HIGH_Y, APPROACH_LOW_Y, PLACE_ROBOT_Y,
    MAX_DX, MAX_DY, MAX_DTHETA,
    BIRRT_NUM_ATTEMPTS, BIRRT_NUM_ITERS, BIRRT_SMOOTH_AMT,
    K_POS, K_THETA, POS_TOL, THETA_TOL, ARM_TOL,
    DROP_X_BASE, DROP_Y_OFFSET,
)

GRASP_HOLD_STEPS = 8     # steps to hold vacuum while grasping
DROP_HOLD_STEPS = 8      # steps to hold release
NAV_TOL = 0.025          # navigation position tolerance
NAV_TOL_LOOSE = 0.04     # looser tolerance for phase transition


def _plan_path(primitives, obs, start_xy, goal_xy, excluded=None, rng=None):
    """Use BiRRT to plan a 2D (x,y) path for the robot base."""
    if rng is None:
        rng = np.random.default_rng(42)
    sample_fn, extend_fn, collision_fn, distance_fn = make_birrt_fns(
        obs, excluded_obs_indices=excluded, rng=rng
    )
    BiRRT = primitives['BiRRT']
    birrt = BiRRT(
        sample_fn, extend_fn, collision_fn, distance_fn,
        rng, BIRRT_NUM_ATTEMPTS, BIRRT_NUM_ITERS, BIRRT_SMOOTH_AMT
    )
    start = np.array(start_xy, dtype=float)
    goal = np.array(goal_xy, dtype=float)
    path = birrt.query(start, goal)
    return path  # list of np.array([x,y]) or None


def _follow_path_action(robot, path, path_idx, vac=0.0):
    """Generate action to move toward path[path_idx]. Returns (action, new_idx)."""
    if path_idx >= len(path):
        return clip_action(vac=vac), path_idx
    wp = path[path_idx]
    rx, ry = robot['x'], robot['y']
    tx, ty = float(wp[0]), float(wp[1])
    if pos_reached(rx, ry, tx, ty, tol=NAV_TOL):
        path_idx += 1
        if path_idx < len(path):
            wp = path[path_idx]
            tx, ty = float(wp[0]), float(wp[1])
        else:
            return clip_action(vac=vac), path_idx
    action = toward_pos(rx, ry, tx, ty, vac=vac, cur_theta=robot['theta'])
    return action, path_idx


class PickOnce(Behavior):
    """Pick object i (obstruction) and move it to a drop zone, then release."""

    # Internal phases
    _PH_HIGH_TO_PICK = 'high_to_pick'
    _PH_LOW_TO_PICK = 'low_to_pick'
    _PH_ORIENT = 'orient'
    _PH_EXTEND = 'extend'
    _PH_GRASP = 'grasp'
    _PH_RETRACT = 'retract'
    _PH_HIGH_TO_DROP = 'high_to_drop'
    _PH_LOW_TO_DROP = 'low_to_drop'
    _PH_EXTEND_DROP = 'extend_drop'
    _PH_RELEASE = 'release'
    _PH_RETRACT2 = 'retract2'
    _PH_DONE = 'done'

    def __init__(self, primitives, obs_idx, drop_x):
        self._primitives = primitives
        self._obs_idx = obs_idx
        self._drop_x = drop_x
        self._excl = {obs_idx}  # exclude self from collision
        self._all_excl = set(range(NUM_OBSTRUCTIONS))  # exclude all for transit
        self._rng = np.random.default_rng(obs_idx * 7 + 13)

        self._phase = self._PH_HIGH_TO_PICK
        self._path = None
        self._path_idx = 0
        self._hold_counter = 0

    def initializable(self, obs): return True
    def terminated(self, obs): return not obstruction_overlaps_surface(obs, self._obs_idx)

    def reset(self, obs):
        self._phase = self._PH_HIGH_TO_PICK
        self._path = None
        self._path_idx = 0
        self._hold_counter = 0
        self._plan_high_to_pick(obs)

    def _pick_x(self, obs): return get_obstruction(obs, self._obs_idx)['x']

    def _plan_high_to_pick(self, obs):
        robot = get_robot(obs)
        px = self._pick_x(obs)
        path = _plan_path(self._primitives, obs,
                          [robot['x'], robot['y']], [px, NAV_HIGH_Y],
                          excluded=self._all_excl, rng=self._rng)
        self._path = path or [np.array([px, NAV_HIGH_Y])]
        self._path_idx = 0

    def _plan_low_to_pick(self, obs):
        robot = get_robot(obs)
        px = self._pick_x(obs)
        # Straight descent — only exclude self
        path = _plan_path(self._primitives, obs,
                          [robot['x'], robot['y']], [px, APPROACH_LOW_Y],
                          excluded=self._excl, rng=self._rng)
        self._path = path or [np.array([px, APPROACH_LOW_Y])]
        self._path_idx = 0

    def _plan_high_to_drop(self, obs):
        robot = get_robot(obs)
        path = _plan_path(self._primitives, obs,
                          [robot['x'], robot['y']], [self._drop_x, NAV_HIGH_Y],
                          excluded=self._all_excl, rng=self._rng)
        self._path = path or [np.array([self._drop_x, NAV_HIGH_Y])]
        self._path_idx = 0

    def _plan_low_to_drop(self, obs):
        robot = get_robot(obs)
        path = _plan_path(self._primitives, obs,
                          [robot['x'], robot['y']], [self._drop_x, APPROACH_LOW_Y],
                          excluded=self._all_excl, rng=self._rng)
        self._path = path or [np.array([self._drop_x, APPROACH_LOW_Y])]
        self._path_idx = 0

    def step(self, obs):
        robot = get_robot(obs)
        rx, ry = robot['x'], robot['y']
        theta = robot['theta']
        arm = robot['arm_joint']
        vac = 1.0 if is_vacuum_on(obs) else 0.0
        px = self._pick_x(obs)

        if self._phase == self._PH_HIGH_TO_PICK:
            if self._path is None or self._path_idx >= len(self._path):
                self._plan_high_to_pick(obs)
            action, self._path_idx = _follow_path_action(robot, self._path, self._path_idx, vac=0.0)
            if pos_reached(rx, ry, px, NAV_HIGH_Y, tol=NAV_TOL_LOOSE) or self._path_idx >= len(self._path):
                self._phase = self._PH_LOW_TO_PICK
                self._plan_low_to_pick(obs)
            return action

        elif self._phase == self._PH_LOW_TO_PICK:
            if self._path is None or self._path_idx >= len(self._path):
                self._plan_low_to_pick(obs)
            action, self._path_idx = _follow_path_action(robot, self._path, self._path_idx, vac=0.0)
            if pos_reached(rx, ry, px, APPROACH_LOW_Y, tol=NAV_TOL_LOOSE) or self._path_idx >= len(self._path):
                self._phase = self._PH_ORIENT
            return action

        elif self._phase == self._PH_ORIENT:
            err = normalize_angle(-np.pi / 2 - theta)
            dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
            if theta_reached(theta):
                self._phase = self._PH_EXTEND
            return clip_action(dtheta=dtheta, vac=0.0)

        elif self._phase == self._PH_EXTEND:
            if arm_reached(arm, PICK_ARM_JOINT):
                self._phase = self._PH_GRASP
                self._hold_counter = 0
            return control_arm(arm, PICK_ARM_JOINT, vac=0.0)

        elif self._phase == self._PH_GRASP:
            self._hold_counter += 1
            if self._hold_counter >= GRASP_HOLD_STEPS:
                self._phase = self._PH_RETRACT
            return clip_action(vac=1.0)

        elif self._phase == self._PH_RETRACT:
            if arm_reached(arm, RETRACT_ARM):
                self._phase = self._PH_HIGH_TO_DROP
                self._plan_high_to_drop(obs)
            return control_arm(arm, RETRACT_ARM, vac=1.0)

        elif self._phase == self._PH_HIGH_TO_DROP:
            if self._path is None or self._path_idx >= len(self._path):
                self._plan_high_to_drop(obs)
            action, self._path_idx = _follow_path_action(robot, self._path, self._path_idx, vac=1.0)
            if pos_reached(rx, ry, self._drop_x, NAV_HIGH_Y, tol=NAV_TOL_LOOSE) or self._path_idx >= len(self._path):
                self._phase = self._PH_LOW_TO_DROP
                self._plan_low_to_drop(obs)
            return action

        elif self._phase == self._PH_LOW_TO_DROP:
            if self._path is None or self._path_idx >= len(self._path):
                self._plan_low_to_drop(obs)
            action, self._path_idx = _follow_path_action(robot, self._path, self._path_idx, vac=1.0)
            if pos_reached(rx, ry, self._drop_x, APPROACH_LOW_Y, tol=NAV_TOL_LOOSE) or self._path_idx >= len(self._path):
                self._phase = self._PH_EXTEND_DROP
            return action

        elif self._phase == self._PH_EXTEND_DROP:
            if arm_reached(arm, DROP_ARM_JOINT):
                self._phase = self._PH_RELEASE
                self._hold_counter = 0
            return control_arm(arm, DROP_ARM_JOINT, vac=1.0)

        elif self._phase == self._PH_RELEASE:
            self._hold_counter += 1
            if self._hold_counter >= DROP_HOLD_STEPS:
                self._phase = self._PH_RETRACT2
            return clip_action(vac=0.0)

        elif self._phase == self._PH_RETRACT2:
            if arm_reached(arm, RETRACT_ARM):
                self._phase = self._PH_DONE
            return control_arm(arm, RETRACT_ARM, vac=0.0)

        return clip_action(vac=0.0)


class ClearAllObstructions(Behavior):
    """Clear all obstructions from the target surface."""

    def __init__(self, primitives):
        self._primitives = primitives
        self._current = None
        self._remaining = []

    def initializable(self, obs):
        return any_obstruction_on_surface(obs)

    def terminated(self, obs):
        return not any_obstruction_on_surface(obs)

    def reset(self, obs):
        self._remaining = [
            i for i in range(NUM_OBSTRUCTIONS)
            if obstruction_overlaps_surface(obs, i)
        ]
        self._start_next(obs)

    def _start_next(self, obs):
        if not self._remaining:
            self._current = None
            return
        i = self._remaining[0]
        drop_x = DROP_X_BASE - i * DROP_Y_OFFSET
        self._current = PickOnce(self._primitives, i, drop_x)
        self._current.reset(obs)

    def step(self, obs):
        if self._current is None:
            return clip_action()
        if self._current.terminated(obs):
            self._remaining.pop(0)
            # Refresh: maybe more are on surface after movement
            new_on = [i for i in range(NUM_OBSTRUCTIONS)
                      if obstruction_overlaps_surface(obs, i)]
            if new_on:
                self._remaining = new_on
                self._start_next(obs)
            else:
                self._current = None
                return clip_action()
        return self._current.step(obs)


class GraspBlock(Behavior):
    """Navigate to target block and grasp it."""

    _PH_HIGH = 'high'
    _PH_LOW = 'low'
    _PH_ORIENT = 'orient'
    _PH_EXTEND = 'extend'
    _PH_GRASP = 'grasp'
    _PH_DONE = 'done'

    def __init__(self, primitives):
        self._primitives = primitives
        self._rng = np.random.default_rng(7)
        self._phase = self._PH_HIGH
        self._path = None
        self._path_idx = 0
        self._hold_counter = 0

    def initializable(self, obs):
        return not any_obstruction_on_surface(obs)

    def terminated(self, obs):
        return is_holding_block(obs)

    def reset(self, obs):
        self._phase = self._PH_HIGH
        self._path = None
        self._path_idx = 0
        self._hold_counter = 0
        self._plan_high(obs)

    def _plan_high(self, obs):
        robot = get_robot(obs)
        b = get_block(obs)
        path = _plan_path(self._primitives, obs,
                          [robot['x'], robot['y']], [b['x'], NAV_HIGH_Y],
                          excluded=set(range(NUM_OBSTRUCTIONS)), rng=self._rng)
        self._path = path or [np.array([b['x'], NAV_HIGH_Y])]
        self._path_idx = 0

    def _plan_low(self, obs):
        robot = get_robot(obs)
        b = get_block(obs)
        path = _plan_path(self._primitives, obs,
                          [robot['x'], robot['y']], [b['x'], APPROACH_LOW_Y],
                          excluded=set(range(NUM_OBSTRUCTIONS)), rng=self._rng)
        self._path = path or [np.array([b['x'], APPROACH_LOW_Y])]
        self._path_idx = 0

    def step(self, obs):
        robot = get_robot(obs)
        rx, ry = robot['x'], robot['y']
        theta = robot['theta']
        arm = robot['arm_joint']
        b = get_block(obs)

        if self._phase == self._PH_HIGH:
            if self._path is None or self._path_idx >= len(self._path):
                self._plan_high(obs)
            action, self._path_idx = _follow_path_action(robot, self._path, self._path_idx, vac=0.0)
            if pos_reached(rx, ry, b['x'], NAV_HIGH_Y, tol=NAV_TOL_LOOSE) or self._path_idx >= len(self._path):
                self._phase = self._PH_LOW
                self._plan_low(obs)
            return action

        elif self._phase == self._PH_LOW:
            if self._path is None or self._path_idx >= len(self._path):
                self._plan_low(obs)
            action, self._path_idx = _follow_path_action(robot, self._path, self._path_idx, vac=0.0)
            if pos_reached(rx, ry, b['x'], APPROACH_LOW_Y, tol=NAV_TOL_LOOSE) or self._path_idx >= len(self._path):
                self._phase = self._PH_ORIENT
            return action

        elif self._phase == self._PH_ORIENT:
            err = normalize_angle(-np.pi / 2 - theta)
            dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
            if theta_reached(theta):
                self._phase = self._PH_EXTEND
            return clip_action(dtheta=dtheta, vac=0.0)

        elif self._phase == self._PH_EXTEND:
            if arm_reached(arm, PICK_ARM_JOINT):
                self._phase = self._PH_GRASP
                self._hold_counter = 0
            return control_arm(arm, PICK_ARM_JOINT, vac=0.0)

        elif self._phase == self._PH_GRASP:
            self._hold_counter += 1
            if self._hold_counter >= GRASP_HOLD_STEPS:
                self._phase = self._PH_DONE
            return clip_action(vac=1.0)

        return clip_action(vac=1.0)


class PlaceBlock(Behavior):
    """Navigate holding block to target surface and release."""

    _PH_HIGH = 'high'
    _PH_LOW = 'low'
    _PH_RELEASE = 'release'

    def __init__(self, primitives):
        self._primitives = primitives
        self._rng = np.random.default_rng(99)
        self._phase = self._PH_HIGH
        self._path = None
        self._path_idx = 0
        self._hold_counter = 0

    def initializable(self, obs):
        return is_holding_block(obs)

    def terminated(self, obs):
        return block_on_surface(obs)

    def reset(self, obs):
        self._phase = self._PH_HIGH
        self._path = None
        self._path_idx = 0
        self._hold_counter = 0
        self._plan_high(obs)

    def _surf_x(self, obs): return get_surface(obs)['x']

    def _plan_high(self, obs):
        robot = get_robot(obs)
        sx = self._surf_x(obs)
        path = _plan_path(self._primitives, obs,
                          [robot['x'], robot['y']], [sx, NAV_HIGH_Y],
                          excluded=set(range(NUM_OBSTRUCTIONS)), rng=self._rng)
        self._path = path or [np.array([sx, NAV_HIGH_Y])]
        self._path_idx = 0

    def _plan_low(self, obs):
        robot = get_robot(obs)
        sx = self._surf_x(obs)
        path = _plan_path(self._primitives, obs,
                          [robot['x'], robot['y']], [sx, PLACE_ROBOT_Y],
                          excluded=set(range(NUM_OBSTRUCTIONS)), rng=self._rng)
        self._path = path or [np.array([sx, PLACE_ROBOT_Y])]
        self._path_idx = 0

    def step(self, obs):
        robot = get_robot(obs)
        rx, ry = robot['x'], robot['y']
        theta = robot['theta']
        arm = robot['arm_joint']
        sx = self._surf_x(obs)

        if self._phase == self._PH_HIGH:
            if self._path is None or self._path_idx >= len(self._path):
                self._plan_high(obs)
            action, self._path_idx = _follow_path_action(robot, self._path, self._path_idx, vac=1.0)
            if pos_reached(rx, ry, sx, NAV_HIGH_Y, tol=NAV_TOL_LOOSE) or self._path_idx >= len(self._path):
                self._phase = self._PH_LOW
                self._plan_low(obs)
            return action

        elif self._phase == self._PH_LOW:
            if self._path is None or self._path_idx >= len(self._path):
                self._plan_low(obs)
            action, self._path_idx = _follow_path_action(robot, self._path, self._path_idx, vac=1.0)
            if pos_reached(rx, ry, sx, PLACE_ROBOT_Y, tol=NAV_TOL_LOOSE) or self._path_idx >= len(self._path):
                self._phase = self._PH_RELEASE
                self._hold_counter = 0
            return action

        elif self._phase == self._PH_RELEASE:
            self._hold_counter += 1
            # Also correct arm to point down and correct x
            err_theta = normalize_angle(-np.pi / 2 - theta)
            dtheta = np.clip(err_theta * K_THETA, -MAX_DTHETA, MAX_DTHETA)
            # Move x toward surface center
            dx = np.clip((sx - rx) * K_POS, -MAX_DX, MAX_DX)
            # Keep arm extended for placement
            target_arm = PICK_ARM_JOINT
            darm = np.clip((target_arm - arm) * 5.0, -0.1, 0.1)
            if self._hold_counter >= 3:
                return clip_action(dx=dx, dtheta=dtheta, darm=darm, vac=0.0)
            return clip_action(dx=dx, dtheta=dtheta, darm=darm, vac=1.0)

        return clip_action(vac=0.0)
