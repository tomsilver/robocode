"""Behavior classes for Obstruction2D-o4-v0."""
import numpy as np
from behavior import Behavior
from obs_helpers import (
    get_robot, get_surface, get_block, get_obstruction,
    gripper_pos, is_vacuum_on, is_holding_block,
    obstruction_overlaps_surface, any_obstruction_on_surface,
    block_on_surface, NUM_OBSTRUCTIONS,
    APPROACH_Y_ABOVE, ARM_MIN, ARM_MAX, TABLE_OBJ_Y,
    WORLD_X_MIN, WORLD_X_MAX,
)
from act_helpers import (
    clip_action, toward_pos, control_arm, normalize_angle,
    pos_reached, theta_reached, arm_reached,
    make_birrt_fns, get_drop_position,
    PICK_ARM_JOINT, DROP_ARM_JOINT, RETRACT_ARM,
    PLACE_ROBOT_Y, MAX_DX, MAX_DY, MAX_DTHETA,
    BIRRT_NUM_ATTEMPTS, BIRRT_NUM_ITERS, BIRRT_SMOOTH_AMT,
    K_POS, K_THETA, POS_TOL, THETA_TOL, ARM_TOL,
)

# ---- Phase constants ----
PHASE_NAV_PICK = 'nav_pick'
PHASE_ORIENT = 'orient'
PHASE_EXTEND = 'extend'
PHASE_GRASP = 'grasp'
PHASE_RETRACT = 'retract'
PHASE_NAV_DROP = 'nav_drop'
PHASE_EXTEND_DROP = 'extend_drop'
PHASE_DROP = 'drop'
PHASE_RETRACT_DROP = 'retract_drop'
PHASE_DONE = 'done'

GRASP_HOLD_STEPS = 5     # steps to hold vacuum while grasping
DROP_HOLD_STEPS = 5      # steps to hold release


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
    if pos_reached(rx, ry, tx, ty, tol=POS_TOL * 2):
        path_idx += 1
        if path_idx < len(path):
            wp = path[path_idx]
            tx, ty = float(wp[0]), float(wp[1])
        else:
            return clip_action(vac=vac), path_idx
    action = toward_pos(rx, ry, tx, ty, vac=vac, cur_theta=robot['theta'])
    return action, path_idx


class PickAndPlaceOneObject(Behavior):
    """Pick up one object at (pick_x, pick_y) and place at (drop_x, table_y)."""

    def __init__(self, primitives, pick_x_fn, pick_y_fn,
                 drop_x, drop_y,
                 is_done_fn,
                 excluded_obs_for_nav=None):
        """
        pick_x_fn(obs), pick_y_fn(obs): callables returning pick target position
        drop_x, drop_y: where to drop
        is_done_fn(obs): callable returning True when this pnp is complete
        excluded_obs_for_nav: set of obstruction indices to ignore in collision
        """
        self._primitives = primitives
        self._pick_x_fn = pick_x_fn
        self._pick_y_fn = pick_y_fn
        self._drop_x = drop_x
        self._drop_y = drop_y
        self._is_done_fn = is_done_fn
        self._excluded = excluded_obs_for_nav or set()
        self._rng = np.random.default_rng(42)

        self._phase = PHASE_NAV_PICK
        self._path = None
        self._path_idx = 0
        self._hold_counter = 0

    def reset(self, obs):
        self._phase = PHASE_NAV_PICK
        self._path = None
        self._path_idx = 0
        self._hold_counter = 0
        self._plan_pick_path(obs)

    def _plan_pick_path(self, obs):
        robot = get_robot(obs)
        px = self._pick_x_fn(obs)
        goal_y = APPROACH_Y_ABOVE  # robot y for approach
        path = _plan_path(
            self._primitives, obs,
            [robot['x'], robot['y']],
            [px, goal_y],
            excluded=self._excluded,
            rng=self._rng,
        )
        if path is None:
            # Fallback: direct line
            path = [np.array([px, goal_y])]
        self._path = path
        self._path_idx = 0

    def _plan_drop_path(self, obs):
        robot = get_robot(obs)
        path = _plan_path(
            self._primitives, obs,
            [robot['x'], robot['y']],
            [self._drop_x, APPROACH_Y_ABOVE],
            excluded=self._excluded,
            rng=self._rng,
        )
        if path is None:
            path = [np.array([self._drop_x, APPROACH_Y_ABOVE])]
        self._path = path
        self._path_idx = 0

    def initializable(self, obs):
        return True

    def terminated(self, obs):
        return self._is_done_fn(obs)

    def step(self, obs):
        robot = get_robot(obs)
        rx, ry, theta = robot['x'], robot['y'], robot['theta']
        arm = robot['arm_joint']
        vac = 1.0 if is_vacuum_on(obs) else 0.0

        if self._phase == PHASE_NAV_PICK:
            # Navigate to approach position above pick target
            px = self._pick_x_fn(obs)
            goal_y = APPROACH_Y_ABOVE
            # Replan if path is empty or exhausted
            if self._path is None or self._path_idx >= len(self._path):
                self._plan_pick_path(obs)
            action, self._path_idx = _follow_path_action(robot, self._path, self._path_idx, vac=0.0)
            # Check if reached
            if pos_reached(rx, ry, px, goal_y, tol=0.025) or self._path_idx >= len(self._path):
                if pos_reached(rx, ry, px, goal_y, tol=0.025):
                    self._phase = PHASE_ORIENT
            return action

        elif self._phase == PHASE_ORIENT:
            # Point arm downward (theta = -pi/2)
            err_theta = normalize_angle(-np.pi / 2 - theta)
            dtheta = np.clip(err_theta * K_THETA, -MAX_DTHETA, MAX_DTHETA)
            if theta_reached(theta):
                self._phase = PHASE_EXTEND
            return clip_action(dtheta=dtheta, vac=0.0)

        elif self._phase == PHASE_EXTEND:
            # Extend arm to pick arm joint
            target_arm = PICK_ARM_JOINT
            if arm_reached(arm, target_arm):
                self._phase = PHASE_GRASP
                self._hold_counter = 0
            return control_arm(arm, target_arm, vac=0.0)

        elif self._phase == PHASE_GRASP:
            # Turn vacuum on and hold
            self._hold_counter += 1
            if self._hold_counter >= GRASP_HOLD_STEPS:
                self._phase = PHASE_RETRACT
            return clip_action(vac=1.0)

        elif self._phase == PHASE_RETRACT:
            # Retract arm
            if arm_reached(arm, RETRACT_ARM):
                self._phase = PHASE_NAV_DROP
                self._plan_drop_path(obs)
            return control_arm(arm, RETRACT_ARM, vac=1.0)

        elif self._phase == PHASE_NAV_DROP:
            # Navigate to drop position
            if self._path is None or self._path_idx >= len(self._path):
                self._plan_drop_path(obs)
            action, self._path_idx = _follow_path_action(robot, self._path, self._path_idx, vac=1.0)
            if pos_reached(rx, ry, self._drop_x, APPROACH_Y_ABOVE, tol=0.025) or \
               self._path_idx >= len(self._path):
                if pos_reached(rx, ry, self._drop_x, APPROACH_Y_ABOVE, tol=0.03):
                    self._phase = PHASE_EXTEND_DROP
            return action

        elif self._phase == PHASE_EXTEND_DROP:
            # Extend arm to drop level (same as pick)
            if arm_reached(arm, DROP_ARM_JOINT):
                self._phase = PHASE_DROP
                self._hold_counter = 0
            return control_arm(arm, DROP_ARM_JOINT, vac=1.0)

        elif self._phase == PHASE_DROP:
            # Turn vacuum off
            self._hold_counter += 1
            if self._hold_counter >= DROP_HOLD_STEPS:
                self._phase = PHASE_RETRACT_DROP
            return clip_action(vac=0.0)

        elif self._phase == PHASE_RETRACT_DROP:
            if arm_reached(arm, RETRACT_ARM):
                self._phase = PHASE_DONE
            return control_arm(arm, RETRACT_ARM, vac=0.0)

        # DONE
        return clip_action(vac=0.0)


class ClearAllObstructions(Behavior):
    """Clear all obstructions from the target surface."""

    def __init__(self, primitives):
        self._primitives = primitives
        self._current_pnp = None
        self._remaining = []
        self._rng = np.random.default_rng(123)

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
            self._current_pnp = None
            return
        i = self._remaining[0]
        drop_x, drop_y = get_drop_position(i)

        def pick_x(o): return get_obstruction(o, i)['x']
        def pick_y(o): return get_obstruction(o, i)['y']

        def done_fn(o):
            return not obstruction_overlaps_surface(o, i)

        self._current_pnp = PickAndPlaceOneObject(
            self._primitives,
            pick_x_fn=pick_x,
            pick_y_fn=pick_y,
            drop_x=drop_x,
            drop_y=APPROACH_Y_ABOVE,
            is_done_fn=done_fn,
            excluded_obs_for_nav={i},
        )
        self._current_pnp.reset(obs)

    def step(self, obs):
        if self._current_pnp is None:
            return clip_action()

        if self._current_pnp.terminated(obs):
            # Move to next
            if self._remaining:
                self._remaining.pop(0)
            # Check for newly overlapping ones
            new_remaining = [
                i for i in range(NUM_OBSTRUCTIONS)
                if obstruction_overlaps_surface(obs, i)
            ]
            if new_remaining:
                self._remaining = new_remaining
                self._start_next(obs)
            else:
                self._current_pnp = None
                return clip_action()

        return self._current_pnp.step(obs)


class GraspBlock(Behavior):
    """Navigate to target block and grasp it."""

    def __init__(self, primitives):
        self._primitives = primitives
        self._pnp = None
        self._phase = PHASE_NAV_PICK
        self._path = None
        self._path_idx = 0
        self._hold_counter = 0
        self._rng = np.random.default_rng(7)

    def initializable(self, obs):
        return not any_obstruction_on_surface(obs)

    def terminated(self, obs):
        return is_holding_block(obs)

    def reset(self, obs):
        self._phase = PHASE_NAV_PICK
        self._path = None
        self._path_idx = 0
        self._hold_counter = 0
        self._plan_path(obs)

    def _plan_path(self, obs):
        robot = get_robot(obs)
        b = get_block(obs)
        path = _plan_path(
            self._primitives, obs,
            [robot['x'], robot['y']],
            [b['x'], APPROACH_Y_ABOVE],
            rng=self._rng,
        )
        if path is None:
            path = [np.array([b['x'], APPROACH_Y_ABOVE])]
        self._path = path
        self._path_idx = 0

    def step(self, obs):
        robot = get_robot(obs)
        rx, ry, theta = robot['x'], robot['y'], robot['theta']
        arm = robot['arm_joint']
        b = get_block(obs)

        if self._phase == PHASE_NAV_PICK:
            if self._path is None or self._path_idx >= len(self._path):
                self._plan_path(obs)
            action, self._path_idx = _follow_path_action(robot, self._path, self._path_idx, vac=0.0)
            if pos_reached(rx, ry, b['x'], APPROACH_Y_ABOVE, tol=0.025) or \
               self._path_idx >= len(self._path):
                if pos_reached(rx, ry, b['x'], APPROACH_Y_ABOVE, tol=0.03):
                    self._phase = PHASE_ORIENT
            return action

        elif self._phase == PHASE_ORIENT:
            err_theta = normalize_angle(-np.pi / 2 - theta)
            dtheta = np.clip(err_theta * K_THETA, -MAX_DTHETA, MAX_DTHETA)
            if theta_reached(theta):
                self._phase = PHASE_EXTEND
            return clip_action(dtheta=dtheta, vac=0.0)

        elif self._phase == PHASE_EXTEND:
            if arm_reached(arm, PICK_ARM_JOINT):
                self._phase = PHASE_GRASP
                self._hold_counter = 0
            return control_arm(arm, PICK_ARM_JOINT, vac=0.0)

        elif self._phase == PHASE_GRASP:
            self._hold_counter += 1
            if self._hold_counter >= GRASP_HOLD_STEPS:
                self._phase = PHASE_DONE
            return clip_action(vac=1.0)

        return clip_action(vac=1.0)


class PlaceBlock(Behavior):
    """Move block to target surface and release."""

    def __init__(self, primitives):
        self._primitives = primitives
        self._phase = PHASE_NAV_DROP
        self._path = None
        self._path_idx = 0
        self._hold_counter = 0
        self._rng = np.random.default_rng(99)

    def initializable(self, obs):
        return is_holding_block(obs)

    def terminated(self, obs):
        return block_on_surface(obs)

    def reset(self, obs):
        self._phase = PHASE_NAV_DROP
        self._path = None
        self._path_idx = 0
        self._hold_counter = 0
        self._plan_path(obs)

    def _plan_path(self, obs):
        robot = get_robot(obs)
        s = get_surface(obs)
        # Navigate robot to (surface_x, PLACE_ROBOT_Y) while holding block
        # Block follows, so block x = robot x (arm vertical)
        path = _plan_path(
            self._primitives, obs,
            [robot['x'], robot['y']],
            [s['x'], PLACE_ROBOT_Y],
            excluded=set(range(NUM_OBSTRUCTIONS)),  # ignore all obstructions (cleared)
            rng=self._rng,
        )
        if path is None:
            path = [np.array([s['x'], PLACE_ROBOT_Y])]
        self._path = path
        self._path_idx = 0

    def step(self, obs):
        robot = get_robot(obs)
        rx, ry, theta = robot['x'], robot['y'], robot['theta']
        arm = robot['arm_joint']
        s = get_surface(obs)

        if self._phase == PHASE_NAV_DROP:
            if self._path is None or self._path_idx >= len(self._path):
                self._plan_path(obs)
            action, self._path_idx = _follow_path_action(robot, self._path, self._path_idx, vac=1.0)
            if pos_reached(rx, ry, s['x'], PLACE_ROBOT_Y, tol=0.025) or \
               self._path_idx >= len(self._path):
                if pos_reached(rx, ry, s['x'], PLACE_ROBOT_Y, tol=0.04):
                    self._phase = PHASE_DROP
                    self._hold_counter = 0
            return action

        elif self._phase == PHASE_DROP:
            # Release vacuum
            self._hold_counter += 1
            return clip_action(vac=0.0)

        return clip_action(vac=0.0)
