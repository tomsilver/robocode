"""Behavior classes for Obstruction2D-o4-v0."""
import numpy as np
from behavior import Behavior
from obs_helpers import (
    get_robot, get_surface, get_block, get_obstruction,
    is_vacuum_on, is_holding_block,
    obstruction_overlaps_surface, any_obstruction_on_surface,
    block_on_surface, NUM_OBSTRUCTIONS,
    pick_robot_y, place_robot_y,
    PICK_ARM_JOINT, TABLE_OBJ_Y, ARM_MIN, ARM_MAX,
    ROB_RADIUS,
)
from act_helpers import (
    clip_action, toward_pos, control_arm, normalize_angle,
    pos_reached, theta_reached, arm_reached,
    plan_path, get_drop_xy,
    PICK_ARM_JOINT as ACT_PICK_ARM, RETRACT_ARM,
    NAV_HIGH_Y, MAX_DTHETA, K_THETA, POS_TOL,
    MAX_DX, MAX_DY, K_POS,
)

# ---- Internal phase names ----
PH_NAV_HIGH = 'nav_high'
PH_NAV_LOW = 'nav_low'
PH_ORIENT = 'orient'
PH_EXTEND = 'extend'
PH_GRASP = 'grasp'
PH_RETRACT = 'retract'
PH_NAV_HIGH2 = 'nav_high2'
PH_NAV_LOW2 = 'nav_low2'
PH_RELEASE = 'release'
PH_DONE = 'done'
PH_PUSH = 'push'

GRASP_HOLD = 8      # steps to hold vacuum during grasp
RELEASE_HOLD = 8    # steps to hold release
NAV_TOL = 0.025     # navigation position tolerance

# Push strategy constants
PUSH_Y = 0.26           # pushing altitude (above table, arm pointing up)
PUSH_MARGIN = 0.12      # extra x margin for approach
PUSH_EXTRA = 0.55       # how far past obstruction to push robot
PUSH_TOL = 0.04         # position tolerance for push nav


def _nav_action(robot, goal_x, goal_y, vac=0.0):
    """Proportional control to navigate to (goal_x, goal_y), arm down."""
    return toward_pos(robot['x'], robot['y'], goal_x, goal_y,
                      vac=vac, cur_theta=robot['theta'])


def _nav_action_arm_up(robot, goal_x, goal_y, vac=0.0):
    """Proportional control to navigate to (goal_x, goal_y), arm pointing UP."""
    rx, ry = robot['x'], robot['y']
    theta = robot['theta']
    dx = np.clip((goal_x - rx) * K_POS, -MAX_DX, MAX_DX)
    dy = np.clip((goal_y - ry) * K_POS, -MAX_DY, MAX_DY)
    err = normalize_angle(np.pi / 2 - theta)   # target theta = +pi/2 (arm up)
    dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
    return clip_action(dx=dx, dy=dy, dtheta=dtheta, vac=vac)


def _at_goal(robot, goal_x, goal_y, tol=NAV_TOL):
    return pos_reached(robot['x'], robot['y'], goal_x, goal_y, tol=tol)


class PushOneObstruction(Behavior):
    """Push obstruction i sideways off the surface (avoids descent problems)."""

    def __init__(self, primitives, obs_idx):
        self._primitives = primitives
        self._i = obs_idx
        self._phase = PH_NAV_HIGH
        self._approach_x = 0.0
        self._push_target_x = 0.0

    def initializable(self, obs): return True

    def terminated(self, obs):
        return not obstruction_overlaps_surface(obs, self._i)

    def reset(self, obs):
        self._phase = PH_NAV_HIGH
        self._compute_push_params(obs)

    def _compute_push_params(self, obs):
        o = get_obstruction(obs, self._i)
        s = get_surface(obs)
        obs_x = o['x']
        obs_w = o['width']
        surf_x = s['x']
        surf_w = s['width']

        # Determine push direction: push away from surface center
        if obs_x <= surf_x:
            # Obstruction left of (or at) surface → push further left
            # Approach from the RIGHT of the obstruction
            self._approach_x = obs_x + obs_w / 2 + ROB_RADIUS + PUSH_MARGIN
            # Push robot far to the left
            self._push_target_x = max(0.15, obs_x - PUSH_EXTRA)
        else:
            # Obstruction right of surface → push further right
            # Approach from the LEFT of the obstruction
            self._approach_x = obs_x - obs_w / 2 - ROB_RADIUS - PUSH_MARGIN
            # Push robot far to the right
            self._push_target_x = min(1.50, obs_x + PUSH_EXTRA)

        # Clamp approach_x to valid range
        self._approach_x = np.clip(self._approach_x, 0.15, 1.50)

    def step(self, obs):
        robot = get_robot(obs)
        rx, ry = robot['x'], robot['y']

        if self._phase == PH_NAV_HIGH:
            # Navigate to (approach_x, NAV_HIGH_Y) with arm pointing up
            action = _nav_action_arm_up(robot, self._approach_x, NAV_HIGH_Y, vac=0.0)
            if _at_goal(robot, self._approach_x, NAV_HIGH_Y, tol=0.05):
                self._phase = PH_NAV_LOW
            return action

        elif self._phase == PH_NAV_LOW:
            # Descend to push altitude (arm pointing up avoids table collision)
            action = _nav_action_arm_up(robot, self._approach_x, PUSH_Y, vac=0.0)
            if _at_goal(robot, self._approach_x, PUSH_Y, tol=PUSH_TOL):
                self._phase = PH_PUSH
            return action

        elif self._phase == PH_PUSH:
            # Push by navigating toward push_target_x at PUSH_Y
            # Re-read obstruction position in case it moved
            o = get_obstruction(obs, self._i)
            s = get_surface(obs)
            # If obstruction moved enough, update params
            action = _nav_action_arm_up(robot, self._push_target_x, PUSH_Y, vac=0.0)
            return action

        return clip_action(vac=0.0)


class ClearAllObstructions(Behavior):

    def __init__(self, primitives):
        self._primitives = primitives
        self._current = None
        self._remaining = []

    def initializable(self, obs):
        return any_obstruction_on_surface(obs)

    def terminated(self, obs):
        return not any_obstruction_on_surface(obs)

    def reset(self, obs):
        self._remaining = [i for i in range(NUM_OBSTRUCTIONS)
                           if obstruction_overlaps_surface(obs, i)]
        self._start_next(obs)

    def _start_next(self, obs):
        if not self._remaining:
            self._current = None
            return
        i = self._remaining[0]
        self._current = PushOneObstruction(self._primitives, i)
        self._current.reset(obs)

    def step(self, obs):
        if self._current is None:
            return clip_action()
        if self._current.terminated(obs):
            self._remaining.pop(0)
            # Refresh list of remaining obstructions
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

    PH_HIGH = 'high'
    PH_LOW = 'low'
    PH_ORIENT = 'orient'
    PH_EXTEND = 'extend'
    PH_GRASP = 'grasp'
    PH_DONE = 'done'

    def __init__(self, primitives):
        self._primitives = primitives
        self._phase = self.PH_HIGH
        self._hold_cnt = 0
        self._pick_y = 0.30

    def initializable(self, obs):
        return not any_obstruction_on_surface(obs)

    def terminated(self, obs):
        return is_holding_block(obs)

    def reset(self, obs):
        self._phase = self.PH_HIGH
        self._hold_cnt = 0
        b = get_block(obs)
        self._pick_y = pick_robot_y(b['y'], b['height'])

    def step(self, obs):
        robot = get_robot(obs)
        rx, ry = robot['x'], robot['y']
        theta = robot['theta']
        arm = robot['arm_joint']
        b = get_block(obs)
        bx, py = b['x'], self._pick_y

        if self._phase == self.PH_HIGH:
            action = _nav_action(robot, bx, NAV_HIGH_Y, vac=0.0)
            if _at_goal(robot, bx, NAV_HIGH_Y, tol=0.04):
                self._pick_y = pick_robot_y(b['y'], b['height'])
                self._phase = self.PH_LOW
            return action

        elif self._phase == self.PH_LOW:
            action = _nav_action(robot, bx, py, vac=0.0)
            if _at_goal(robot, bx, py, tol=0.025):
                self._phase = self.PH_ORIENT
            return action

        elif self._phase == self.PH_ORIENT:
            err = normalize_angle(-np.pi / 2 - theta)
            dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
            dx_n = np.clip((bx - rx) * K_POS, -MAX_DX, MAX_DX)
            dy_n = np.clip((py - ry) * K_POS, -MAX_DY, MAX_DY)
            if theta_reached(theta):
                self._phase = self.PH_EXTEND
            return clip_action(dx=dx_n, dy=dy_n, dtheta=dtheta, vac=0.0)

        elif self._phase == self.PH_EXTEND:
            if arm_reached(arm, PICK_ARM_JOINT):
                self._phase = self.PH_GRASP
                self._hold_cnt = 0
            dx_n = np.clip((bx - rx) * K_POS, -MAX_DX, MAX_DX)
            dy_n = np.clip((py - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((PICK_ARM_JOINT - arm) * 8.0, -0.1, 0.1)
            return clip_action(dx=dx_n, dy=dy_n, darm=darm, vac=0.0)

        elif self._phase == self.PH_GRASP:
            self._hold_cnt += 1
            if self._hold_cnt >= GRASP_HOLD:
                self._phase = self.PH_DONE
            dx_n = np.clip((bx - rx) * K_POS, -MAX_DX, MAX_DX)
            dy_n = np.clip((py - ry) * K_POS, -MAX_DY, MAX_DY)
            return clip_action(dx=dx_n, dy=dy_n, vac=1.0)

        # PH_DONE: keep vacuum on
        return clip_action(vac=1.0)


class PlaceBlock(Behavior):

    PH_RETRACT = 'retract'
    PH_HIGH = 'high'
    PH_LOW = 'low'
    PH_RELEASE = 'release'

    def __init__(self, primitives):
        self._primitives = primitives
        self._phase = self.PH_RETRACT
        self._place_y = 0.20
        self._hold_cnt = 0

    def initializable(self, obs):
        return is_holding_block(obs)

    def terminated(self, obs):
        return block_on_surface(obs)

    def reset(self, obs):
        self._phase = self.PH_RETRACT
        self._hold_cnt = 0
        b = get_block(obs)
        self._place_y = place_robot_y(b['height'])

    def step(self, obs):
        robot = get_robot(obs)
        rx, ry = robot['x'], robot['y']
        theta = robot['theta']
        arm = robot['arm_joint']
        s = get_surface(obs)
        b = get_block(obs)
        sx = s['x']
        py = self._place_y

        if self._phase == self.PH_RETRACT:
            if arm_reached(arm, RETRACT_ARM):
                self._place_y = place_robot_y(b['height'])
                self._phase = self.PH_HIGH
            darm = np.clip((RETRACT_ARM - arm) * 8.0, -0.1, 0.1)
            err = normalize_angle(-np.pi / 2 - theta)
            dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
            return clip_action(darm=darm, dtheta=dtheta, vac=1.0)

        elif self._phase == self.PH_HIGH:
            action = _nav_action(robot, rx, NAV_HIGH_Y, vac=1.0)
            if _at_goal(robot, rx, NAV_HIGH_Y, tol=0.04):
                self._phase = self.PH_LOW
            return action

        elif self._phase == self.PH_LOW:
            action = _nav_action(robot, sx, py, vac=1.0)
            if _at_goal(robot, sx, py, tol=0.025):
                self._phase = self.PH_RELEASE
                self._hold_cnt = 0
            return action

        elif self._phase == self.PH_RELEASE:
            self._hold_cnt += 1
            dx_n = np.clip((sx - rx) * K_POS, -MAX_DX, MAX_DX)
            return clip_action(dx=dx_n, vac=0.0)

        return clip_action(vac=0.0)
