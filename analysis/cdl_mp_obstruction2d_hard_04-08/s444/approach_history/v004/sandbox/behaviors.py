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

GRASP_HOLD = 8
RELEASE_HOLD = 8
NAV_TOL = 0.025

# Pushing constants
PUSH_Y = 0.26        # altitude for sideways sweep push
PUSH_TOL = 0.05      # position tolerance during push
SWEEP_LEFT = 0.15    # world x left boundary
SWEEP_RIGHT = 1.50   # world x right boundary


def _nav_action_arm_up(robot, goal_x, goal_y, vac=0.0):
    """Navigate with arm pointing UP (avoids table/obs collision below)."""
    rx, ry = robot['x'], robot['y']
    theta = robot['theta']
    dx = np.clip((goal_x - rx) * K_POS, -MAX_DX, MAX_DX)
    dy = np.clip((goal_y - ry) * K_POS, -MAX_DY, MAX_DY)
    err = normalize_angle(np.pi / 2 - theta)
    dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
    return clip_action(dx=dx, dy=dy, dtheta=dtheta, vac=vac)


def _nav_action(robot, goal_x, goal_y, vac=0.0):
    return toward_pos(robot['x'], robot['y'], goal_x, goal_y,
                      vac=vac, cur_theta=robot['theta'])


def _at_goal(robot, goal_x, goal_y, tol=NAV_TOL):
    return pos_reached(robot['x'], robot['y'], goal_x, goal_y, tol=tol)


class ClearAllObstructions(Behavior):
    """Sweep robot left-to-right (and back) at push altitude to clear surface.

    Strategy:
    1. Navigate to far left at NAV_HIGH_Y (arm up)
    2. Descend to PUSH_Y (arm up, so arm doesn't interfere)
    3. Sweep RIGHT: robot base pushes all surface obstructions to the right
    4. If not done, sweep LEFT: push remaining obstructions to the left
    5. Repeat until terminated
    """

    PH_NAV_HIGH = 'nav_high'
    PH_NAV_LOW = 'nav_low'
    PH_SWEEP = 'sweep'

    def __init__(self, primitives):
        self._primitives = primitives
        self._phase = self.PH_NAV_HIGH
        self._start_x = SWEEP_LEFT
        self._target_x = SWEEP_RIGHT
        self._sweep_count = 0

    def initializable(self, obs):
        return any_obstruction_on_surface(obs)

    def terminated(self, obs):
        return not any_obstruction_on_surface(obs)

    def reset(self, obs):
        self._phase = self.PH_NAV_HIGH
        self._sweep_count = 0
        # Start from left, sweep right
        self._start_x = SWEEP_LEFT
        self._target_x = SWEEP_RIGHT

    def step(self, obs):
        robot = get_robot(obs)
        rx, ry = robot['x'], robot['y']

        if self._phase == self.PH_NAV_HIGH:
            action = _nav_action_arm_up(robot, self._start_x, NAV_HIGH_Y)
            if _at_goal(robot, self._start_x, NAV_HIGH_Y, tol=0.05):
                self._phase = self.PH_NAV_LOW
            return action

        elif self._phase == self.PH_NAV_LOW:
            action = _nav_action_arm_up(robot, self._start_x, PUSH_Y)
            if _at_goal(robot, self._start_x, PUSH_Y, tol=PUSH_TOL):
                self._phase = self.PH_SWEEP
            return action

        elif self._phase == self.PH_SWEEP:
            # Keep sweeping toward target_x at PUSH_Y
            action = _nav_action_arm_up(robot, self._target_x, PUSH_Y)
            # Check if we've reached the far end (robot stuck or at target)
            if abs(rx - self._target_x) < 0.08:
                # Reverse sweep direction
                self._sweep_count += 1
                if self._target_x == SWEEP_RIGHT:
                    self._start_x = SWEEP_RIGHT
                    self._target_x = SWEEP_LEFT
                else:
                    self._start_x = SWEEP_LEFT
                    self._target_x = SWEEP_RIGHT
                self._phase = self.PH_NAV_HIGH
            return action

        return clip_action()


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
