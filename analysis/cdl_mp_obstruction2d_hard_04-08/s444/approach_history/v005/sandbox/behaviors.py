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
    ROB_RADIUS, OBJ_TOP_OFFSET,
)
from act_helpers import (
    clip_action, toward_pos, control_arm, normalize_angle,
    pos_reached, theta_reached, arm_reached,
    plan_path, get_drop_xy,
    PICK_ARM_JOINT as ACT_PICK_ARM, RETRACT_ARM,
    NAV_HIGH_Y, MAX_DTHETA, K_THETA, POS_TOL,
    MAX_DX, MAX_DY, K_POS,
)

# ---- Phases ----
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

GRASP_HOLD = 12
RELEASE_HOLD = 8
NAV_TOL = 0.025

# Obstruction pick constants:
# min_y above obs = obs_top + 0.17 (empirical).
# We need pick_y > min_y, so OBS_PICK_OFFSET > 0.17.
# Suction contact: arm_joint = pick_y - suction_extra - obs_top
#                             = obs_top + OBS_PICK_OFFSET - 0.015 - obs_top
#                             = OBS_PICK_OFFSET - 0.015 = 0.175
OBS_PICK_OFFSET = 0.19   # robot_y = obs_top + OBS_PICK_OFFSET
OBS_PICK_ARM = 0.175     # arm extension so suction reaches inside obs


def _nav_action(robot, goal_x, goal_y, vac=0.0):
    return toward_pos(robot['x'], robot['y'], goal_x, goal_y,
                      vac=vac, cur_theta=robot['theta'])


def _at_goal(robot, goal_x, goal_y, tol=NAV_TOL):
    return pos_reached(robot['x'], robot['y'], goal_x, goal_y, tol=tol)


class PickOneObstruction(Behavior):
    """Pick obstruction i and move it to the drop zone."""

    def __init__(self, primitives, obs_idx):
        self._primitives = primitives
        self._i = obs_idx
        self._phase = PH_NAV_HIGH
        self._hold_cnt = 0
        self._pick_y = 0.40
        self._drop_x = 1.35
        self._drop_y = NAV_HIGH_Y

    def initializable(self, obs): return True

    def terminated(self, obs):
        return not obstruction_overlaps_surface(obs, self._i)

    def reset(self, obs):
        self._phase = PH_NAV_HIGH
        self._hold_cnt = 0
        o = get_obstruction(obs, self._i)
        obs_top = o['y'] + o['height'] / 2
        self._pick_y = obs_top + OBS_PICK_OFFSET
        dx, dy = get_drop_xy(self._i)
        self._drop_x = dx
        self._drop_y = NAV_HIGH_Y

    def _pick_x(self, obs):
        return get_obstruction(obs, self._i)['x']

    def step(self, obs):
        robot = get_robot(obs)
        rx, ry = robot['x'], robot['y']
        theta = robot['theta']
        arm = robot['arm_joint']

        px = self._pick_x(obs)
        py = self._pick_y

        if self._phase == PH_NAV_HIGH:
            action = _nav_action(robot, px, NAV_HIGH_Y, vac=0.0)
            # Wait for tight x-alignment before descent
            if abs(rx - px) < 0.025 and abs(ry - NAV_HIGH_Y) < 0.05:
                self._phase = PH_NAV_LOW
            return action

        elif self._phase == PH_NAV_LOW:
            action = _nav_action(robot, px, py, vac=0.0)
            if _at_goal(robot, px, py, tol=0.025):
                self._phase = PH_ORIENT
            return action

        elif self._phase == PH_ORIENT:
            err = normalize_angle(-np.pi / 2 - theta)
            dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
            dx_n = np.clip((px - rx) * K_POS, -MAX_DX, MAX_DX)
            dy_n = np.clip((py - ry) * K_POS, -MAX_DY, MAX_DY)
            if theta_reached(theta):
                self._phase = PH_EXTEND
            return clip_action(dx=dx_n, dy=dy_n, dtheta=dtheta, vac=0.0)

        elif self._phase == PH_EXTEND:
            if arm_reached(arm, OBS_PICK_ARM):
                self._phase = PH_GRASP
                self._hold_cnt = 0
            dx_n = np.clip((px - rx) * K_POS, -MAX_DX, MAX_DX)
            dy_n = np.clip((py - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((OBS_PICK_ARM - arm) * 8.0, -0.1, 0.1)
            return clip_action(dx=dx_n, dy=dy_n, darm=darm, vac=0.0)

        elif self._phase == PH_GRASP:
            self._hold_cnt += 1
            if self._hold_cnt >= GRASP_HOLD:
                self._phase = PH_RETRACT
            dx_n = np.clip((px - rx) * K_POS, -MAX_DX, MAX_DX)
            dy_n = np.clip((py - ry) * K_POS, -MAX_DY, MAX_DY)
            return clip_action(dx=dx_n, dy=dy_n, vac=1.0)

        elif self._phase == PH_RETRACT:
            if arm_reached(arm, RETRACT_ARM):
                self._phase = PH_NAV_HIGH2
            darm = np.clip((RETRACT_ARM - arm) * 8.0, -0.1, 0.1)
            return clip_action(darm=darm, vac=1.0)

        elif self._phase == PH_NAV_HIGH2:
            action = _nav_action(robot, rx, NAV_HIGH_Y, vac=1.0)
            if _at_goal(robot, rx, NAV_HIGH_Y, tol=0.05):
                self._phase = PH_NAV_LOW2
            return action

        elif self._phase == PH_NAV_LOW2:
            action = _nav_action(robot, self._drop_x, NAV_HIGH_Y, vac=1.0)
            if _at_goal(robot, self._drop_x, NAV_HIGH_Y, tol=0.05):
                self._phase = PH_RELEASE
                self._hold_cnt = 0
            return action

        elif self._phase == PH_RELEASE:
            self._hold_cnt += 1
            if self._hold_cnt >= RELEASE_HOLD:
                self._phase = PH_DONE
            return clip_action(vac=0.0)

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
        self._current = PickOneObstruction(self._primitives, i)
        self._current.reset(obs)

    def step(self, obs):
        if self._current is None:
            return clip_action()
        if self._current.terminated(obs):
            self._remaining.pop(0)
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
