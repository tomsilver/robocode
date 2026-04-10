"""Behavior classes for Obstruction2D-o4-v0."""
import numpy as np
from behavior import Behavior
from obs_helpers import (
    extract_robot, extract_surface, extract_block, extract_obstruction,
    gripper_tip_pos, obstruction_overlaps_surface, block_is_on_surface,
    block_held, ARM_MIN_JOINT, ARM_MAX_JOINT, NAV_XY_TOL, NAV_THETA_TOL,
    ARM_EXTEND_TOL, NUM_OBSTRUCTIONS, NAV_CLEAR_Y, ROBOT_RADIUS, TABLE_TOP_Y,
)
from act_helpers import (
    navigate_to_pose, robot_goal_for_grasp, follow_path, make_birrt,
    get_rects_for_nav, angle_diff, clip_action, ZERO_ACTION,
    KP_XY, KP_THETA, KP_ARM, GRASP_REACH, NAV_THETA,
    DX_MAX, DY_MAX, WORLD_MIN_X, WORLD_MAX_X, WORLD_MIN_Y, WORLD_MAX_Y,
    choose_grasp_approach, _clamp,
)

# ── Named constants ──────────────────────────────────────────────────────────
HOLD_STEPS          = 5     # steps to hold vacuum after grasping
RELEASE_STEPS       = 8     # steps with vacuum off before declaring released
NAV_PHASE_TOL       = NAV_XY_TOL * 2.5
ANG_PHASE_TOL       = NAV_THETA_TOL * 2.0
EXTEND_VACUUM_DIST  = 0.04  # activate vacuum when arm within this of max
DROP_MARGIN         = 0.05

# ── Phases ───────────────────────────────────────────────────────────────────
PHASE_ROTATE_NAV = "rotate_nav"  # rotate arm to NAV_THETA=pi/2 before navigating
PHASE_NAV    = "nav"
PHASE_ROTATE = "rotate"   # rotate arm to grasp theta before extending
PHASE_EXTEND = "extend"
PHASE_RETRACT = "retract"
PHASE_NAV_DROP = "nav_drop"
PHASE_RELEASE = "release"
PHASE_RETREAT = "retreat"   # move away from drop site after release
PHASE_DONE   = "done"


def _drop_position(surf, idx):
    """Compute drop position for obstruction idx (far from surface)."""
    offsets = [
        (-0.40, +0.30),
        (+0.40, +0.30),
        (-0.40, -0.20),
        (+0.40, -0.20),
    ]
    dx, dy = offsets[idx % len(offsets)]
    x = _clamp(surf["cx"] + dx, WORLD_MIN_X + 0.15, WORLD_MAX_X - 0.15)
    y = _clamp(surf["cy"] + dy, NAV_CLEAR_Y + 0.15, WORLD_MAX_Y - 0.10)
    return x, y


def _plan_nav_path(primitives, obs, start_r, goal_x, goal_y, skip_indices=None, skip_block=False):
    """Plan a BiRRT path keeping theta=NAV_THETA throughout, final waypoint at goal theta."""
    if not primitives:
        return None
    from act_helpers import make_birrt_xy
    rects = get_rects_for_nav(obs, skip_indices=skip_indices or [])
    if skip_block:
        # remove last rect (block)
        rects = rects[:-1]
    birrt = make_birrt_xy(primitives, obs, rects)
    start = np.array([start_r["x"], start_r["y"]])
    goal  = np.array([goal_x, goal_y])
    path_xy = birrt.query(start, goal)
    if path_xy is None:
        return None
    # Build full (x,y,theta) path with fixed NAV_THETA
    path = []
    for pt in path_xy:
        path.append(np.array([pt[0], pt[1], NAV_THETA]))
    return path


class ClearObstruction(Behavior):
    """Pick up obstruction i and move it off the target surface."""

    def __init__(self, obs_idx: int, primitives: dict = None):
        self._i = obs_idx
        self._primitives = primitives or {}
        self._phase = PHASE_NAV
        self._path = None
        self._path_step = 0
        self._hold_count = 0
        self._release_count = 0
        self._drop_x = 0.0
        self._drop_y = 0.0
        self._goal_x = 0.0
        self._goal_y = 0.0
        self._goal_theta = 0.0

    def initializable(self, obs) -> bool:
        return obstruction_overlaps_surface(obs, self._i)

    def terminated(self, obs) -> bool:
        return self._phase == PHASE_DONE

    def reset(self, obs):
        self._phase = PHASE_ROTATE_NAV
        self._path = None
        self._path_step = 0
        self._hold_count = 0
        self._release_count = 0
        ob   = extract_obstruction(obs, self._i)
        surf = extract_surface(obs)
        self._drop_x, self._drop_y = _drop_position(surf, self._i)
        # Choose non-colliding approach direction (pass faces for precise positioning)
        self._goal_x, self._goal_y, self._goal_theta = choose_grasp_approach(
            obs, ob["cx"], ob["cy"], obj_x1=ob["x1"], obj_x2=ob["x2"], obj_y2=ob["y2"], skip_idx=self._i)

    def _plan_path(self, obs):
        r = extract_robot(obs)
        # Include ALL obstacles in nav collision check (including self._i).
        # The robot body must navigate around obstacles; only the arm tip touches the target.
        self._path = _plan_nav_path(
            self._primitives, obs, r,
            self._goal_x, self._goal_y,
            skip_indices=[], skip_block=False)
        self._path_step = 0

    def step(self, obs):
        ob = extract_obstruction(obs, self._i)
        r  = extract_robot(obs)

        if self._phase == PHASE_ROTATE_NAV:
            theta_err = angle_diff(NAV_THETA, r["theta"])
            if abs(theta_err) < ANG_PHASE_TOL:
                self._phase = PHASE_NAV
                self._plan_path(obs)
                return clip_action(0, 0, 0, 0, 0)
            return clip_action(0, 0, theta_err * KP_THETA, 0, 0)

        elif self._phase == PHASE_NAV:
            # Recompute goal in case obstruction moved
            self._goal_x, self._goal_y, self._goal_theta = choose_grasp_approach(
                obs, ob["cx"], ob["cy"], obj_x1=ob["x1"], obj_x2=ob["x2"], obj_y2=ob["y2"], skip_idx=self._i)
            dist     = np.hypot(r["x"] - self._goal_x, r["y"] - self._goal_y)
            if dist < NAV_PHASE_TOL:
                self._phase = PHASE_ROTATE
                return clip_action(0, 0, 0, 0, 0)

            if self._path:
                act, self._path_step = follow_path(self._path, self._path_step, obs)
                return act
            return navigate_to_pose(obs, self._goal_x, self._goal_y,
                                    NAV_THETA, vac=0.0)

        elif self._phase == PHASE_ROTATE:
            # Rotate arm to grasp theta before extending
            theta_err = angle_diff(self._goal_theta, r["theta"])
            if abs(theta_err) < ANG_PHASE_TOL:
                self._phase = PHASE_EXTEND
                return clip_action(0, 0, 0, 0, 0)
            return clip_action(0, 0, theta_err * KP_THETA, 0, 0)

        elif self._phase == PHASE_EXTEND:
            if r["vacuum"] > 0.5:
                self._phase = PHASE_RETRACT
                self._hold_count = 0
                return clip_action(0, 0, 0, 0, 1.0)
            # Fine-position robot AND extend arm
            theta_err = angle_diff(self._goal_theta, r["theta"])
            dx = (self._goal_x - r["x"]) * KP_XY
            dy = (self._goal_y - r["y"]) * KP_XY
            darm   = (ARM_MAX_JOINT - r["arm_joint"]) * KP_ARM
            v = 1.0 if r["arm_joint"] >= ARM_MAX_JOINT - EXTEND_VACUUM_DIST else 0.0
            return clip_action(dx, dy, theta_err * KP_THETA, darm, v)

        elif self._phase == PHASE_RETRACT:
            self._hold_count += 1
            darm = (ARM_MIN_JOINT - r["arm_joint"]) * KP_ARM
            if abs(r["arm_joint"] - ARM_MIN_JOINT) < ARM_EXTEND_TOL and \
               self._hold_count >= HOLD_STEPS:
                self._phase = PHASE_NAV_DROP
            return clip_action(0, 0, 0, darm, 1.0)

        elif self._phase == PHASE_NAV_DROP:
            dist = np.hypot(r["x"] - self._drop_x, r["y"] - self._drop_y)
            if dist < NAV_PHASE_TOL:
                self._phase = PHASE_RELEASE
                self._release_count = 0
            return navigate_to_pose(obs, self._drop_x, self._drop_y,
                                    goal_theta=NAV_THETA, vac=1.0)

        elif self._phase == PHASE_RELEASE:
            self._release_count += 1
            if self._release_count >= RELEASE_STEPS:
                self._phase = PHASE_RETREAT
            return clip_action(0, 0, 0, 0, 0.0)

        elif self._phase == PHASE_RETREAT:
            # Move away from drop site so dropped obstruction doesn't block robot
            retreat_y = _clamp(self._drop_y + 0.30, NAV_CLEAR_Y, WORLD_MAX_Y - 0.05)
            retreat_x = self._drop_x
            dist = np.hypot(r["x"] - retreat_x, r["y"] - retreat_y)
            if dist < NAV_PHASE_TOL:
                self._phase = PHASE_DONE
                return clip_action(0, 0, 0, 0, 0)
            dx = (retreat_x - r["x"]) * KP_XY
            dy = (retreat_y - r["y"]) * KP_XY
            darm = (ARM_MIN_JOINT - r["arm_joint"]) * KP_ARM
            return clip_action(dx, dy, 0, darm, 0.0)

        else:
            return ZERO_ACTION.copy()


class PickBlock(Behavior):
    """Navigate to target block, extend arm, grasp."""

    def __init__(self, primitives: dict = None):
        self._primitives = primitives or {}
        self._phase = PHASE_NAV
        self._path = None
        self._path_step = 0
        self._goal_x = 0.0
        self._goal_y = 0.0
        self._goal_theta = 0.0

    def initializable(self, obs) -> bool:
        return not block_held(obs)

    def terminated(self, obs) -> bool:
        return block_held(obs)

    def reset(self, obs):
        self._phase = PHASE_ROTATE_NAV
        self._path = None
        self._path_step = 0
        blk = extract_block(obs)
        # skip_idx=-1 means skip block from collision check (we want to approach it)
        self._goal_x, self._goal_y, self._goal_theta = choose_grasp_approach(
            obs, blk["cx"], blk["cy"], obj_x1=blk["x1"], obj_x2=blk["x2"], obj_y2=blk["y2"], skip_idx=-1)

    def _plan_path(self, obs):
        r = extract_robot(obs)
        self._path = _plan_nav_path(
            self._primitives, obs, r,
            self._goal_x, self._goal_y,
            skip_indices=[], skip_block=True)
        self._path_step = 0

    def step(self, obs):
        blk = extract_block(obs)
        r   = extract_robot(obs)

        if self._phase == PHASE_ROTATE_NAV:
            theta_err = angle_diff(NAV_THETA, r["theta"])
            if abs(theta_err) < ANG_PHASE_TOL:
                self._phase = PHASE_NAV
                self._plan_path(obs)
                return clip_action(0, 0, 0, 0, 0)
            return clip_action(0, 0, theta_err * KP_THETA, 0, 0)

        elif self._phase == PHASE_NAV:
            self._goal_x, self._goal_y, self._goal_theta = choose_grasp_approach(
                obs, blk["cx"], blk["cy"], obj_x1=blk["x1"], obj_x2=blk["x2"], obj_y2=blk["y2"], skip_idx=-1)
            dist    = np.hypot(r["x"] - self._goal_x, r["y"] - self._goal_y)
            if dist < NAV_PHASE_TOL:
                self._phase = PHASE_ROTATE
                return clip_action(0, 0, 0, 0, 0)
            if self._path:
                act, self._path_step = follow_path(self._path, self._path_step, obs)
                return act
            return navigate_to_pose(obs, self._goal_x, self._goal_y,
                                    NAV_THETA, vac=0.0)

        elif self._phase == PHASE_ROTATE:
            theta_err = angle_diff(self._goal_theta, r["theta"])
            if abs(theta_err) < ANG_PHASE_TOL:
                self._phase = PHASE_EXTEND
                return clip_action(0, 0, 0, 0, 0)
            return clip_action(0, 0, theta_err * KP_THETA, 0, 0)

        elif self._phase == PHASE_EXTEND:
            if block_held(obs):
                return clip_action(0, 0, 0, 0, 1.0)
            # Fine-position robot AND extend arm
            theta_err = angle_diff(self._goal_theta, r["theta"])
            dx = (self._goal_x - r["x"]) * KP_XY
            dy = (self._goal_y - r["y"]) * KP_XY
            darm = (ARM_MAX_JOINT - r["arm_joint"]) * KP_ARM
            v = 1.0 if r["arm_joint"] >= ARM_MAX_JOINT - EXTEND_VACUUM_DIST else 0.0
            return clip_action(dx, dy, theta_err * KP_THETA, darm, v)

        else:
            return clip_action(0, 0, 0, 0, 1.0)


class PlaceBlock(Behavior):
    """Navigate with held block to target surface and release."""

    def __init__(self, primitives: dict = None):
        self._primitives = primitives or {}
        self._phase = PHASE_NAV
        self._path = None
        self._path_step = 0
        self._goal_x = 0.0
        self._goal_y = 0.0
        self._goal_theta = 0.0
        self._release_count = 0

    def initializable(self, obs) -> bool:
        return block_held(obs)

    def terminated(self, obs) -> bool:
        return block_is_on_surface(obs)

    def reset(self, obs):
        self._phase = PHASE_ROTATE_NAV
        self._release_count = 0
        self._path = None
        surf = extract_surface(obs)
        blk  = extract_block(obs)
        # Target: block centered on surface
        block_cx_target = surf["cx"]
        block_cy_target = surf["y2"] + blk["height"] / 2
        block_y2_target = surf["y2"] + blk["height"]
        self._goal_x, self._goal_y, self._goal_theta = choose_grasp_approach(
            obs, block_cx_target, block_cy_target,
            obj_x1=surf["cx"] - blk["width"]/2, obj_x2=surf["cx"] + blk["width"]/2,
            obj_y2=block_y2_target, skip_idx=-1)

    def _plan_path(self, obs):
        r = extract_robot(obs)
        self._path = _plan_nav_path(
            self._primitives, obs, r,
            self._goal_x, self._goal_y,
            skip_indices=list(range(NUM_OBSTRUCTIONS)), skip_block=True)
        self._path_step = 0

    def step(self, obs):
        r    = extract_robot(obs)
        surf = extract_surface(obs)
        blk  = extract_block(obs)

        if self._phase == PHASE_ROTATE_NAV:
            theta_err = angle_diff(NAV_THETA, r["theta"])
            darm = (ARM_MIN_JOINT - r["arm_joint"]) * KP_ARM  # retract while rotating to lift block off table
            arm_done = abs(r["arm_joint"] - ARM_MIN_JOINT) < ARM_EXTEND_TOL
            if abs(theta_err) < ANG_PHASE_TOL and arm_done:
                self._phase = PHASE_NAV
                self._plan_path(obs)
                return clip_action(0, 0, 0, darm, 1.0)
            return clip_action(0, 0, theta_err * KP_THETA, darm, 1.0)

        elif self._phase == PHASE_NAV:
            dist    = np.hypot(r["x"] - self._goal_x, r["y"] - self._goal_y)
            if dist < NAV_PHASE_TOL:
                self._phase = PHASE_ROTATE
                return clip_action(0, 0, 0, 0, 1.0)
            if self._path:
                act, self._path_step = follow_path(self._path, self._path_step, obs, vac=1.0)
                return act
            return navigate_to_pose(obs, self._goal_x, self._goal_y,
                                    NAV_THETA, vac=1.0)

        elif self._phase == PHASE_ROTATE:
            theta_err = angle_diff(self._goal_theta, r["theta"])
            if abs(theta_err) < ANG_PHASE_TOL:
                self._phase = PHASE_EXTEND
                return clip_action(0, 0, 0, 0, 1.0)
            return clip_action(0, 0, theta_err * KP_THETA, 0, 1.0)

        elif self._phase == PHASE_EXTEND:
            # Fine-position robot AND extend arm
            theta_err = angle_diff(self._goal_theta, r["theta"])
            dx = (self._goal_x - r["x"]) * KP_XY
            dy = (self._goal_y - r["y"]) * KP_XY
            darm = (ARM_MAX_JOINT - r["arm_joint"]) * KP_ARM
            if abs(r["arm_joint"] - ARM_MAX_JOINT) < ARM_EXTEND_TOL:
                self._phase = PHASE_RELEASE
                self._release_count = 0
            return clip_action(dx, dy, theta_err * KP_THETA, darm, 1.0)

        elif self._phase == PHASE_RELEASE:
            self._release_count += 1
            if self._release_count >= RELEASE_STEPS:
                self._phase = PHASE_DONE
            return clip_action(0, 0, 0, 0, 0.0)

        else:
            return ZERO_ACTION.copy()
