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
PHASE_NAV    = "nav"
PHASE_EXTEND = "extend"
PHASE_RETRACT = "retract"
PHASE_NAV_DROP = "nav_drop"
PHASE_RELEASE = "release"
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
        self._phase = PHASE_NAV
        self._path = None
        self._path_step = 0
        self._hold_count = 0
        self._release_count = 0
        ob   = extract_obstruction(obs, self._i)
        surf = extract_surface(obs)
        self._drop_x, self._drop_y = _drop_position(surf, self._i)
        # Choose non-colliding approach direction
        self._goal_x, self._goal_y, self._goal_theta = choose_grasp_approach(
            obs, ob["cx"], ob["cy"], skip_idx=self._i)
        # Plan BiRRT path if primitives available
        if self._primitives:
            rects = get_rects_for_nav(obs, skip_indices=[self._i])
            birrt = make_birrt(self._primitives, obs, rects)
            r = extract_robot(obs)
            start = np.array([r["x"], r["y"], r["theta"]])
            goal  = np.array([self._goal_x, self._goal_y, self._goal_theta])
            self._path = birrt.query(start, goal)
            self._path_step = 0

    def step(self, obs):
        ob = extract_obstruction(obs, self._i)
        r  = extract_robot(obs)

        if self._phase == PHASE_NAV:
            # Recompute goal in case obstruction moved
            self._goal_x, self._goal_y, self._goal_theta = choose_grasp_approach(
                obs, ob["cx"], ob["cy"], skip_idx=self._i)
            dist     = np.hypot(r["x"] - self._goal_x, r["y"] - self._goal_y)
            ang_err  = abs(angle_diff(self._goal_theta, r["theta"]))
            if dist < NAV_PHASE_TOL and ang_err < ANG_PHASE_TOL:
                self._phase = PHASE_EXTEND
                return clip_action(0, 0, 0, 0, 0)

            if self._path:
                act, self._path_step = follow_path(self._path, self._path_step, obs)
                return act
            return navigate_to_pose(obs, self._goal_x, self._goal_y,
                                    self._goal_theta, vac=0.0)

        elif self._phase == PHASE_EXTEND:
            if r["vacuum"] > 0.5:
                self._phase = PHASE_RETRACT
                self._hold_count = 0
                return clip_action(0, 0, 0, 0, 1.0)
            # Keep arm aligned, extend
            theta_err = angle_diff(self._goal_theta, r["theta"])
            dtheta = theta_err * KP_THETA
            darm   = (ARM_MAX_JOINT - r["arm_joint"]) * KP_ARM
            v = 1.0 if r["arm_joint"] >= ARM_MAX_JOINT - EXTEND_VACUUM_DIST else 0.0
            return clip_action(0, 0, dtheta, darm, v)

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
            return navigate_to_pose(obs, self._drop_x, self._drop_y, vac=1.0)

        elif self._phase == PHASE_RELEASE:
            self._release_count += 1
            if self._release_count >= RELEASE_STEPS:
                self._phase = PHASE_DONE
            return clip_action(0, 0, 0, 0, 0.0)

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
        self._phase = PHASE_NAV
        self._path = None
        self._path_step = 0
        blk = extract_block(obs)
        # skip_idx=None means use all obstructions but skip block itself
        self._goal_x, self._goal_y, self._goal_theta = choose_grasp_approach(
            obs, blk["cx"], blk["cy"], skip_idx=-1)  # -1 means skip block from check
        if self._primitives:
            rects = get_rects_for_nav(obs, skip_indices=[])  # avoid all obs
            # Remove block from rects (we want to approach it)
            rects = rects[:-1]
            birrt = make_birrt(self._primitives, obs, rects)
            r = extract_robot(obs)
            start = np.array([r["x"], r["y"], r["theta"]])
            goal  = np.array([self._goal_x, self._goal_y, self._goal_theta])
            self._path = birrt.query(start, goal)
            self._path_step = 0

    def step(self, obs):
        blk = extract_block(obs)
        r   = extract_robot(obs)

        if self._phase == PHASE_NAV:
            self._goal_x, self._goal_y, self._goal_theta = choose_grasp_approach(
                obs, blk["cx"], blk["cy"], skip_idx=-1)
            dist    = np.hypot(r["x"] - self._goal_x, r["y"] - self._goal_y)
            ang_err = abs(angle_diff(self._goal_theta, r["theta"]))
            if dist < NAV_PHASE_TOL and ang_err < ANG_PHASE_TOL:
                self._phase = PHASE_EXTEND
                return clip_action(0, 0, 0, 0, 0)
            if self._path:
                act, self._path_step = follow_path(self._path, self._path_step, obs)
                return act
            return navigate_to_pose(obs, self._goal_x, self._goal_y,
                                    self._goal_theta, vac=0.0)

        elif self._phase == PHASE_EXTEND:
            if block_held(obs):
                return clip_action(0, 0, 0, 0, 1.0)
            theta_err = angle_diff(self._goal_theta, r["theta"])
            darm = (ARM_MAX_JOINT - r["arm_joint"]) * KP_ARM
            v = 1.0 if r["arm_joint"] >= ARM_MAX_JOINT - EXTEND_VACUUM_DIST else 0.0
            return clip_action(0, 0, theta_err * KP_THETA, darm, v)

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
        self._phase = PHASE_NAV
        self._release_count = 0
        self._path = None
        surf = extract_surface(obs)
        blk  = extract_block(obs)
        # Target: place block so its bottom-left is at:
        #   x = surf_cx - block_w/2  (centered on surface)
        #   y = surf_y2 (block bottom at surface top)
        target_x = surf["cx"] - blk["width"] / 2  # bottom-left x of block after placement
        target_y = surf["y2"]                       # block rests on surface top
        # Block center when placed: (target_x + w/2, target_y + h/2) = (surf_cx, surf_y2 + h/2)
        block_cx_target = surf["cx"]
        block_cy_target = surf["y2"] + blk["height"] / 2
        # Robot needs suction zone to reach block_cx_target, block_cy_target
        self._goal_x, self._goal_y, self._goal_theta = choose_grasp_approach(
            obs, block_cx_target, block_cy_target, skip_idx=-1)  # ignore block collision (held)

        if self._primitives:
            rects = []  # when carrying block, no obstacles to avoid except walls
            birrt = make_birrt(self._primitives, obs, rects)
            r_cur = extract_robot(obs)
            start = np.array([r_cur["x"], r_cur["y"], r_cur["theta"]])
            goal  = np.array([self._goal_x, self._goal_y, self._goal_theta])
            self._path = birrt.query(start, goal)
            self._path_step = 0

    def step(self, obs):
        r    = extract_robot(obs)
        surf = extract_surface(obs)
        blk  = extract_block(obs)

        if self._phase == PHASE_NAV:
            dist    = np.hypot(r["x"] - self._goal_x, r["y"] - self._goal_y)
            ang_err = abs(angle_diff(self._goal_theta, r["theta"]))
            if dist < NAV_PHASE_TOL and ang_err < ANG_PHASE_TOL:
                self._phase = PHASE_EXTEND
                return clip_action(0, 0, 0, 0, 1.0)
            if self._path:
                act, self._path_step = follow_path(self._path, self._path_step, obs, vac=1.0)
                return act
            return navigate_to_pose(obs, self._goal_x, self._goal_y,
                                    self._goal_theta, vac=1.0)

        elif self._phase == PHASE_EXTEND:
            theta_err = angle_diff(self._goal_theta, r["theta"])
            darm = (ARM_MAX_JOINT - r["arm_joint"]) * KP_ARM
            if abs(r["arm_joint"] - ARM_MAX_JOINT) < ARM_EXTEND_TOL:
                self._phase = PHASE_RELEASE
                self._release_count = 0
            return clip_action(0, 0, theta_err * KP_THETA, darm, 1.0)

        elif self._phase == PHASE_RELEASE:
            self._release_count += 1
            if self._release_count >= RELEASE_STEPS:
                self._phase = PHASE_DONE
            return clip_action(0, 0, 0, 0, 0.0)

        else:
            return ZERO_ACTION.copy()
