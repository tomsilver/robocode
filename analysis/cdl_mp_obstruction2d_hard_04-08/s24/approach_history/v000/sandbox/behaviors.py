"""Behavior classes for Obstruction2D-o4-v0."""
import numpy as np
from behavior import Behavior
from obs_helpers import (
    extract_robot, extract_surface, extract_block, extract_obstruction,
    gripper_tip_pos, obstruction_overlaps_surface, block_is_on_surface,
    block_held, ARM_MIN_JOINT, ARM_MAX_JOINT, NAV_XY_TOL, NAV_THETA_TOL,
    ARM_EXTEND_TOL, GRASP_TOL, NUM_OBSTRUCTIONS,
)
from act_helpers import (
    navigate_to_pose, extend_arm_to, face_target, robot_goal_for_grasp,
    follow_path, make_birrt, get_static_rects, angle_diff, clip_action,
    KP_XY, KP_THETA, KP_ARM, ZERO_ACTION, DX_MAX, DY_MAX,
    GRASP_REACH, ARM_MAX_JOINT as ACT_ARM_MAX,
)

# ── Named constants ──────────────────────────────────────────────────────────
DROP_OFFSETS = [(0.35, 0.30), (-0.35, 0.30), (0.35, -0.30), (-0.35, -0.30)]
DROP_MARGIN_X_MIN = 0.15
DROP_MARGIN_X_MAX = 1.50
DROP_MARGIN_Y_MIN = 0.20
DROP_MARGIN_Y_MAX = 0.85

HOLD_STEPS    = 3
RELEASE_STEPS = 5
NAV_PHASE_TOL = NAV_XY_TOL * 2.5   # coarser tolerance for phase transitions
ANG_PHASE_TOL = NAV_THETA_TOL * 2.0

# ── Phase names ──────────────────────────────────────────────────────────────
PHASE_NAV    = "nav"
PHASE_EXTEND = "extend"
PHASE_RETRACT = "retract"
PHASE_NAV_DROP = "nav_drop"
PHASE_RELEASE = "release"
PHASE_DONE   = "done"


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _drop_zone(surf, i):
    dx, dy = DROP_OFFSETS[i % len(DROP_OFFSETS)]
    x = _clamp(surf["x"] + dx, DROP_MARGIN_X_MIN, DROP_MARGIN_X_MAX)
    y = _clamp(surf["y"] + dy, DROP_MARGIN_Y_MIN, DROP_MARGIN_Y_MAX)
    return x, y


class ClearObstruction(Behavior):
    """Pick up one obstruction and move it off the target surface."""

    def __init__(self, obs_idx: int):
        self._i = obs_idx
        self._phase = PHASE_NAV
        self._hold_count = 0
        self._release_count = 0
        self._drop_x = 0.0
        self._drop_y = 0.0

    def initializable(self, obs) -> bool:
        return obstruction_overlaps_surface(obs, self._i)

    def terminated(self, obs) -> bool:
        return self._phase == PHASE_DONE

    def reset(self, obs):
        self._phase = PHASE_NAV
        self._hold_count = 0
        self._release_count = 0
        surf = extract_surface(obs)
        self._drop_x, self._drop_y = _drop_zone(surf, self._i)

    def step(self, obs):
        ob  = extract_obstruction(obs, self._i)
        ox, oy = ob["x"], ob["y"]
        r   = extract_robot(obs)

        if self._phase == PHASE_NAV:
            # Navigate to position where arm can reach obstruction
            theta_to_obj = np.arctan2(oy - r["y"], ox - r["x"])
            goal_x, goal_y = robot_goal_for_grasp(ox, oy, theta_to_obj)
            dist = np.hypot(r["x"] - goal_x, r["y"] - goal_y)
            ang_err = abs(angle_diff(theta_to_obj, r["theta"]))
            if dist < NAV_PHASE_TOL and ang_err < ANG_PHASE_TOL:
                self._phase = PHASE_EXTEND
            return navigate_to_pose(obs, goal_x, goal_y, theta_to_obj, vac=0.0)

        elif self._phase == PHASE_EXTEND:
            # If we grabbed something (vacuum triggered), go to retract
            if obs[6] > 0.5:
                self._phase = PHASE_RETRACT
                self._hold_count = 0
                return clip_action(0.0, 0.0, 0.0, 0.0, 1.0)
            # Extend arm and turn vacuum on as we approach
            darm = (ARM_MAX_JOINT - r["arm_joint"]) * KP_ARM
            # Turn vacuum on to trigger suction once arm is extended enough
            v = 1.0 if r["arm_joint"] >= ARM_MAX_JOINT - ARM_EXTEND_TOL * 3 else 0.0
            return clip_action(0.0, 0.0, 0.0, darm, v)

        elif self._phase == PHASE_RETRACT:
            self._hold_count += 1
            darm = (ARM_MIN_JOINT - r["arm_joint"]) * KP_ARM
            if abs(r["arm_joint"] - ARM_MIN_JOINT) < ARM_EXTEND_TOL and self._hold_count >= HOLD_STEPS:
                self._phase = PHASE_NAV_DROP
            return clip_action(0.0, 0.0, 0.0, darm, 1.0)

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
            return clip_action(0.0, 0.0, 0.0, 0.0, 0.0)

        else:
            return ZERO_ACTION.copy()


class PickBlock(Behavior):
    """Navigate to the target block, extend arm, and grasp it."""

    def __init__(self):
        self._phase = PHASE_NAV

    def initializable(self, obs) -> bool:
        return not block_held(obs)

    def terminated(self, obs) -> bool:
        return block_held(obs)

    def reset(self, obs):
        self._phase = PHASE_NAV

    def step(self, obs):
        blk = extract_block(obs)
        bx, by = blk["x"], blk["y"]
        r = extract_robot(obs)

        if self._phase == PHASE_NAV:
            theta_to_obj = np.arctan2(by - r["y"], bx - r["x"])
            goal_x, goal_y = robot_goal_for_grasp(bx, by, theta_to_obj)
            dist = np.hypot(r["x"] - goal_x, r["y"] - goal_y)
            ang_err = abs(angle_diff(theta_to_obj, r["theta"]))
            if dist < NAV_PHASE_TOL and ang_err < ANG_PHASE_TOL:
                self._phase = PHASE_EXTEND
            return navigate_to_pose(obs, goal_x, goal_y, theta_to_obj, vac=0.0)

        elif self._phase == PHASE_EXTEND:
            if block_held(obs):
                return clip_action(0.0, 0.0, 0.0, 0.0, 1.0)
            darm = (ARM_MAX_JOINT - r["arm_joint"]) * KP_ARM
            v = 1.0 if r["arm_joint"] >= ARM_MAX_JOINT - ARM_EXTEND_TOL * 3 else 0.0
            return clip_action(0.0, 0.0, 0.0, darm, v)

        else:
            return clip_action(0.0, 0.0, 0.0, 0.0, 1.0)


class PlaceBlock(Behavior):
    """Navigate with held block to target surface and release."""

    def __init__(self):
        self._phase = PHASE_NAV
        self._release_count = 0

    def initializable(self, obs) -> bool:
        return block_held(obs)

    def terminated(self, obs) -> bool:
        return block_is_on_surface(obs)

    def reset(self, obs):
        self._phase = PHASE_NAV
        self._release_count = 0

    def step(self, obs):
        surf = extract_surface(obs)
        sx, sy = surf["x"], surf["y"]
        # Target: place block at surface center. Block needs to sit just above surface.
        # Surface center y + some offset so block rests on top
        target_y = sy + surf["height"] / 2.0 + 0.01
        r = extract_robot(obs)

        if self._phase == PHASE_NAV:
            theta_to_surf = np.arctan2(target_y - r["y"], sx - r["x"])
            goal_x, goal_y = robot_goal_for_grasp(sx, target_y, theta_to_surf)
            dist = np.hypot(r["x"] - goal_x, r["y"] - goal_y)
            ang_err = abs(angle_diff(theta_to_surf, r["theta"]))
            if dist < NAV_PHASE_TOL and ang_err < ANG_PHASE_TOL:
                self._phase = PHASE_EXTEND
            return navigate_to_pose(obs, goal_x, goal_y, theta_to_surf, vac=1.0)

        elif self._phase == PHASE_EXTEND:
            darm = (ARM_MAX_JOINT - r["arm_joint"]) * KP_ARM
            if abs(r["arm_joint"] - ARM_MAX_JOINT) < ARM_EXTEND_TOL:
                self._phase = PHASE_RELEASE
                self._release_count = 0
            return clip_action(0.0, 0.0, 0.0, darm, 1.0)

        elif self._phase == PHASE_RELEASE:
            self._release_count += 1
            if self._release_count >= RELEASE_STEPS:
                self._phase = PHASE_DONE
            return clip_action(0.0, 0.0, 0.0, 0.0, 0.0)

        else:
            return ZERO_ACTION.copy()
