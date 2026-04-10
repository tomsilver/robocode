"""All behavior classes. Each implements Behavior from behavior.py."""
import numpy as np
from behavior import Behavior
from obs_helpers import (
    extract_robot, extract_target_surface, extract_target_block,
    extract_obstruction, rect_center_x, rect_top,
    obstruction_on_surface, is_block_grasped, block_is_on_surface,
    SUCTION_OFFSET, NUM_OBSTRUCTIONS, TABLE_TOP_Y,
)
from act_helpers import (
    make_action, move_toward, rotate_toward, arm_toward,
    pick_robot_y, place_robot_y,
    ARM_MIN, ARM_PICK, TARGET_THETA,
    SAFE_NAV_HEIGHT, POS_TOL, ARM_TOL, THETA_TOL,
    GRASP_WAIT_STEPS, MAX_PHASE_STEPS,
)

# NOTE: arm can ONLY be extended when vacuum is OFF.
# With vacuum ON, arm is locked at current position.
# PickAndDrop: carry with ARM_PICK extended (no retract), lower to pick_y, release.
# PickAndPlace: retract arm after grasp (allowed), carry with ARM_MIN.

# Safe carry height for PickAndDrop (arm extended at ARM_PICK=0.175)
# Need carried-obs bottom to clear target block on table (max top ≈ 0.1+0.15=0.25)
# obs_bottom = robot_y - ARM_PICK - obs_h - SUCTION_OFFSET
# For obs_h=0.15: robot_y > 0.25 + 0.15 + 0.175 + 0.015 = 0.59 → use 0.65
SAFE_CARRY_HEIGHT_DROP = 0.65


# ─── Phase constants for PickAndDrop ────────────────────────────────────────
_D_RISE        = 0   # retract arm, align θ, rise to safe height
_D_ALIGN_X     = 1   # move to pick_x at safe height
_D_LOWER       = 2   # descend to pick_y (arm=ARM_MIN)
_D_EXTEND      = 3   # extend arm to ARM_PICK (vac OFF)
_D_GRASP       = 4   # vacuum ON, wait
_D_RISE_POST   = 5   # rise to SAFE_CARRY_HEIGHT (arm stays ARM_PICK, vac ON)
_D_CARRY       = 6   # navigate to drop_x at SAFE_CARRY_HEIGHT
_D_LOWER_CARRY = 7   # descend to pick_y at drop_x (arm ARM_PICK, vac ON)
_D_RELEASE     = 8   # vacuum OFF → object lands at table level
_D_DONE        = 9

# ─── Phase constants for PickAndPlace ───────────────────────────────────────
_P_RISE        = 0
_P_ALIGN_X     = 1
_P_LOWER       = 2
_P_EXTEND      = 3
_P_GRASP       = 4
_P_RETRACT     = 5   # retract arm (allowed with vac ON)
_P_RISE_POST   = 6
_P_CARRY       = 7
_P_LOWER_PLACE = 8
_P_EXTEND_PL   = 9   # NOT possible with vac ON — must release first!
_P_RELEASE     = 10
_P_DONE        = 11


class PickAndDrop(Behavior):
    """Pick obstruction i and deposit it at drop_x at table level.

    Carries with arm EXTENDED (ARM_PICK) so that:
      release at (drop_x, pick_y, arm=ARM_PICK) → obs_bottom = TABLE_TOP ✓
    """

    def __init__(self, obs_idx: int, drop_x: float):
        self._obs_idx = obs_idx
        self._drop_x  = drop_x
        self._phase   = _D_DONE
        self._timer   = 0
        self._step_count = 0
        self._pick_x  = 0.0
        self._pick_y  = 0.0

    def reset(self, obs):
        self._phase = _D_RISE
        self._timer = 0
        self._step_count = 0
        self._update_pick_target(obs)

    def _update_pick_target(self, obs):
        obj = extract_obstruction(obs, self._obs_idx)
        self._pick_x = rect_center_x(obj)
        self._pick_y = pick_robot_y(rect_top(obj))

    def initializable(self, obs) -> bool:
        return True

    def terminated(self, obs) -> bool:
        # Phase-based: done only after releasing vacuum
        return self._phase >= _D_RELEASE

    def step(self, obs) -> np.ndarray:
        r = extract_robot(obs)
        rx, ry, rth = r['x'], r['y'], r['theta']
        arm = r['arm_joint']

        self._step_count += 1
        if self._step_count > MAX_PHASE_STEPS:
            self._update_pick_target(obs)
            self._phase = _D_RISE
            self._step_count = 0

        if self._phase == _D_RISE:
            dx, dy = move_toward(rx, ry, rx, SAFE_NAV_HEIGHT)
            dth    = rotate_toward(rth)
            darm   = arm_toward(arm, ARM_MIN)
            done   = (ry >= SAFE_NAV_HEIGHT - POS_TOL and
                      abs(arm - ARM_MIN) < ARM_TOL and
                      abs(rth - TARGET_THETA) < THETA_TOL)
            if done:
                self._phase = _D_ALIGN_X
                self._step_count = 0
            return make_action(dx, dy, dth, darm, 0.0)

        elif self._phase == _D_ALIGN_X:
            dx, dy = move_toward(rx, ry, self._pick_x, SAFE_NAV_HEIGHT)
            dth    = rotate_toward(rth)
            darm   = arm_toward(arm, ARM_MIN)
            done   = (abs(rx - self._pick_x) < POS_TOL and
                      ry >= SAFE_NAV_HEIGHT - POS_TOL)
            if done:
                self._phase = _D_LOWER
                self._step_count = 0
            return make_action(dx, dy, dth, darm, 0.0)

        elif self._phase == _D_LOWER:
            self._update_pick_target(obs)
            dx, dy = move_toward(rx, ry, self._pick_x, self._pick_y)
            dth    = rotate_toward(rth)
            darm   = arm_toward(arm, ARM_MIN)
            done   = (abs(rx - self._pick_x) < POS_TOL and
                      abs(ry - self._pick_y) < POS_TOL)
            if done:
                self._phase = _D_EXTEND
                self._step_count = 0
            return make_action(dx, dy, dth, darm, 0.0)

        elif self._phase == _D_EXTEND:
            # Extend arm (vacuum OFF — arm can only extend when not holding)
            dth  = rotate_toward(rth)
            darm = arm_toward(arm, ARM_PICK)
            done = abs(arm - ARM_PICK) < ARM_TOL
            if done:
                self._phase = _D_GRASP
                self._timer = 0
                self._step_count = 0
            return make_action(0, 0, dth, darm, 0.0)

        elif self._phase == _D_GRASP:
            # Turn on vacuum, wait for grasp
            self._timer += 1
            dth  = rotate_toward(rth)
            done = self._timer >= GRASP_WAIT_STEPS
            if done:
                self._phase = _D_RISE_POST
                self._step_count = 0
            return make_action(0, 0, dth, 0, 1.0)

        elif self._phase == _D_RISE_POST:
            # Rise to SAFE_CARRY_HEIGHT with arm EXTENDED (do NOT retract arm)
            # Arm locked by vacuum — stays at ARM_PICK automatically
            dx, dy = move_toward(rx, ry, rx, SAFE_CARRY_HEIGHT_DROP)
            dth    = rotate_toward(rth)
            done   = ry >= SAFE_CARRY_HEIGHT_DROP - POS_TOL
            if done:
                self._phase = _D_CARRY
                self._step_count = 0
            return make_action(dx, dy, dth, 0, 1.0)

        elif self._phase == _D_CARRY:
            # Carry to drop_x at SAFE_CARRY_HEIGHT (arm stays ARM_PICK)
            dx, dy = move_toward(rx, ry, self._drop_x, SAFE_CARRY_HEIGHT_DROP)
            dth    = rotate_toward(rth)
            done   = abs(rx - self._drop_x) < POS_TOL
            if done:
                self._phase = _D_LOWER_CARRY
                self._step_count = 0
            return make_action(dx, dy, dth, 0, 1.0)

        elif self._phase == _D_LOWER_CARRY:
            # Descend to pick_y at drop_x (arm=ARM_PICK, vac ON)
            # At this robot_y with arm=ARM_PICK: obs_bottom = TABLE_TOP ✓
            dx, dy = move_toward(rx, ry, self._drop_x, self._pick_y)
            dth    = rotate_toward(rth)
            done   = (abs(rx - self._drop_x) < POS_TOL and
                      abs(ry - self._pick_y) < POS_TOL)
            if done:
                self._phase = _D_RELEASE
                self._step_count = 0
            return make_action(dx, dy, dth, 0, 1.0)

        else:  # _D_RELEASE / _D_DONE
            # Release vacuum — obs deposits at table level
            return make_action(0, 0, 0, 0, 0.0)


class PickAndPlace(Behavior):
    """Pick the target block and place it on the target surface."""

    def __init__(self):
        self._phase  = _P_DONE
        self._timer  = 0
        self._step_count = 0
        self._pick_x = 0.0
        self._pick_y = 0.0
        self._place_x = 0.0
        self._place_y = 0.0

    def reset(self, obs):
        self._phase = _P_RISE
        self._timer = 0
        self._step_count = 0
        self._update_targets(obs)

    def _update_targets(self, obs):
        blk  = extract_target_block(obs)
        surf = extract_target_surface(obs)
        self._pick_x  = rect_center_x(blk)
        self._pick_y  = pick_robot_y(rect_top(blk))
        self._place_x = surf['x'] + surf['width'] / 2.0
        self._place_y = place_robot_y(
            surf['y'] + surf['height'],
            blk['height'],
        )

    def initializable(self, obs) -> bool:
        from obs_helpers import any_obstruction_on_surface
        return not any_obstruction_on_surface(obs)

    def terminated(self, obs) -> bool:
        return block_is_on_surface(obs)

    def step(self, obs) -> np.ndarray:
        r = extract_robot(obs)
        rx, ry, rth = r['x'], r['y'], r['theta']
        arm = r['arm_joint']

        self._step_count += 1
        if self._step_count > MAX_PHASE_STEPS:
            self._update_targets(obs)
            if is_block_grasped(obs):
                self._phase = _P_RISE_POST
            else:
                self._phase = _P_RISE
            self._step_count = 0

        if self._phase == _P_RISE:
            dx, dy = move_toward(rx, ry, rx, SAFE_NAV_HEIGHT)
            dth    = rotate_toward(rth)
            darm   = arm_toward(arm, ARM_MIN)
            done   = (ry >= SAFE_NAV_HEIGHT - POS_TOL and
                      abs(arm - ARM_MIN) < ARM_TOL and
                      abs(rth - TARGET_THETA) < THETA_TOL)
            if done:
                self._phase = _P_ALIGN_X
                self._step_count = 0
                self._update_targets(obs)
            return make_action(dx, dy, dth, darm, 0.0)

        elif self._phase == _P_ALIGN_X:
            dx, dy = move_toward(rx, ry, self._pick_x, SAFE_NAV_HEIGHT)
            dth    = rotate_toward(rth)
            darm   = arm_toward(arm, ARM_MIN)
            done   = (abs(rx - self._pick_x) < POS_TOL and
                      ry >= SAFE_NAV_HEIGHT - POS_TOL)
            if done:
                self._phase = _P_LOWER
                self._step_count = 0
            return make_action(dx, dy, dth, darm, 0.0)

        elif self._phase == _P_LOWER:
            self._update_targets(obs)
            dx, dy = move_toward(rx, ry, self._pick_x, self._pick_y)
            dth    = rotate_toward(rth)
            darm   = arm_toward(arm, ARM_MIN)
            done   = (abs(rx - self._pick_x) < POS_TOL and
                      abs(ry - self._pick_y) < POS_TOL)
            if done:
                self._phase = _P_EXTEND
                self._step_count = 0
            return make_action(dx, dy, dth, darm, 0.0)

        elif self._phase == _P_EXTEND:
            dth  = rotate_toward(rth)
            darm = arm_toward(arm, ARM_PICK)
            done = abs(arm - ARM_PICK) < ARM_TOL
            if done:
                self._phase = _P_GRASP
                self._timer = 0
                self._step_count = 0
            return make_action(0, 0, dth, darm, 0.0)

        elif self._phase == _P_GRASP:
            self._timer += 1
            dth  = rotate_toward(rth)
            done = self._timer >= GRASP_WAIT_STEPS
            if done:
                self._phase = _P_RETRACT
                self._step_count = 0
            return make_action(0, 0, dth, 0, 1.0)

        elif self._phase == _P_RETRACT:
            # Retract arm (allowed with vac ON) to lift block
            dth  = rotate_toward(rth)
            darm = arm_toward(arm, ARM_MIN)
            done = abs(arm - ARM_MIN) < ARM_TOL
            if done:
                self._phase = _P_RISE_POST
                self._step_count = 0
            return make_action(0, 0, dth, darm, 1.0)

        elif self._phase == _P_RISE_POST:
            dx, dy = move_toward(rx, ry, rx, SAFE_NAV_HEIGHT)
            dth    = rotate_toward(rth)
            done   = ry >= SAFE_NAV_HEIGHT - POS_TOL
            if done:
                self._phase = _P_CARRY
                self._step_count = 0
                self._update_targets(obs)
            return make_action(dx, dy, dth, 0, 1.0)

        elif self._phase == _P_CARRY:
            dx, dy = move_toward(rx, ry, self._place_x, SAFE_NAV_HEIGHT)
            dth    = rotate_toward(rth)
            done   = abs(rx - self._place_x) < POS_TOL
            if done:
                self._phase = _P_LOWER_PLACE
                self._step_count = 0
            return make_action(dx, dy, dth, 0, 1.0)

        elif self._phase == _P_LOWER_PLACE:
            self._update_targets(obs)
            dx, dy = move_toward(rx, ry, self._place_x, self._place_y)
            dth    = rotate_toward(rth)
            done   = (abs(rx - self._place_x) < POS_TOL and
                      abs(ry - self._place_y) < POS_TOL)
            if done:
                self._phase = _P_EXTEND_PL
                self._step_count = 0
            return make_action(dx, dy, dth, 0, 1.0)

        elif self._phase == _P_EXTEND_PL:
            # NOTE: arm can only extend with vacuum OFF.
            # Release block first, then extend to push into place.
            # Actually: release vacuum at _P_LOWER_PLACE height, block falls onto surface.
            # Then _P_EXTEND_PL just verifies placement.
            # Simpler: just release here (arm is ARM_MIN, block just above surface).
            # The place_y formula already accounts for block height above surface.
            # So just release.
            self._phase = _P_RELEASE
            self._step_count = 0
            return make_action(0, 0, 0, 0, 0.0)

        else:  # _P_RELEASE / _P_DONE
            return make_action(0, 0, 0, 0, 0.0)
