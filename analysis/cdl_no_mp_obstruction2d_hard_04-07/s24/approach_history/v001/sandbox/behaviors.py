"""All behavior classes. Each implements Behavior from behavior.py."""
import numpy as np
from behavior import Behavior
from obs_helpers import (
    extract_robot, extract_target_surface, extract_target_block,
    extract_obstruction, rect_center_x, rect_top,
    obstruction_on_surface, is_block_grasped, block_is_on_surface,
    SUCTION_OFFSET, NUM_OBSTRUCTIONS,
)
from act_helpers import (
    make_action, move_toward, rotate_toward, arm_toward,
    pick_robot_y, place_robot_y,
    ARM_MIN, ARM_PICK, TARGET_THETA,
    SAFE_NAV_HEIGHT, POS_TOL, ARM_TOL, THETA_TOL,
    GRASP_WAIT_STEPS, MAX_PHASE_STEPS,
)


# ─── Phase constants (shared) ────────────────────────────────────────────────
_PH_RISE        = 0   # retract arm, align θ, rise to safe height
_PH_ALIGN_X     = 1   # move horizontally to target x
_PH_LOWER       = 2   # descend to pick/place height
_PH_EXTEND      = 3   # extend arm to ARM_PICK
_PH_GRASP       = 4   # enable vacuum, count wait steps
_PH_RETRACT     = 5   # retract arm to ARM_MIN (lifts object)
_PH_RISE_POST   = 6   # rise back to safe height while holding
_PH_CARRY       = 7   # navigate to drop/place x at safe height
_PH_LOWER_PLACE = 8   # descend to place height (PickAndPlace only)
_PH_EXTEND_PL   = 9   # extend arm again at place height
_PH_RELEASE     = 10  # vacuum off
_PH_DONE        = 11


class PickAndDrop(Behavior):
    """Pick obstruction i and drop it at drop_x."""

    def __init__(self, obs_idx: int, drop_x: float):
        self._obs_idx = obs_idx   # 0-3
        self._drop_x  = drop_x
        self._phase   = _PH_DONE
        self._timer   = 0
        self._step_count = 0
        # Store pick targets computed at reset time
        self._pick_x  = 0.0
        self._pick_y  = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────────
    def reset(self, obs):
        self._phase = _PH_RISE
        self._timer = 0
        self._step_count = 0
        self._update_pick_target(obs)

    def _update_pick_target(self, obs):
        obj = extract_obstruction(obs, self._obs_idx)
        self._pick_x = rect_center_x(obj)
        self._pick_y = pick_robot_y(rect_top(obj))

    # ── predicates ───────────────────────────────────────────────────────────
    def initializable(self, obs) -> bool:
        return True  # always safe to start; will be skipped if already terminated

    def terminated(self, obs) -> bool:
        return not obstruction_on_surface(obs, self._obs_idx)

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, obs) -> np.ndarray:
        r = extract_robot(obs)
        rx, ry, rth = r['x'], r['y'], r['theta']
        arm = r['arm_joint']

        self._step_count += 1
        # Safety: if stuck in a phase too long, recompute target and retry
        if self._step_count > MAX_PHASE_STEPS:
            self._update_pick_target(obs)
            self._phase = _PH_RISE
            self._step_count = 0

        vac = 0.0  # vacuum off by default

        if self._phase == _PH_RISE:
            # Retract arm, align θ, rise to safe height
            dx, dy = move_toward(rx, ry, rx, SAFE_NAV_HEIGHT)
            dth    = rotate_toward(rth)
            darm   = arm_toward(arm, ARM_MIN)
            done   = (ry >= SAFE_NAV_HEIGHT - POS_TOL and
                      abs(arm - ARM_MIN) < ARM_TOL and
                      abs(rth - TARGET_THETA) < THETA_TOL)
            if done:
                self._phase = _PH_ALIGN_X
                self._step_count = 0
            return make_action(dx, dy, dth, darm, vac)

        elif self._phase == _PH_ALIGN_X:
            # Retract arm, align θ, move to pick x at safe height
            dx, dy = move_toward(rx, ry, self._pick_x, SAFE_NAV_HEIGHT)
            dth    = rotate_toward(rth)
            darm   = arm_toward(arm, ARM_MIN)
            done   = (abs(rx - self._pick_x) < POS_TOL and
                      ry >= SAFE_NAV_HEIGHT - POS_TOL)
            if done:
                self._phase = _PH_LOWER
                self._step_count = 0
            return make_action(dx, dy, dth, darm, vac)

        elif self._phase == _PH_LOWER:
            # Descend to pick height (arm retracted)
            # Recompute in case object moved
            self._update_pick_target(obs)
            dx, dy = move_toward(rx, ry, self._pick_x, self._pick_y)
            dth    = rotate_toward(rth)
            darm   = arm_toward(arm, ARM_MIN)
            done   = (abs(rx - self._pick_x) < POS_TOL and
                      abs(ry - self._pick_y) < POS_TOL)
            if done:
                self._phase = _PH_EXTEND
                self._step_count = 0
            return make_action(dx, dy, dth, darm, vac)

        elif self._phase == _PH_EXTEND:
            # Extend arm to ARM_PICK
            dth  = rotate_toward(rth)
            darm = arm_toward(arm, ARM_PICK)
            done = abs(arm - ARM_PICK) < ARM_TOL
            if done:
                self._phase = _PH_GRASP
                self._timer = 0
                self._step_count = 0
            return make_action(0, 0, dth, darm, vac)

        elif self._phase == _PH_GRASP:
            # Vacuum on, wait
            self._timer += 1
            dth  = rotate_toward(rth)
            done = self._timer >= GRASP_WAIT_STEPS
            if done:
                self._phase = _PH_RETRACT
                self._step_count = 0
            return make_action(0, 0, dth, 0, 1.0)

        elif self._phase == _PH_RETRACT:
            # Retract arm (lifts object)
            dth  = rotate_toward(rth)
            darm = arm_toward(arm, ARM_MIN)
            done = abs(arm - ARM_MIN) < ARM_TOL
            if done:
                self._phase = _PH_RISE_POST
                self._step_count = 0
            return make_action(0, 0, dth, darm, 1.0)

        elif self._phase == _PH_RISE_POST:
            # Rise to safe height (still holding)
            dx, dy = move_toward(rx, ry, rx, SAFE_NAV_HEIGHT)
            dth    = rotate_toward(rth)
            darm   = arm_toward(arm, ARM_MIN)
            done   = ry >= SAFE_NAV_HEIGHT - POS_TOL
            if done:
                self._phase = _PH_CARRY
                self._step_count = 0
            return make_action(dx, dy, dth, darm, 1.0)

        elif self._phase == _PH_CARRY:
            # Navigate to drop x (still holding)
            dx, dy = move_toward(rx, ry, self._drop_x, SAFE_NAV_HEIGHT)
            dth    = rotate_toward(rth)
            done   = abs(rx - self._drop_x) < POS_TOL
            if done:
                self._phase = _PH_RELEASE
                self._step_count = 0
            return make_action(dx, dy, dth, 0, 1.0)

        else:  # _PH_RELEASE / _PH_DONE
            # Drop
            return make_action(0, 0, 0, 0, 0.0)


class PickAndPlace(Behavior):
    """Pick the target block and place it on the target surface."""

    def __init__(self):
        self._phase  = _PH_DONE
        self._timer  = 0
        self._step_count = 0
        self._pick_x = 0.0
        self._pick_y = 0.0
        self._place_x = 0.0
        self._place_y = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────────
    def reset(self, obs):
        self._phase = _PH_RISE
        self._timer = 0
        self._step_count = 0
        self._update_targets(obs)

    def _update_targets(self, obs):
        blk  = extract_target_block(obs)
        surf = extract_target_surface(obs)
        self._pick_x  = rect_center_x(blk)
        self._pick_y  = pick_robot_y(rect_top(blk))
        # Place: center block over surface
        self._place_x = surf['x'] + surf['width'] / 2.0
        self._place_y = place_robot_y(
            surf['y'] + surf['height'],
            blk['height'],
        )

    # ── predicates ───────────────────────────────────────────────────────────
    def initializable(self, obs) -> bool:
        # Surface must be clear of obstructions
        from obs_helpers import any_obstruction_on_surface
        return not any_obstruction_on_surface(obs)

    def terminated(self, obs) -> bool:
        return block_is_on_surface(obs)

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, obs) -> np.ndarray:
        r = extract_robot(obs)
        rx, ry, rth = r['x'], r['y'], r['theta']
        arm = r['arm_joint']

        self._step_count += 1
        if self._step_count > MAX_PHASE_STEPS:
            self._update_targets(obs)
            # If block is already grasped, skip to place phase
            if is_block_grasped(obs):
                self._phase = _PH_RISE_POST
            else:
                self._phase = _PH_RISE
            self._step_count = 0

        if self._phase == _PH_RISE:
            # Retract arm, align θ, rise
            dx, dy = move_toward(rx, ry, rx, SAFE_NAV_HEIGHT)
            dth    = rotate_toward(rth)
            darm   = arm_toward(arm, ARM_MIN)
            done   = (ry >= SAFE_NAV_HEIGHT - POS_TOL and
                      abs(arm - ARM_MIN) < ARM_TOL and
                      abs(rth - TARGET_THETA) < THETA_TOL)
            if done:
                self._phase = _PH_ALIGN_X
                self._step_count = 0
                self._update_targets(obs)  # refresh block position
            return make_action(dx, dy, dth, darm, 0.0)

        elif self._phase == _PH_ALIGN_X:
            # Move to block center x at safe height
            dx, dy = move_toward(rx, ry, self._pick_x, SAFE_NAV_HEIGHT)
            dth    = rotate_toward(rth)
            darm   = arm_toward(arm, ARM_MIN)
            done   = (abs(rx - self._pick_x) < POS_TOL and
                      ry >= SAFE_NAV_HEIGHT - POS_TOL)
            if done:
                self._phase = _PH_LOWER
                self._step_count = 0
            return make_action(dx, dy, dth, darm, 0.0)

        elif self._phase == _PH_LOWER:
            # Lower to pick height
            self._update_targets(obs)
            dx, dy = move_toward(rx, ry, self._pick_x, self._pick_y)
            dth    = rotate_toward(rth)
            darm   = arm_toward(arm, ARM_MIN)
            done   = (abs(rx - self._pick_x) < POS_TOL and
                      abs(ry - self._pick_y) < POS_TOL)
            if done:
                self._phase = _PH_EXTEND
                self._step_count = 0
            return make_action(dx, dy, dth, darm, 0.0)

        elif self._phase == _PH_EXTEND:
            dth  = rotate_toward(rth)
            darm = arm_toward(arm, ARM_PICK)
            done = abs(arm - ARM_PICK) < ARM_TOL
            if done:
                self._phase = _PH_GRASP
                self._timer = 0
                self._step_count = 0
            return make_action(0, 0, dth, darm, 0.0)

        elif self._phase == _PH_GRASP:
            self._timer += 1
            dth  = rotate_toward(rth)
            done = self._timer >= GRASP_WAIT_STEPS
            if done:
                self._phase = _PH_RETRACT
                self._step_count = 0
            return make_action(0, 0, dth, 0, 1.0)

        elif self._phase == _PH_RETRACT:
            dth  = rotate_toward(rth)
            darm = arm_toward(arm, ARM_MIN)
            done = abs(arm - ARM_MIN) < ARM_TOL
            if done:
                self._phase = _PH_RISE_POST
                self._step_count = 0
            return make_action(0, 0, dth, darm, 1.0)

        elif self._phase == _PH_RISE_POST:
            # Rise to safe height while holding block
            dx, dy = move_toward(rx, ry, rx, SAFE_NAV_HEIGHT)
            dth    = rotate_toward(rth)
            done   = ry >= SAFE_NAV_HEIGHT - POS_TOL
            if done:
                self._phase = _PH_CARRY
                self._step_count = 0
                self._update_targets(obs)  # refresh surface/block dims
            return make_action(dx, dy, dth, 0, 1.0)

        elif self._phase == _PH_CARRY:
            # Navigate to place x at safe height
            dx, dy = move_toward(rx, ry, self._place_x, SAFE_NAV_HEIGHT)
            dth    = rotate_toward(rth)
            done   = abs(rx - self._place_x) < POS_TOL
            if done:
                self._phase = _PH_LOWER_PLACE
                self._step_count = 0
            return make_action(dx, dy, dth, 0, 1.0)

        elif self._phase == _PH_LOWER_PLACE:
            # Descend to place height
            self._update_targets(obs)
            dx, dy = move_toward(rx, ry, self._place_x, self._place_y)
            dth    = rotate_toward(rth)
            done   = (abs(rx - self._place_x) < POS_TOL and
                      abs(ry - self._place_y) < POS_TOL)
            if done:
                self._phase = _PH_EXTEND_PL
                self._step_count = 0
            return make_action(dx, dy, dth, 0, 1.0)

        elif self._phase == _PH_EXTEND_PL:
            # Extend arm to ARM_PICK (lowers block to surface)
            dth  = rotate_toward(rth)
            darm = arm_toward(arm, ARM_PICK)
            done = abs(arm - ARM_PICK) < ARM_TOL
            if done:
                self._phase = _PH_RELEASE
                self._step_count = 0
            return make_action(0, 0, dth, darm, 1.0)

        else:  # _PH_RELEASE / _PH_DONE
            return make_action(0, 0, 0, 0, 0.0)
