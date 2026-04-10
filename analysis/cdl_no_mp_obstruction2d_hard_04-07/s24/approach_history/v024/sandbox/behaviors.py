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
    SAFE_NAV_HEIGHT, MAX_DY, POS_TOL, ARM_TOL, THETA_TOL,
    GRASP_WAIT_STEPS, MAX_PHASE_STEPS,
)

# Key constraints discovered through testing:
# - ARM can EXTEND only with vacuum OFF
# - ARM can RETRACT with vacuum ON or OFF
#
# PickAndDrop strategy:
# 1. Carry object with ARM RETRACTED (ARM_MIN) — allows arm to have been extended for grasp
# 2. Descend to deposit_y = pick_y - (ARM_PICK - ARM_MIN) at the chosen drop zone
#    → At deposit_y with arm=ARM_MIN: obs_bottom = TABLE_TOP ✓
# 3. Release vacuum → object rests at table level
#
# drop zones are pre-computed to avoid non-surface obstacles (so descent is unblocked)

# ─── Phase constants for PickAndDrop ────────────────────────────────────────
_D_RISE        = 0   # arm→ARM_MIN, θ→TARGET, rise to SAFE_NAV_HEIGHT
_D_ALIGN_X     = 1   # move to pick_x at SAFE_NAV_HEIGHT
_D_LOWER       = 2   # descend to pick_y at pick_x (arm=ARM_MIN)
_D_EXTEND      = 3   # extend arm to ARM_PICK (vacuum OFF — only way to extend)
_D_GRASP       = 4   # vacuum ON, wait GRASP_WAIT_STEPS
_D_RETRACT     = 5   # retract arm to ARM_MIN (vacuum ON — allowed)
_D_RISE_POST   = 6   # rise to SAFE_NAV_HEIGHT (arm=ARM_MIN, vacuum ON)
_D_CARRY       = 7   # navigate to drop_x at SAFE_NAV_HEIGHT
_D_LOWER_CARRY = 8   # descend to deposit_y at drop_x (arm=ARM_MIN, vacuum ON)
                     # deposit_y = pick_y - (ARM_PICK - ARM_MIN) → obs_bottom=TABLE_TOP
_D_REPOSITION  = 11  # rise to SAFE_NAV_HEIGHT then retry CARRY at new drop_x
_D_RELEASE     = 9   # vacuum OFF → obs lands at TABLE_TOP
_D_DONE        = 10

# ─── Phase constants for PickAndPlace ───────────────────────────────────────
_P_RISE        = 0
_P_ALIGN_X     = 1
_P_LOWER       = 2
_P_EXTEND      = 3
_P_GRASP       = 4
_P_RETRACT     = 5   # retract arm to ARM_MIN (vacuum ON — allowed)
_P_RISE_POST   = 6
_P_CARRY       = 7
_P_LOWER_PLACE = 8   # descend to place_y (arm=ARM_MIN, vacuum ON)
_P_RELEASE     = 9   # vacuum OFF → block rests on surface
_P_DONE        = 10


class PickAndDrop(Behavior):
    """Pick obstruction i and deposit it at table level at drop_x.

    Uses distinct zone_idx to pick a unique drop zone.
    Deposit formula: robot lowers to deposit_y = pick_y - (ARM_PICK - ARM_MIN).
    At that height with arm=ARM_MIN: obs_bottom = TABLE_TOP exactly.
    RELEASE: vac=0 + rise, so table holds object while arm moves away.
    """

    def __init__(self, obs_idx: int, zone_idx: int = 0):
        self._obs_idx = obs_idx
        self._zone_idx = zone_idx
        self._drop_x  = 0.0
        self._phase   = _D_DONE
        self._timer   = 0
        self._step_count = 0
        self._pick_x  = 0.0
        self._pick_y  = 0.0
        self._deposit_y = 0.0
        self._prev_ry   = 0.0
        self._stuck_count = 0
        self._shift_attempts = 0  # how many lateral shifts tried during LOWER_CARRY

    def reset(self, obs):
        from obs_helpers import get_drop_zones
        zones = get_drop_zones(obs)
        idx = min(self._zone_idx, len(zones) - 1)
        self._drop_x = zones[idx]
        self._phase = _D_RISE
        self._timer = 0
        self._step_count = 0
        self._prev_ry = 0.0
        self._stuck_count = 0
        self._shift_attempts = 0
        self._update_pick_target(obs)

    def _update_pick_target(self, obs):
        obj = extract_obstruction(obs, self._obs_idx)
        self._pick_x = rect_center_x(obj)
        self._pick_y = pick_robot_y(rect_top(obj))
        # deposit_y: height where obs_bottom = TABLE_TOP with arm=ARM_MIN
        # pick_y = TABLE_TOP + obs_h + ARM_PICK + SUCTION_OFFSET
        # deposit_y = TABLE_TOP + ARM_MIN + obs_h + SUCTION_OFFSET = pick_y - (ARM_PICK - ARM_MIN)
        self._deposit_y = self._pick_y - (ARM_PICK - ARM_MIN)

    def _find_safe_drop_x(self, obs, cur_rx):
        """Find nearest x to cur_rx that clears surface + non-surface obs (robot body)."""
        from obs_helpers import (extract_obstruction, extract_target_surface,
                                 extract_target_block, obstruction_on_surface,
                                 NUM_OBSTRUCTIONS, WORLD_MIN_X, WORLD_MAX_X,
                                 DROP_MARGIN, CARRY_OBJ_HALF_W)
        ROBOT_HALF = 0.10
        CARRY_HALF = CARRY_OBJ_HALF_W
        GAP = 0.02

        # Build exclusion list (mirrors get_drop_zones)
        surf = extract_target_surface(obs)
        sx, sw = surf['x'], surf['width']
        excl = [(sx - DROP_MARGIN - CARRY_HALF, sx + sw + DROP_MARGIN + CARRY_HALF)]
        blk = extract_target_block(obs)
        pick_x = blk['x'] + blk['width'] / 2.0
        excl.append((pick_x - ROBOT_HALF - CARRY_HALF, pick_x + ROBOT_HALF + CARRY_HALF))
        for j in range(NUM_OBSTRUCTIONS):
            if j == self._obs_idx:
                continue
            if not obstruction_on_surface(obs, j):
                o = extract_obstruction(obs, j)
                excl.append((o['x'] - ROBOT_HALF - GAP, o['x'] + o['width'] + ROBOT_HALF + GAP))
        # Exclude current drop_x: we're stuck there, don't return same zone
        excl.append((self._drop_x - 0.20, self._drop_x + 0.20))

        def blocked(x):
            return any(lo <= x <= hi for (lo, hi) in excl)

        def next_clear(x):
            changed = True
            while changed:
                changed = False
                for (lo, hi) in excl:
                    if lo <= x <= hi:
                        x = hi + 0.01
                        changed = True
            return x

        # Try candidates: past each exclusion right/left, and fixed shifts from cur_rx
        candidates = []
        for (lo, hi) in excl:
            candidates.append(hi + 0.01)
            candidates.append(lo - 0.01)
        candidates.append(cur_rx - 0.20)
        candidates.append(cur_rx + 0.25)

        best = None
        for c in candidates:
            c = float(np.clip(c, WORLD_MIN_X + 0.20, WORLD_MAX_X - DROP_MARGIN))
            c = next_clear(c)  # jump past any exclusion
            c = float(np.clip(c, WORLD_MIN_X + 0.20, WORLD_MAX_X - DROP_MARGIN))
            if not blocked(c):
                if best is None or abs(c - cur_rx) < abs(best - cur_rx):
                    best = c
        return best if best is not None else float(np.clip(cur_rx + 0.25, 0.20, 1.40))

    def initializable(self, obs) -> bool:
        return True

    def terminated(self, obs) -> bool:
        return self._phase == _D_DONE

    def step(self, obs) -> np.ndarray:
        r = extract_robot(obs)
        rx, ry, rth = r['x'], r['y'], r['theta']
        arm = r['arm_joint']

        self._step_count += 1
        if self._step_count > MAX_PHASE_STEPS and self._phase != _D_REPOSITION:
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
            # Recompute target in case object moved
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
            # Extend arm (vacuum MUST be OFF for extension)
            dth  = rotate_toward(rth)
            darm = arm_toward(arm, ARM_PICK)
            done = abs(arm - ARM_PICK) < ARM_TOL
            if done:
                self._phase = _D_GRASP
                self._timer = 0
                self._step_count = 0
            return make_action(0, 0, dth, darm, 0.0)

        elif self._phase == _D_GRASP:
            self._timer += 1
            dth  = rotate_toward(rth)
            done = self._timer >= GRASP_WAIT_STEPS
            if done:
                self._phase = _D_RETRACT
                self._step_count = 0
            return make_action(0, 0, dth, 0, 1.0)

        elif self._phase == _D_RETRACT:
            # Retract arm to ARM_MIN (vacuum ON — retraction allowed)
            dth  = rotate_toward(rth)
            darm = arm_toward(arm, ARM_MIN)
            done = abs(arm - ARM_MIN) < ARM_TOL
            if done:
                self._phase = _D_RISE_POST
                self._step_count = 0
            return make_action(0, 0, dth, darm, 1.0)

        elif self._phase == _D_RISE_POST:
            # Rise to SAFE_NAV_HEIGHT while holding (arm=ARM_MIN)
            dx, dy = move_toward(rx, ry, rx, SAFE_NAV_HEIGHT)
            dth    = rotate_toward(rth)
            done   = ry >= SAFE_NAV_HEIGHT - POS_TOL
            if done:
                self._phase = _D_CARRY
                self._step_count = 0
            return make_action(dx, dy, dth, 0, 1.0)

        elif self._phase == _D_CARRY:
            # Navigate to drop_x at SAFE_NAV_HEIGHT
            dx, dy = move_toward(rx, ry, self._drop_x, SAFE_NAV_HEIGHT)
            dth    = rotate_toward(rth)
            done   = (abs(rx - self._drop_x) < POS_TOL and
                      abs(ry - SAFE_NAV_HEIGHT) < POS_TOL)
            if done:
                self._phase = _D_LOWER_CARRY
                self._step_count = 0
            return make_action(dx, dy, dth, 0, 1.0)

        elif self._phase == _D_LOWER_CARRY:
            # Reactive descent: push down at MAX_DY until robot stops moving
            # (floor or previously-stacked objects stop us). This handles stacking.
            dx = float(np.clip((self._drop_x - rx) * 1.0, -0.05, 0.05))
            dth = rotate_toward(rth)
            # Track downward progress
            if self._prev_ry > 0.0:
                if ry > self._prev_ry - 0.001:  # not moving down
                    self._stuck_count += 1
                else:
                    self._stuck_count = 0
            self._prev_ry = ry
            # Also stop if reached deposit_y (normal case, no stacking)
            at_deposit = abs(ry - self._deposit_y) < POS_TOL and abs(rx - self._drop_x) < POS_TOL
            if self._stuck_count >= 5:
                if self._shift_attempts < 3:
                    # Rise first, then navigate to safe zone (can't move through obs laterally)
                    new_x = self._find_safe_drop_x(obs, rx)
                    self._drop_x = new_x
                    self._shift_attempts += 1
                    self._stuck_count = 0
                    self._prev_ry = 0.0
                    self._phase = _D_REPOSITION
                    self._step_count = 0
                else:
                    self._phase = _D_RELEASE
                    self._step_count = 0
                    self._timer = 0
                    self._stuck_count = 0
            elif at_deposit:
                self._phase = _D_RELEASE
                self._step_count = 0
                self._timer = 0
                self._stuck_count = 0
            return make_action(dx, -MAX_DY, dth, 0, 1.0)

        elif self._phase == _D_REPOSITION:
            # Rise to SAFE_NAV_HEIGHT before navigating to new drop_x
            dx, dy = move_toward(rx, ry, self._drop_x, SAFE_NAV_HEIGHT)
            dth = rotate_toward(rth)
            done = (ry >= SAFE_NAV_HEIGHT - POS_TOL and
                    abs(rx - self._drop_x) < POS_TOL)
            if done:
                self._phase = _D_LOWER_CARRY
                self._step_count = 0
                self._prev_ry = 0.0
                self._stuck_count = 0
            elif self._step_count > MAX_PHASE_STEPS:
                # Physically stuck during REPOSITION — release here
                self._phase = _D_RELEASE
                self._step_count = 0
                self._timer = 0
            return make_action(dx, dy, dth, 0, 1.0)

        elif self._phase == _D_RELEASE:
            # Vacuum OFF + rise: table holds object while arm separates
            self._timer += 1
            if self._timer >= GRASP_WAIT_STEPS:
                self._phase = _D_DONE
            return make_action(0, MAX_DY, 0, 0, 0.0)

        else:  # _D_DONE
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
        self._prev_ry   = 0.0
        self._stuck_count = 0

    def reset(self, obs):
        self._phase = _P_RISE
        self._timer = 0
        self._step_count = 0
        self._prev_ry = 0.0
        self._stuck_count = 0
        self._update_targets(obs)

    def _update_targets(self, obs):
        blk  = extract_target_block(obs)
        surf = extract_target_surface(obs)
        self._pick_x  = rect_center_x(blk)
        self._pick_y  = pick_robot_y(rect_top(blk))
        self._place_x = surf['x'] + surf['width'] / 2.0
        # place_y: at this height with arm=ARM_MIN, block bottom = surface top
        self._place_y = place_robot_y(
            surf['y'] + surf['height'],
            blk['height'],
        )

    def initializable(self, obs) -> bool:
        from obs_helpers import any_obstruction_on_surface
        return not any_obstruction_on_surface(obs)

    def terminated(self, obs) -> bool:
        return block_is_on_surface(obs) or self._phase >= _P_DONE

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
            self._prev_ry = 0.0
            self._stuck_count = 0

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
                self._prev_ry = 0.0
                self._stuck_count = 0
            return make_action(dx, dy, dth, 0, 1.0)

        elif self._phase == _P_LOWER_PLACE:
            # Descend to place_y using actual arm_joint for accuracy
            # At place_y: block_bottom = surface_top exactly
            self._update_targets(obs)
            surf = extract_target_surface(obs)
            blk_obs = extract_target_block(obs)
            # Use actual arm_joint (not ARM_MIN) to compute exact place height
            actual_place_y = (surf['y'] + surf['height'] + blk_obs['height']
                              + arm + SUCTION_OFFSET)
            dx, dy = move_toward(rx, ry, self._place_x, actual_place_y)
            dth    = rotate_toward(rth)
            done   = (abs(rx - self._place_x) < POS_TOL and
                      abs(ry - actual_place_y) < POS_TOL)
            if done:
                self._phase = _P_RELEASE
                self._step_count = 0
                self._timer = 0
            return make_action(dx, dy, dth, 0, 1.0)

        elif self._phase == _P_RELEASE:
            # Vacuum OFF + rise: surface holds block while arm separates
            self._timer += 1
            if self._timer >= GRASP_WAIT_STEPS:
                self._phase = _P_DONE
            return make_action(0, MAX_DY, 0, 0, 0.0)

        else:  # _P_DONE
            return make_action(0, 0, 0, 0, 0.0)
