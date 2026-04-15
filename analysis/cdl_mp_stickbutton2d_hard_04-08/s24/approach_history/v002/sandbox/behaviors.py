"""Behavior classes for StickButton2D-b5-v0."""
import numpy as np
from collections import deque

from behavior import Behavior
from obs_helpers import (
    extract_robot, extract_stick, extract_button,
    robot_pos, robot_theta, robot_arm_joint, stick_corner, stick_center,
    button_pos, gripper_pos, is_button_pressed, all_buttons_pressed,
    get_unpressed_buttons, is_button_on_table, vacuum_on,
    NUM_BUTTONS, ARM_MAX, ARM_MIN, BUTTON_RADIUS, ROBOT_BASE_R,
    WORLD_MIN_X, WORLD_MAX_X, WORLD_MIN_Y, WORLD_MAX_Y,
    TABLE_Y, FLOOR_MAX_Y, STICK_WIDTH, STICK_HEIGHT,
    SUCTION_OFFSET, GRIPPER_HEIGHT,
)
from act_helpers import (
    clip_action, move_toward, rotate_toward, extend_arm_toward,
    angle_to, robot_state_collision, wrap_angle,
    DX_MAX, DY_MAX, DTHETA_MAX, DARM_MAX, VAC_ON, VAC_OFF,
    POS_TOL, THETA_TOL, ARM_TOL,
    BIRRT_ATTEMPTS, BIRRT_ITERS, BIRRT_SMOOTH, BIRRT_STEP,
    GRASP_X_OFFSET, SUCTION_OFFSET as ACT_SUCTION_OFFSET,
    STICK_PRESS_X_OFFSET, WORLD_MARGIN,
)


# ─────────────────────────────────────────────────────────────────────────────
# BiRRT helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plan_path_2d(obs, target_pos, rng, primitives, stick_held=False):
    """Plan 2-D (x,y) path for robot base. Stick is excluded from obstacles when held."""
    stk = extract_stick(obs)
    sx, sy, sth = stk["x"], stk["y"], stk["theta"]
    sw, sh = stk["width"], stk["height"]

    rob = extract_robot(obs)
    start = np.array([rob["x"], rob["y"]])
    goal  = np.array(target_pos[:2], dtype=float)

    # Clamp goal
    goal[0] = np.clip(goal[0], WORLD_MARGIN, WORLD_MAX_X - WORLD_MARGIN)
    goal[1] = np.clip(goal[1], ROBOT_BASE_R + 0.01, FLOOR_MAX_Y - 0.01)

    def sample_fn(_):
        x = rng.uniform(WORLD_MARGIN, WORLD_MAX_X - WORLD_MARGIN)
        y = rng.uniform(ROBOT_BASE_R + 0.01, FLOOR_MAX_Y - 0.01)
        return np.array([x, y])

    def extend_fn(s1, s2):
        delta = s2 - s1
        dist  = np.linalg.norm(delta)
        if dist < 1e-9:
            return
        n = max(1, int(np.ceil(dist / BIRRT_STEP)))
        for i in range(1, n + 1):
            yield s1 + (i / n) * delta

    def collision_fn(state):
        return robot_state_collision(
            state[0], state[1], sx, sy, sth, sw, sh, stick_held
        )

    def distance_fn(s1, s2):
        return np.linalg.norm(s1 - s2)

    BiRRT = primitives["BiRRT"]
    birrt = BiRRT(sample_fn, extend_fn, collision_fn, distance_fn,
                  rng, BIRRT_ATTEMPTS, BIRRT_ITERS, BIRRT_SMOOTH)

    if collision_fn(goal):
        # Try nearby free positions
        for _ in range(50):
            offset = rng.uniform(-0.2, 0.2, size=2)
            cand = goal + offset
            cand[0] = np.clip(cand[0], WORLD_MARGIN, WORLD_MAX_X - WORLD_MARGIN)
            cand[1] = np.clip(cand[1], ROBOT_BASE_R + 0.01, FLOOR_MAX_Y - 0.01)
            if not collision_fn(cand):
                goal = cand
                break

    return birrt.query(start, goal)


def _follow_path(rob_p, path, stuck_ctr):
    """Advance path, return (dx, dy, path, stuck_ctr)."""
    while path and np.linalg.norm(rob_p - path[0]) < POS_TOL:
        path.popleft()
    if path:
        wp     = path[0]
        dx, dy = move_toward(rob_p, wp, DX_MAX)
        return dx, dy, path, stuck_ctr
    return 0.0, 0.0, path, stuck_ctr


# ─────────────────────────────────────────────────────────────────────────────
# Behavior: PressFloorButton
# Presses one button that is on the floor (y < TABLE_Y) by driving robot base
# directly onto it (buttons have ZOrder.NONE so they don't block movement).
# ─────────────────────────────────────────────────────────────────────────────

class PressFloorButton(Behavior):

    def __init__(self, btn_idx):
        self._btn_idx    = btn_idx
        self._primitives = None
        self._rng        = np.random.default_rng(btn_idx + 1)
        self._path       = deque()
        self._phase      = "plan"
        self._stuck_ctr  = 0
        self._last_pos   = None

    def set_primitives(self, p):
        self._primitives = p

    def initializable(self, obs):
        return not is_button_pressed(obs, self._btn_idx)

    def terminated(self, obs):
        return is_button_pressed(obs, self._btn_idx)

    def reset(self, obs):
        self._path.clear()
        self._phase     = "plan"
        self._stuck_ctr = 0
        self._last_pos  = robot_pos(obs).copy()

    def step(self, obs):
        if is_button_pressed(obs, self._btn_idx):
            return clip_action(0, 0, 0, 0, VAC_OFF)

        rob_p  = robot_pos(obs)
        btn_p  = button_pos(obs, self._btn_idx)
        rob    = extract_robot(obs)

        # Stuck detection
        if self._last_pos is not None:
            moved = np.linalg.norm(rob_p - self._last_pos)
            self._stuck_ctr = 0 if moved > 0.003 else self._stuck_ctr + 1
        self._last_pos = rob_p.copy()

        if self._stuck_ctr > 40:
            self._phase     = "plan"
            self._path.clear()
            self._stuck_ctr = 0

        if self._phase == "navigate" and self._path:
            dx, dy, self._path, self._stuck_ctr = _follow_path(
                rob_p, self._path, self._stuck_ctr
            )
            if self._path:
                return clip_action(dx, dy, 0, 0, VAC_OFF)
            # Path consumed; do final nudge
            dx, dy = move_toward(rob_p, btn_p, DX_MAX)
            return clip_action(dx, dy, 0, 0, VAC_OFF)

        if self._phase == "navigate" and not self._path:
            # Done navigating, nudge toward button
            dx, dy = move_toward(rob_p, btn_p, DX_MAX)
            return clip_action(dx, dy, 0, 0, VAC_OFF)

        # Phase: plan
        # Target: robot centre directly on button (floor buttons only)
        btn_y_clamp = min(btn_p[1], FLOOR_MAX_Y - 0.01)
        target = np.array([btn_p[0], btn_y_clamp])

        if self._primitives is not None:
            path = _plan_path_2d(obs, target, self._rng, self._primitives,
                                 stick_held=False)
            if path and len(path) > 1:
                self._path  = deque(path[1:])
                self._phase = "navigate"
                # Return first step immediately
                dx, dy, self._path, _ = _follow_path(rob_p, self._path, 0)
                return clip_action(dx, dy, 0, 0, VAC_OFF)

        # Fallback: direct
        dx, dy = move_toward(rob_p, target, DX_MAX)
        self._phase = "navigate"
        return clip_action(dx, dy, 0, 0, VAC_OFF)


# ─────────────────────────────────────────────────────────────────────────────
# Behavior: GraspStick
# Navigate robot to position where suction zone enters stick, turn vacuum on.
# Approach: from BELOW the stick, arm pointing UP (theta=pi/2).
#
# Geometry:
#   SUCTION_OFFSET = ARM_MIN + GRIPPER_WIDTH + SUCTION_WIDTH/2 = 0.115
#   target_y = stk_y - GRASP_BELOW_OFFSET   (e.g. 0.108)
#   suction_y = target_y + 0.115 = stk_y + 0.007  → inside stick ✓
#   robot_base distance to stick bottom = GRASP_BELOW_OFFSET = 0.108 > 0.1 ✓
#
# This ensures stick stays vertical (theta=0) when arm is later at pi/2
# for pressing table buttons.
# ─────────────────────────────────────────────────────────────────────────────

GRASP_ARM_THETA    = np.pi / 2   # arm points UP
GRASP_BELOW_OFFSET = 0.108       # robot_y = stk_y - GRASP_BELOW_OFFSET
GRASP_ARM_JOINT    = ARM_MIN     # arm at minimum extension

class GraspStick(Behavior):

    def __init__(self):
        self._primitives = None
        self._rng        = np.random.default_rng(99)
        self._path       = deque()
        self._phase      = "plan"
        self._stuck_ctr  = 0
        self._last_pos   = None

    def set_primitives(self, p):
        self._primitives = p

    def initializable(self, obs):
        return not vacuum_on(obs)

    def terminated(self, obs):
        return vacuum_on(obs)

    def reset(self, obs):
        self._path.clear()
        self._phase     = "plan"
        self._stuck_ctr = 0
        self._last_pos  = robot_pos(obs).copy()

    def _grasp_target(self, obs):
        """Return (target_xy, arm_theta) for grasping the stick from below."""
        stk = extract_stick(obs)
        # Approach from below: robot just below stick bottom, arm pointing up
        target_y = stk["y"] - GRASP_BELOW_OFFSET
        target_y = np.clip(target_y, ROBOT_BASE_R + 0.01, FLOOR_MAX_Y - 0.01)
        # X: align with stick center width
        target_x = stk["x"] + stk["width"] / 2.0
        target_x = np.clip(target_x, WORLD_MARGIN, WORLD_MAX_X - WORLD_MARGIN)
        return np.array([target_x, target_y]), GRASP_ARM_THETA

    def step(self, obs):
        if vacuum_on(obs):
            return clip_action(0, 0, 0, 0, VAC_ON)

        rob    = extract_robot(obs)
        rob_p  = robot_pos(obs)
        target, arm_theta = self._grasp_target(obs)

        # Stuck detection
        if self._last_pos is not None:
            moved = np.linalg.norm(rob_p - self._last_pos)
            self._stuck_ctr = 0 if moved > 0.003 else self._stuck_ctr + 1
        self._last_pos = rob_p.copy()

        if self._stuck_ctr > 40:
            self._phase     = "plan"
            self._path.clear()
            self._stuck_ctr = 0

        pos_err   = np.linalg.norm(rob_p - target)
        theta_err = abs(wrap_angle(rob["theta"] - arm_theta))

        # ── Phase: activate vacuum ────────────────────────────────────────
        if self._phase == "vacuum":
            return clip_action(0, 0, 0, 0, VAC_ON)

        # ── Phase: orient arm ─────────────────────────────────────────────
        if self._phase == "orient":
            dtheta = rotate_toward(rob["theta"], arm_theta)
            if theta_err < THETA_TOL:
                self._phase = "vacuum"
            return clip_action(0, 0, dtheta, 0, VAC_OFF)

        # ── Phase: navigate ───────────────────────────────────────────────
        if self._phase == "navigate" and self._path:
            dx, dy, self._path, _ = _follow_path(rob_p, self._path, 0)
            if not self._path:
                self._phase = "orient"
            return clip_action(dx, dy, 0, 0, VAC_OFF)

        if self._phase == "navigate" and not self._path:
            self._phase = "orient"
            return clip_action(0, 0, 0, 0, VAC_OFF)

        # ── Phase: plan ───────────────────────────────────────────────────
        if pos_err < POS_TOL * 2:
            self._phase = "orient"
            return clip_action(0, 0, 0, 0, VAC_OFF)

        if self._primitives is not None:
            # NOTE: stick is an obstacle here (not yet held)
            path = _plan_path_2d(obs, target, self._rng, self._primitives,
                                 stick_held=False)
            if path and len(path) > 1:
                self._path  = deque(path[1:])
                self._phase = "navigate"
                dx, dy, self._path, _ = _follow_path(rob_p, self._path, 0)
                return clip_action(dx, dy, 0, 0, VAC_OFF)

        # Fallback
        dx, dy = move_toward(rob_p, target, DX_MAX)
        self._phase = "navigate"
        return clip_action(dx, dy, 0, 0, VAC_OFF)


# ─────────────────────────────────────────────────────────────────────────────
# Behavior: PressTableButton
# Presses one button that is on the table (y ≥ TABLE_Y) using the held stick.
# Strategy: navigate robot to (button_x + OFFSET, FLOOR_MAX_Y - ε),
#           arm pointing UP (theta=pi/2), stick extends over button.
# ─────────────────────────────────────────────────────────────────────────────

TABLE_BTN_ARM_THETA = np.pi / 2   # arm points up toward table
TABLE_BTN_Y_TARGET  = FLOOR_MAX_Y - 0.02   # robot y near table edge

# From below-grasp geometry (arm=pi/2, grasped from below stick center):
# suction_to_corner_x = stk_x - (stk_x + stk_w/2) = -stk_w/2 = -0.025
# Stick x spans [robot_x - 0.025, robot_x + 0.025], centered on robot_x.
# So robot_x = button_x centers stick on button.
TABLE_BTN_X_OFFSET  = 0.0

class PressTableButton(Behavior):

    def __init__(self, btn_idx):
        self._btn_idx    = btn_idx
        self._primitives = None
        self._rng        = np.random.default_rng(btn_idx + 200)
        self._path       = deque()
        self._phase      = "plan"
        self._stuck_ctr  = 0
        self._last_pos   = None

    def set_primitives(self, p):
        self._primitives = p

    def initializable(self, obs):
        return (not is_button_pressed(obs, self._btn_idx)) and vacuum_on(obs)

    def terminated(self, obs):
        return is_button_pressed(obs, self._btn_idx)

    def reset(self, obs):
        self._path.clear()
        self._phase     = "plan"
        self._stuck_ctr = 0
        self._last_pos  = robot_pos(obs).copy()

    def _target_pos(self, obs):
        btn_p = button_pos(obs, self._btn_idx)
        tx = btn_p[0] + TABLE_BTN_X_OFFSET
        tx = np.clip(tx, WORLD_MARGIN, WORLD_MAX_X - WORLD_MARGIN)
        ty = TABLE_BTN_Y_TARGET
        return np.array([tx, ty])

    def step(self, obs):
        if is_button_pressed(obs, self._btn_idx):
            return clip_action(0, 0, 0, 0, VAC_ON)

        rob   = extract_robot(obs)
        rob_p = robot_pos(obs)
        target = self._target_pos(obs)

        # Stuck detection
        if self._last_pos is not None:
            moved = np.linalg.norm(rob_p - self._last_pos)
            self._stuck_ctr = 0 if moved > 0.003 else self._stuck_ctr + 1
        self._last_pos = rob_p.copy()

        if self._stuck_ctr > 50:
            self._phase     = "plan"
            self._path.clear()
            self._stuck_ctr = 0

        pos_err   = np.linalg.norm(rob_p - target)
        theta_err = abs(wrap_angle(rob["theta"] - TABLE_BTN_ARM_THETA))

        # ── Phase: press (arm already up, extend it) ──────────────────────
        if self._phase == "press":
            dtheta = rotate_toward(rob["theta"], TABLE_BTN_ARM_THETA)
            # Keep arm extended
            darm = extend_arm_toward(rob["arm_joint"], ARM_MAX)
            # Nudge toward target to ensure overlap
            dx, dy = move_toward(rob_p, target, DX_MAX * 0.5)
            return clip_action(dx, dy, dtheta, darm, VAC_ON)

        # ── Phase: orient + extend ────────────────────────────────────────
        if self._phase == "orient":
            dtheta = rotate_toward(rob["theta"], TABLE_BTN_ARM_THETA)
            darm   = extend_arm_toward(rob["arm_joint"], ARM_MAX)
            if theta_err < THETA_TOL and abs(rob["arm_joint"] - ARM_MAX) < ARM_TOL * 2:
                self._phase = "press"
            return clip_action(0, 0, dtheta, darm, VAC_ON)

        # ── Phase: navigate ───────────────────────────────────────────────
        if self._phase == "navigate" and self._path:
            dx, dy, self._path, _ = _follow_path(rob_p, self._path, 0)
            if not self._path:
                self._phase = "orient"
            return clip_action(dx, dy, 0, 0, VAC_ON)

        if self._phase == "navigate" and not self._path:
            self._phase = "orient"
            return clip_action(0, 0, 0, 0, VAC_ON)

        # ── Phase: plan ───────────────────────────────────────────────────
        if pos_err < POS_TOL * 2:
            self._phase = "orient"
            return clip_action(0, 0, 0, 0, VAC_ON)

        if self._primitives is not None:
            # Stick is held: exclude from obstacles
            path = _plan_path_2d(obs, target, self._rng, self._primitives,
                                 stick_held=True)
            if path and len(path) > 1:
                self._path  = deque(path[1:])
                self._phase = "navigate"
                dx, dy, self._path, _ = _follow_path(rob_p, self._path, 0)
                return clip_action(dx, dy, 0, 0, VAC_ON)

        dx, dy = move_toward(rob_p, target, DX_MAX)
        self._phase = "navigate"
        return clip_action(dx, dy, 0, 0, VAC_ON)


# ─────────────────────────────────────────────────────────────────────────────
# Behavior: PressAllButtons
# Orchestrates: floor buttons first, then grasp stick, then table buttons.
# ─────────────────────────────────────────────────────────────────────────────

class PressAllButtons(Behavior):

    def __init__(self):
        self._sub        = None
        self._primitives = None

    def set_primitives(self, p):
        self._primitives = p

    def initializable(self, obs):
        return True

    def terminated(self, obs):
        return all_buttons_pressed(obs)

    def reset(self, obs):
        self._sub = None
        self._pick_next(obs)

    def _make_sub(self, cls, *args):
        sub = cls(*args)
        if self._primitives:
            sub.set_primitives(self._primitives)
        return sub

    def _pick_next(self, obs):
        unpressed = get_unpressed_buttons(obs)
        if not unpressed:
            self._sub = None
            return

        rob_p = robot_pos(obs)

        # Separate floor and table buttons
        floor_btns = [i for i in unpressed if not is_button_on_table(obs, i)]
        table_btns = [i for i in unpressed if is_button_on_table(obs, i)]

        if floor_btns:
            # Press nearest floor button
            idx = min(floor_btns, key=lambda i: np.linalg.norm(rob_p - button_pos(obs, i)))
            sub = self._make_sub(PressFloorButton, idx)
            sub.reset(obs)
            self._sub = sub
            return

        if table_btns:
            # Need stick for table buttons
            if not vacuum_on(obs):
                sub = self._make_sub(GraspStick)
                sub.reset(obs)
                self._sub = sub
                return
            # Press nearest table button with stick
            idx = min(table_btns, key=lambda i: np.linalg.norm(rob_p - button_pos(obs, i)))
            sub = self._make_sub(PressTableButton, idx)
            sub.reset(obs)
            self._sub = sub
            return

        self._sub = None

    def step(self, obs):
        if self._sub is None:
            return np.zeros(5, dtype=np.float32)

        # Advance sub-behavior when it terminates
        if self._sub.terminated(obs):
            self._pick_next(obs)
            if self._sub is None:
                return np.zeros(5, dtype=np.float32)

        return self._sub.step(obs)
