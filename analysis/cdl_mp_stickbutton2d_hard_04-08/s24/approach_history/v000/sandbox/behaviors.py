"""Behavior classes for StickButton2D-b5-v0."""
import numpy as np
from collections import deque

from behavior import Behavior
from obs_helpers import (
    extract_robot, extract_stick, extract_button,
    robot_pos, robot_theta, robot_arm_joint, stick_pos, stick_dims,
    button_pos, gripper_pos, is_button_pressed, all_buttons_pressed,
    get_unpressed_buttons, NUM_BUTTONS, ARM_MAX, ARM_MIN, BUTTON_RADIUS,
    ROBOT_BASE_R, WORLD_MIN_X, WORLD_MAX_X, WORLD_MIN_Y, WORLD_MAX_Y,
)
from act_helpers import (
    clip_action, move_toward, rotate_toward, extend_arm_toward,
    angle_to, robot_state_collision, wrap_angle,
    DX_MAX, DY_MAX, DTHETA_MAX, DARM_MAX, VAC_ON, VAC_OFF,
    POS_TOL, THETA_TOL, ARM_TOL,
    BIRRT_ATTEMPTS, BIRRT_ITERS, BIRRT_SMOOTH, BIRRT_STEP, BIRRT_THETA_W,
    PRESS_OVERSHOOT,
)


# ─────────────────────────────────────────────────────────────────────────────
# BiRRT helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_stick_geom(obs):
    """Return (stk_x, stk_y, stk_theta, half_w, half_h) for collision checks."""
    stk = extract_stick(obs)
    half_w = stk["width"]  / 2.0
    half_h = stk["height"] / 2.0
    return stk["x"], stk["y"], stk["theta"], half_w, half_h


def _make_birrt_fns(obs, rng):
    """Build BiRRT functions for robot (x, y) navigation. Theta not planned."""
    sx, sy, sth, shw, shh = _get_stick_geom(obs)

    def sample_fn(_state):
        x = rng.uniform(WORLD_MIN_X + ROBOT_BASE_R, WORLD_MAX_X - ROBOT_BASE_R)
        y = rng.uniform(WORLD_MIN_Y + ROBOT_BASE_R, WORLD_MAX_Y - ROBOT_BASE_R)
        return np.array([x, y])

    def extend_fn(s1, s2):
        delta = s2 - s1
        dist  = np.linalg.norm(delta)
        if dist < 1e-9:
            return
        n = max(1, int(np.ceil(dist / BIRRT_STEP)))
        for i in range(1, n + 1):
            t = i / n
            yield s1 + t * delta

    def collision_fn(state):
        return robot_state_collision(
            state[0], state[1],
            sx, sy, sth, shw, shh
        )

    def distance_fn(s1, s2):
        return np.linalg.norm(s1 - s2)

    return sample_fn, extend_fn, collision_fn, distance_fn


def _plan_path(obs, target_pos, rng, primitives):
    """Plan collision-free 2-D path for robot base to target_pos."""
    sample_fn, extend_fn, collision_fn, distance_fn = _make_birrt_fns(obs, rng)
    rob    = extract_robot(obs)
    start  = np.array([rob["x"], rob["y"]])
    goal   = np.array(target_pos, dtype=float)

    # Clamp goal to valid bounds
    goal[0] = np.clip(goal[0], WORLD_MIN_X + ROBOT_BASE_R, WORLD_MAX_X - ROBOT_BASE_R)
    goal[1] = np.clip(goal[1], WORLD_MIN_Y + ROBOT_BASE_R, WORLD_MAX_Y - ROBOT_BASE_R)

    if collision_fn(goal):
        # Goal is in collision; find a nearby free point
        for _ in range(30):
            offset = rng.uniform(-0.15, 0.15, size=2)
            cand = goal + offset
            cand[0] = np.clip(cand[0], WORLD_MIN_X + ROBOT_BASE_R, WORLD_MAX_X - ROBOT_BASE_R)
            cand[1] = np.clip(cand[1], WORLD_MIN_Y + ROBOT_BASE_R, WORLD_MAX_Y - ROBOT_BASE_R)
            if not collision_fn(cand):
                goal = cand
                break
        else:
            return None  # couldn't find free goal

    BiRRT = primitives["BiRRT"]
    birrt = BiRRT(sample_fn, extend_fn, collision_fn, distance_fn,
                  rng, BIRRT_ATTEMPTS, BIRRT_ITERS, BIRRT_SMOOTH)
    return birrt.query(start, goal)


# ─────────────────────────────────────────────────────────────────────────────
# Behavior: PressButton – presses ONE button
# Buttons have ZOrder.NONE, so the robot can drive directly onto them.
# ─────────────────────────────────────────────────────────────────────────────

class PressButton(Behavior):
    """Navigate robot base onto button[btn_idx] to press it."""

    def __init__(self, btn_idx):
        self._btn_idx   = btn_idx
        self._primitives = None
        self._rng       = np.random.default_rng(btn_idx + 42)
        self._path      = deque()
        self._phase     = "plan"
        self._stuck_ctr = 0
        self._last_pos  = None

    def set_primitives(self, primitives):
        self._primitives = primitives

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
        # Already pressed?
        if is_button_pressed(obs, self._btn_idx):
            return clip_action(0, 0, 0, 0, VAC_OFF)

        rob_p  = robot_pos(obs)
        btn_p  = button_pos(obs, self._btn_idx)
        rob    = extract_robot(obs)

        # ── Phase: navigate ───────────────────────────────────────────────
        if self._phase == "navigate":
            # Stuck detection
            if self._last_pos is not None:
                if np.linalg.norm(rob_p - self._last_pos) < 0.003:
                    self._stuck_ctr += 1
                else:
                    self._stuck_ctr = 0
            self._last_pos = rob_p.copy()

            if self._stuck_ctr > 40:
                # Re-plan from current position
                self._phase     = "plan"
                self._path.clear()
                self._stuck_ctr = 0
                return clip_action(0, 0, 0, 0, VAC_OFF)

            # Skip already-reached waypoints
            while self._path:
                wp    = self._path[0]
                dist  = np.linalg.norm(rob_p - wp)
                if dist < POS_TOL:
                    self._path.popleft()
                else:
                    break

            if self._path:
                wp     = self._path[0]
                dx, dy = move_toward(rob_p, wp, DX_MAX)
                return clip_action(dx, dy, 0, 0, VAC_OFF)
            else:
                # Path consumed – do a final direct nudge toward button
                dx, dy = move_toward(rob_p, btn_p, DX_MAX)
                return clip_action(dx, dy, 0, 0, VAC_OFF)

        # ── Phase: plan ───────────────────────────────────────────────────
        # Target: robot base overlaps button → just go to button position
        # Aim slightly past centre for robustness
        dist = np.linalg.norm(rob_p - btn_p)

        # Nudge target slightly past button centre in current approach direction
        if dist > 1e-3:
            direction  = (btn_p - rob_p) / dist
            target_pos = btn_p + direction * PRESS_OVERSHOOT
        else:
            target_pos = btn_p.copy()

        if self._primitives is not None:
            path = _plan_path(obs, target_pos, self._rng, self._primitives)
            if path is not None and len(path) > 1:
                self._path = deque(path[1:])
                self._phase = "navigate"
                # Return first step immediately
                wp     = self._path[0] if self._path else target_pos
                dist_wp = np.linalg.norm(rob_p - wp)
                if dist_wp < POS_TOL:
                    self._path.popleft() if self._path else None
                dx, dy = move_toward(rob_p, wp if self._path else target_pos, DX_MAX)
                return clip_action(dx, dy, 0, 0, VAC_OFF)

        # Fallback: direct movement
        dx, dy = move_toward(rob_p, target_pos, DX_MAX)
        self._phase = "navigate"  # try again next step
        return clip_action(dx, dy, 0, 0, VAC_OFF)


# ─────────────────────────────────────────────────────────────────────────────
# Behavior: PressAllButtons
# ─────────────────────────────────────────────────────────────────────────────

class PressAllButtons(Behavior):
    """Press all 5 buttons sequentially."""

    def __init__(self):
        self._sub        = None
        self._primitives = None

    def set_primitives(self, primitives):
        self._primitives = primitives

    def initializable(self, obs):
        return True

    def terminated(self, obs):
        return all_buttons_pressed(obs)

    def reset(self, obs):
        self._sub = None
        self._pick_next(obs)

    def _pick_next(self, obs):
        unpressed = get_unpressed_buttons(obs)
        if not unpressed:
            self._sub = None
            return
        # Pick nearest unpressed button
        rob_p = robot_pos(obs)
        idx   = min(unpressed, key=lambda i: np.linalg.norm(rob_p - button_pos(obs, i)))
        self._sub = PressButton(idx)
        if self._primitives is not None:
            self._sub.set_primitives(self._primitives)
        self._sub.reset(obs)

    def step(self, obs):
        if self._sub is None:
            return np.zeros(5, dtype=np.float32)

        if self._sub.terminated(obs):
            self._pick_next(obs)
            if self._sub is None:
                return np.zeros(5, dtype=np.float32)

        return self._sub.step(obs)
