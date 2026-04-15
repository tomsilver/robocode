"""All action generation helpers. All magic numbers live here."""
import numpy as np

# ── Action limits ────────────────────────────────────────────────────────────
DX_MAX     = 0.050
DY_MAX     = 0.050
DTHETA_MAX = 0.196
DARM_MAX   = 0.100
VAC_ON     = 1.0
VAC_OFF    = 0.0

# ── Controller gains / tolerances ────────────────────────────────────────────
POS_KP    = 1.5
THETA_KP  = 1.0
ARM_KP    = 1.0

POS_TOL   = 0.04
THETA_TOL = 0.08
ARM_TOL   = 0.01

# ── World / robot constants ──────────────────────────────────────────────────
ROBOT_BASE_R = 0.1
WORLD_MIN_X  = 0.0
WORLD_MAX_X  = 3.5
WORLD_MIN_Y  = 0.0
WORLD_MAX_Y  = 2.5
TABLE_Y      = 1.25
FLOOR_MAX_Y  = 1.14   # max robot centre y (table_y - base_r - margin)
ARM_MAX      = 0.2
ARM_MIN      = 0.1
BUTTON_RADIUS = 0.05

# ── BiRRT parameters ──────────────────────────────────────────────────────────
BIRRT_ATTEMPTS   = 5
BIRRT_ITERS      = 3000
BIRRT_SMOOTH     = 50
BIRRT_STEP       = 0.06

# ── Stick grasping constants ──────────────────────────────────────────────────
# Robot approaches stick from its right side (x-axis), arm points left (theta=pi)
# Robot x offset from stick right edge (stick_x + stick_width)
GRASP_X_OFFSET  = 0.13   # robot_x = stick_right_x + GRASP_X_OFFSET
# Robot y alignment: align with stick's mid-floor section
# Suction enters stick when robot_x - SUCTION_DIST_FROM_ROBOT < stick_right_x
GRIPPER_WIDTH   = 0.01
SUCTION_WIDTH   = 0.01
SUCTION_OFFSET  = ARM_MIN + GRIPPER_WIDTH + SUCTION_WIDTH / 2  # ≈ 0.115

# Stick x pressing offset: robot_x = button_x + STICK_PRESS_X_OFFSET
# (depends on gripper_to_stick transform captured at grasp time)
STICK_PRESS_X_OFFSET = 0.035

# Safety margins
WORLD_MARGIN  = 0.12   # keep robot center at least this far from world edges


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def clip_action(dx, dy, dtheta, darm, vac):
    return np.array([
        np.clip(dx,     -DX_MAX,     DX_MAX),
        np.clip(dy,     -DY_MAX,     DY_MAX),
        np.clip(dtheta, -DTHETA_MAX, DTHETA_MAX),
        np.clip(darm,   -DARM_MAX,   DARM_MAX),
        np.clip(vac,     VAC_OFF,    VAC_ON),
    ], dtype=np.float32)


def move_toward(pos, target, max_step=DX_MAX):
    delta = np.array(target, dtype=float) - np.array(pos, dtype=float)
    dist  = np.linalg.norm(delta)
    if dist < 1e-9:
        return 0.0, 0.0
    step  = min(max_step * POS_KP, dist)
    d     = delta * (step / dist)
    return float(np.clip(d[0], -max_step, max_step)), float(np.clip(d[1], -max_step, max_step))


def rotate_toward(current, target, max_step=DTHETA_MAX):
    err = wrap_angle(target - current)
    return float(np.clip(err * THETA_KP, -max_step, max_step))


def extend_arm_toward(current, target, max_step=DARM_MAX):
    err = target - current
    return float(np.clip(err * ARM_KP, -max_step, max_step))


def angle_to(from_pos, to_pos):
    delta = np.array(to_pos) - np.array(from_pos)
    return float(np.arctan2(delta[1], delta[0]))


def circle_rect_collide(cx, cy, radius, rx, ry, half_w, half_h, theta):
    """Check if circle (cx,cy,radius) intersects rectangle with center (rx,ry),
    half-extents (half_w, half_h), rotated by theta."""
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    lx = cos_t * (cx - rx) - sin_t * (cy - ry)
    ly = sin_t * (cx - rx) + cos_t * (cy - ry)
    nearest_x = np.clip(lx, -half_w, half_w)
    nearest_y = np.clip(ly, -half_h, half_h)
    return (lx - nearest_x) ** 2 + (ly - nearest_y) ** 2 < radius * radius


def robot_state_collision(rx, ry, stk_x, stk_y, stk_theta, stk_w, stk_h,
                          stick_held=False):
    """
    True if robot position (rx, ry) is in collision.

    Obstacles (with ZOrder interactions):
    - World bounds (walls): ALL z-order → always block
    - Table (y ≥ TABLE_Y): FLOOR z-order → blocks robot base (ALL)
    - Stick (SURFACE z-order): blocks robot base (ALL) when not held

    Buttons have ZOrder.NONE → never block robot movement.
    """
    # World bounds (approximate via simple limits)
    margin = WORLD_MARGIN
    if rx < margin or rx > WORLD_MAX_X - margin:
        return True
    if ry < ROBOT_BASE_R or ry > FLOOR_MAX_Y:
        return True

    # Stick (only when not held)
    if not stick_held:
        # Stick is RectangleType: corner at (stk_x, stk_y), center at:
        stk_cx = stk_x + stk_w / 2
        stk_cy = stk_y + stk_h / 2
        if circle_rect_collide(rx, ry, ROBOT_BASE_R, stk_cx, stk_cy,
                               stk_w / 2, stk_h / 2, stk_theta):
            return True
    return False
