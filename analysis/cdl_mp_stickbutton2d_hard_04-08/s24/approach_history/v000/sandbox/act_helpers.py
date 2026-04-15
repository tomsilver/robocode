"""All action generation helpers. All magic numbers live here."""
import numpy as np

# ── Action limits ────────────────────────────────────────────────────────────
DX_MAX     = 0.050
DY_MAX     = 0.050
DTHETA_MAX = 0.196   # ~pi/16
DARM_MAX   = 0.100
VAC_ON     = 1.0
VAC_OFF    = 0.0

# ── Controller gains / tolerances ────────────────────────────────────────────
POS_KP    = 1.5    # proportional gain for position control
THETA_KP  = 1.0
ARM_KP    = 1.0

# Tolerances
POS_TOL   = 0.04   # robot position within this of waypoint → advance
THETA_TOL = 0.08   # radians
ARM_TOL   = 0.01

# ── World / robot constants ──────────────────────────────────────────────────
ROBOT_BASE_R  = 0.1
WORLD_MIN_X   = 0.0
WORLD_MAX_X   = 3.5
WORLD_MIN_Y   = 0.0
WORLD_MAX_Y   = 2.5
ARM_MAX       = 0.2
ARM_MIN       = 0.1
BUTTON_RADIUS = 0.05
GRIPPER_WIDTH = 0.01
GRIPPER_HEIGHT = 0.07

# ── Stick collision constants ─────────────────────────────────────────────────
STICK_HALF_W = 0.025   # half-width of stick (stick width = 0.05)
# Extra margin added to stick half-extents for BiRRT collision check
STICK_COLLISION_MARGIN = 0.005

# ── BiRRT parameters ──────────────────────────────────────────────────────────
BIRRT_ATTEMPTS   = 5
BIRRT_ITERS      = 3000
BIRRT_SMOOTH     = 50
BIRRT_STEP       = 0.06   # max position step per extend segment
BIRRT_THETA_W    = 0.05   # weight of theta in distance metric

# ── Grasp parameters ──────────────────────────────────────────────────────────
# Robot center must be at least this far from stick face to avoid collision
GRASP_APPROACH_DIST = 0.14   # robot center to stick narrow-face center
# Robot center must be at most this far for suction zone to enter stick
GRASP_MAX_DIST      = 0.15

# Target press distance: how close robot centre gets to button centre
PRESS_OVERSHOOT = 0.02   # aim INSIDE button centre by this much


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
    """Return (dx, dy) step toward target, capped at max_step."""
    delta = np.array(target, dtype=float) - np.array(pos, dtype=float)
    dist  = np.linalg.norm(delta)
    if dist < 1e-9:
        return 0.0, 0.0
    scale = min(max_step * POS_KP, dist) / dist
    d = delta * scale
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


def point_in_rect(px, py, cx, cy, half_w, half_h, theta):
    """Check if point (px,py) is inside axis-aligned or rotated rectangle."""
    # Transform point to rectangle's local frame
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    lx = cos_t * (px - cx) - sin_t * (py - cy)
    ly = sin_t * (px - cx) + cos_t * (py - cy)
    return abs(lx) <= half_w and abs(ly) <= half_h


def circle_rect_collide(cx, cy, radius, rx, ry, half_w, half_h, theta):
    """Check if circle (cx,cy,radius) intersects rectangle (rx,ry,half_w,half_h,theta)."""
    # Transform circle centre to rectangle local frame
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    lx = cos_t * (cx - rx) - sin_t * (cy - ry)
    ly = sin_t * (cx - rx) + cos_t * (cy - ry)
    # Clamp to rectangle
    nearest_x = np.clip(lx, -half_w, half_w)
    nearest_y = np.clip(ly, -half_h, half_h)
    dist_sq = (lx - nearest_x) ** 2 + (ly - nearest_y) ** 2
    return dist_sq < radius * radius


def robot_collides_with_stick(rx, ry, stk_x, stk_y, stk_theta, stk_half_w, stk_half_h):
    """Return True if robot base circle collides with stick rectangle."""
    return circle_rect_collide(
        rx, ry, ROBOT_BASE_R + STICK_COLLISION_MARGIN,
        stk_x, stk_y, stk_half_w, stk_half_h, stk_theta
    )


def robot_in_bounds(rx, ry):
    """Return True if robot base is within world bounds."""
    return (WORLD_MIN_X + ROBOT_BASE_R <= rx <= WORLD_MAX_X - ROBOT_BASE_R and
            WORLD_MIN_Y + ROBOT_BASE_R <= ry <= WORLD_MAX_Y - ROBOT_BASE_R)


def robot_state_collision(rx, ry, stk_x, stk_y, stk_theta, stk_half_w, stk_half_h):
    """
    True if robot position is in collision.
    NOTE: Buttons do NOT block movement (ZOrder.NONE), only stick does.
    """
    if not robot_in_bounds(rx, ry):
        return True
    if robot_collides_with_stick(rx, ry, stk_x, stk_y, stk_theta, stk_half_w, stk_half_h):
        return True
    return False
